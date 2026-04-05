"""
Transformer last-layer pooling with *subword* TF–IDF weights.
Uses the same WordPiece tokenizer as sentence-transformers/all-MiniLM-L6-v2.

For each span:
  M = number of non-special subword positions (after truncation).
  For token type t: tf(t) = count(t) / M.
  w_j at position j with type t_j:  w_j ∝ tf(t_j) * idf[t_j], normalized to sum_j w_j = 1.
  Pool:  h_bar = sum_j w_j * h_j  then L2-normalize.

IDF is computed over the *same* collection of spans (chunk = document).
"""

from __future__ import annotations

import numpy as np
import torch
from collections import Counter
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None
_model = None


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_tok_model(device: torch.device | None = None):
    global _tokenizer, _model
    if device is None:
        device = default_device()
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    _model.to(device)
    return _tokenizer, _model, device


def compute_subword_idf(texts: list, max_length: int = 256) -> np.ndarray:
    """
    Document = one span string. df[token_id] = number of spans containing that id.
    Returns idf vector (vocab_size,) float64: log((N+1)/(df+1)) + 1
    """
    tok, _, _ = _get_tok_model()
    vs = int(tok.vocab_size)
    df = np.zeros(vs, dtype=np.int64)
    exclude = set()
    for x in (tok.pad_token_id, tok.cls_token_id, tok.sep_token_id, tok.unk_token_id):
        if x is not None:
            exclude.add(int(x))
    N = len(texts)
    if N == 0:
        return np.ones(vs, dtype=np.float64)
    for text in texts:
        enc = tok(
            str(text),
            add_special_tokens=True,
            max_length=int(max_length),
            truncation=True,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        seen = set(int(i) for i in ids) - exclude
        for i in seen:
            if 0 <= i < vs:
                df[i] += 1
    idf = np.log((float(N) + 1.0) / (df.astype(np.float64) + 1.0)) + 1.0
    return idf


def _weights_for_span(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    idf_np: np.ndarray,
    cls_id: int,
    sep_id: int,
    pad_id: int | None,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Returns (indices_j, w_j normalized) over valid subword positions, or (None, None)."""
    L = int(input_ids.shape[0])
    mask = attention_mask.astype(bool)
    valid_j = []
    valid_t = []
    for j in range(L):
        if not mask[j]:
            continue
        tid = int(input_ids[j])
        if tid == cls_id or tid == sep_id or (pad_id is not None and tid == pad_id):
            continue
        valid_j.append(j)
        valid_t.append(tid)
    if not valid_j:
        return None, None
    M = len(valid_t)
    cnt = Counter(valid_t)
    raw = []
    for tid in valid_t:
        raw.append((cnt[tid] / float(M)) * float(idf_np[tid]))
    Z = float(sum(raw))
    if Z < 1e-16:
        u = 1.0 / len(raw)
        w = np.array([u] * len(raw), dtype=np.float64)
    else:
        w = np.array([r / Z for r in raw], dtype=np.float64)
    return np.array(valid_j, dtype=np.int64), w


def encode_tfidf_weighted_pool(
    texts: list,
    idf: np.ndarray,
    max_length: int = 256,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    texts: list of strings (same corpus used to build idf).
    idf: numpy (vocab_size,) from compute_subword_idf(texts).
    Returns (n, hidden) float64 L2-normalized rows.
    """
    tok, model, device = _get_tok_model(device)
    idf_np = np.asarray(idf, dtype=np.float64)
    if idf_np.shape[0] != tok.vocab_size:
        raise ValueError(f"idf length {idf_np.shape[0]} != vocab_size {tok.vocab_size}")

    cls_id = int(tok.cls_token_id)
    sep_id = int(tok.sep_token_id)
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else None

    out_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch_txt = [str(t) for t in texts[start : start + batch_size]]
            enc = tok(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            ids = enc["input_ids"].cpu().numpy()
            am = enc["attention_mask"].cpu().numpy()
            Hdim = hidden.shape[-1]

            for b in range(hidden.shape[0]):
                row_ids = ids[b]
                row_am = am[b]
                hj, wj = _weights_for_span(row_ids, row_am, idf_np, cls_id, sep_id, pad_id)
                if hj is None:
                    out_rows.append(np.zeros(Hdim, dtype=np.float64))
                    continue
                h_sel = hidden[b, torch.from_numpy(hj).to(device), :]  # (k, H)
                w_t = torch.from_numpy(wj).to(device=device, dtype=h_sel.dtype).view(-1, 1)
                pooled = (h_sel * w_t).sum(dim=0)
                pv = pooled.cpu().numpy().astype(np.float64)
                nrm = np.linalg.norm(pv)
                if nrm > 1e-12:
                    pv = pv / nrm
                out_rows.append(pv)

    return np.stack(out_rows, axis=0)


def unit_mean_direction(emb: np.ndarray) -> np.ndarray | None:
    """
    Mean of embedding rows (e.g. hand-picked exemplar spans), then L2-normalize.
    Each row is typically already L2-normalized; the mean direction is the unified pole.
    """
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] == 0:
        return None
    c = e.mean(axis=0)
    n = np.linalg.norm(c)
    if n < 1e-12:
        return None
    return (c / n).astype(np.float64)


def centroid_unit(emb: np.ndarray, mask) -> np.ndarray | None:
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != emb.shape[0]:
        raise ValueError("mask length mismatch")
    if not np.any(m):
        return None
    c = emb[m].mean(axis=0)
    n = np.linalg.norm(c)
    if n < 1e-12:
        return None
    return (c / n).astype(np.float64)


def cosine_to_rows(emb: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).ravel()
    u = u / (np.linalg.norm(u) + 1e-12)
    e = np.asarray(emb, dtype=np.float64)
    return e @ u
