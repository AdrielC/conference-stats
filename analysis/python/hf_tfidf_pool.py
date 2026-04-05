"""
HuggingFace transformer last-layer pooling with subword TF–IDF weights.
Works with any `AutoModel` + `AutoTokenizer` (e.g. BAAI/bge-small-en-v1.5 — the
same weights FastEmbed distributes as ONNX; FastEmbed does not expose
per-token hidden states, so tf–idf-weighted pooling uses PyTorch here).

Performance:
- `compute_subword_idf(..., num_workers>1)` shards tokenization across processes.
- `encode_tfidf_weighted_pool(..., prefetch=True)` overlaps CPU tokenization with
  GPU/MPS forward (no extra VRAM for a second model).
"""

from __future__ import annotations

import os
import queue
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from progress_util import executor_map_as_completed, tqdm_maybe

# --- IDF multiprocessing (tokenizer only per child) ---
_IDF_TOK: Any = None


def _idf_worker_init(model_name: str) -> None:
    global _IDF_TOK
    _IDF_TOK = AutoTokenizer.from_pretrained(model_name)


def _idf_accumulate_shard(payload: tuple[list[str], int, int, tuple[int, ...]]) -> np.ndarray:
    texts, max_length, vocab_size, exclude_tuple = payload
    exclude = set(exclude_tuple)
    tok = _IDF_TOK
    df = np.zeros(vocab_size, dtype=np.int64)
    for text in texts:
        enc = tok(
            str(text),
            add_special_tokens=True,
            max_length=int(max_length),
            truncation=True,
            return_tensors=None,
        )
        ids = enc["input_ids"]
        seen = {int(i) for i in ids} - exclude
        for i in seen:
            if 0 <= i < vocab_size:
                df[i] += 1
    return df


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HfTfidfPooler:
    def __init__(self, model_name: str, device: torch.device | None = None):
        self.model_name = model_name
        self.device = device if device is not None else default_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model: PreTrainedModel | None = None

    def _get_model(self) -> PreTrainedModel:
        if self._model is None:
            m = AutoModel.from_pretrained(self.model_name)
            m.eval()
            m.to(self.device)
            self._model = m
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def compute_subword_idf(
        self,
        texts: list,
        max_length: int = 512,
        num_workers: int = 0,
        show_progress: bool = False,
    ) -> np.ndarray:
        tok = self._tokenizer
        vs = int(tok.vocab_size)
        exclude: set[int] = set()
        for x in (tok.pad_token_id, tok.cls_token_id, tok.sep_token_id, tok.unk_token_id):
            if x is not None:
                exclude.add(int(x))
        n = len(texts)
        if n == 0:
            return np.ones(vs, dtype=np.float64)

        nw = int(num_workers) if num_workers else 0
        ex_tuple = tuple(sorted(exclude))

        if nw <= 1:
            df = np.zeros(vs, dtype=np.int64)
            it = tqdm_maybe(
                texts,
                total=n,
                desc="IDF tokenize",
                disable=not show_progress,
            )
            for text in it:
                enc = tok(
                    str(text),
                    add_special_tokens=True,
                    max_length=int(max_length),
                    truncation=True,
                    return_tensors=None,
                )
                ids = enc["input_ids"]
                seen = {int(i) for i in ids} - exclude
                for i in seen:
                    if 0 <= i < vs:
                        df[i] += 1
        else:
            cpu = os.cpu_count() or 4
            nw = min(nw, cpu, max(1, n // 32))
            shard_n = max(1, (n + nw - 1) // nw)
            shards: list[list[str]] = [texts[i : i + shard_n] for i in range(0, n, shard_n)]
            payloads = [(sh, max_length, vs, ex_tuple) for sh in shards]
            df = np.zeros(vs, dtype=np.int64)
            ctx = ProcessPoolExecutor(
                max_workers=len(shards),
                initializer=_idf_worker_init,
                initargs=(self.model_name,),
            )
            try:
                dfs = executor_map_as_completed(
                    ctx,
                    _idf_accumulate_shard,
                    payloads,
                    "IDF tokenize",
                    disable=not show_progress,
                )
                for d in dfs:
                    df += d
            finally:
                ctx.shutdown(wait=True)

        idf = np.log((float(n) + 1.0) / (df.astype(np.float64) + 1.0)) + 1.0
        return idf

    @staticmethod
    def _weights_for_span(
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        idf_np: np.ndarray,
        cls_id: int,
        sep_id: int,
        pad_id: int | None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        l = int(input_ids.shape[0])
        mask = attention_mask.astype(bool)
        valid_j = []
        valid_t = []
        for j in range(l):
            if not mask[j]:
                continue
            tid = int(input_ids[j])
            if tid == cls_id or tid == sep_id or (pad_id is not None and tid == pad_id):
                continue
            valid_j.append(j)
            valid_t.append(tid)
        if not valid_j:
            return None, None
        m_ct = len(valid_t)
        cnt = Counter(valid_t)
        raw = [(cnt[tid] / float(m_ct)) * float(idf_np[tid]) for tid in valid_t]
        z = float(sum(raw))
        if z < 1e-16:
            w = np.full(len(raw), 1.0 / len(raw), dtype=np.float64)
        else:
            w = np.array([r / z for r in raw], dtype=np.float64)
        return np.array(valid_j, dtype=np.int64), w

    def encode_tfidf_weighted_pool(
        self,
        texts: list,
        idf: np.ndarray,
        max_length: int = 512,
        batch_size: int = 8,
        prefetch: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        tok, model, device = self._tokenizer, self._get_model(), self.device
        idf_np = np.asarray(idf, dtype=np.float64)
        if idf_np.shape[0] != tok.vocab_size:
            raise ValueError(f"idf length {idf_np.shape[0]} != vocab_size {tok.vocab_size}")

        cls_id = int(tok.cls_token_id)
        sep_id = int(tok.sep_token_id)
        pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else None

        n_text = len(texts)
        bs = int(batch_size)
        n_batch = (n_text + bs - 1) // bs if n_text else 0
        pbar = tqdm_maybe(total=n_batch, desc="Encode tf–idf pool", disable=not show_progress)
        out_rows: list[np.ndarray] = []
        model.eval()

        def run_batch(enc_cpu: dict[str, torch.Tensor]) -> None:
            enc = {k: v.to(device) for k, v in enc_cpu.items()}
            hidden = model(**enc).last_hidden_state
            ids = enc["input_ids"].cpu().numpy()
            am = enc["attention_mask"].cpu().numpy()
            hdim = hidden.shape[-1]
            for b in range(hidden.shape[0]):
                row_ids = ids[b]
                row_am = am[b]
                hj, wj = self._weights_for_span(row_ids, row_am, idf_np, cls_id, sep_id, pad_id)
                if hj is None:
                    out_rows.append(np.zeros(hdim, dtype=np.float64))
                    continue
                h_sel = hidden[b, torch.from_numpy(hj).to(device), :]
                w_t = torch.from_numpy(wj).to(device=device, dtype=h_sel.dtype).view(-1, 1)
                pooled = (h_sel * w_t).sum(dim=0)
                pv = pooled.cpu().numpy().astype(np.float64)
                nrm = np.linalg.norm(pv)
                if nrm > 1e-12:
                    pv = pv / nrm
                out_rows.append(pv)
            pbar.update(1)

        use_prefetch = bool(prefetch) and device.type in ("cuda", "mps")

        with torch.inference_mode():
            if not use_prefetch or n_text == 0:
                for start in range(0, n_text, bs):
                    batch_txt = [str(t) for t in texts[start : start + bs]]
                    enc = tok(
                        batch_txt,
                        padding=True,
                        truncation=True,
                        max_length=int(max_length),
                        return_tensors="pt",
                    )
                    run_batch(enc)
            else:
                q: queue.Queue[dict[str, torch.Tensor] | None] = queue.Queue(maxsize=2)

                def producer() -> None:
                    try:
                        for start in range(0, n_text, bs):
                            batch_txt = [str(t) for t in texts[start : start + bs]]
                            enc = tok(
                                batch_txt,
                                padding=True,
                                truncation=True,
                                max_length=int(max_length),
                                return_tensors="pt",
                            )
                            q.put(enc)
                    finally:
                        q.put(None)  # type: ignore[arg-type]

                th = threading.Thread(target=producer, daemon=True)
                th.start()
                while True:
                    enc_cpu = q.get()
                    if enc_cpu is None:
                        break
                    run_batch(enc_cpu)
                th.join(timeout=60)

        pbar.close()
        return np.stack(out_rows, axis=0)


def unit_mean_direction(emb: np.ndarray) -> np.ndarray | None:
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] == 0:
        return None
    c = e.mean(axis=0)
    n = np.linalg.norm(c)
    if n < 1e-12:
        return None
    return (c / n).astype(np.float64)
