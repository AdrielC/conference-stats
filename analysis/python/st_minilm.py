# sentence-transformers/all-MiniLM-L6-v2 — lazy singleton, batch encode.
# Called from R via reticulate::import_from_path("st_minilm", ...).

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def encode_texts(texts, batch_size=48, normalize_embeddings=True):
    """
    Encode a sequence of strings to a dense matrix (n, 384), float32.

    The model uses its own HuggingFace WordPiece tokenizer internally; use
    full natural-language chunks (paragraphs / short spans), not pre-tokenized
    word lists.
    """
    import numpy as np

    if texts is None:
        raise ValueError("texts is None")
    texts = [str(t) if t is not None else "" for t in texts]
    model = _get_model()
    arr = model.encode(
        texts,
        batch_size=int(batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(normalize_embeddings),
    )
    return np.asarray(arr, dtype=np.float32)


def centroid_unit(emb, mask):
    """Mean of selected rows of emb, then L2-normalize. emb: (n, d); mask: length n."""
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != emb.shape[0]:
        raise ValueError("mask length must match nrows of emb")
    if not m.any():
        return None
    c = emb[m].mean(axis=0)
    nrm = np.linalg.norm(c)
    if nrm < 1e-12:
        return None
    return (c / nrm).astype(np.float64)


def cosine_to_rows(emb, u):
    """emb (n, d), u (d,) unit vector -> (n,) cosine similarities."""
    import numpy as np

    u = np.asarray(u, dtype=np.float64).ravel()
    u = u / (np.linalg.norm(u) + 1e-12)
    e = np.asarray(emb, dtype=np.float64)
    return e @ u
