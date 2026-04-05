#!/usr/bin/env python3
"""
General Conference talk → semantic chunks → tf–idf-weighted BGE embeddings → cosine
scores vs regex-mined exemplar poles → per-talk aggregate.

- Chunking: semantic-text-splitter with the *same* HuggingFace tokenizer as the
  embedding model (token budget per chunk).
- Embeddings: PyTorch `AutoModel` last-hidden-state pooling with subword tf×idf
  weights (same scheme as hf_tfidf_pool.py). This matches **BAAI/bge-small-en-v1.5**,
  the default ONNX model shipped by **FastEmbed**; FastEmbed’s public `embed()` API
  does not expose token vectors, so pooling is done here in Transformers.
- Exemplars: sentences from raw talk text that match prescriptive / invitational
  regex bundles (keep in sync with analysis/general_conference_nlp_report.Rmd).

Outputs (Parquet): chunks_scored.parquet, talk_scores.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hf_tfidf_pool import HfTfidfPooler, unit_mean_direction
from progress_util import executor_map_as_completed, tqdm_maybe

# --- Regex bundles (mirror general_conference_nlp_report.Rmd) ---
PRESC_RX = re.compile(
    "|".join(
        [
            r"do\s+not",
            r"don'?t",
            r"can\s+not",
            r"cannot",
            r"must\s+not",
            r"should\s+not",
            r"ought\s+not",
            r"thou\s+shalt\s+not",
            r"shall\s+not",
            r"never\s+\w+",
            r"cease\s+(from|to)",
            r"beware",
            r"wo\s*(e)?\s*unto",
            r"woe\s+unto",
            r"condemn",
            r"rebellion\s+against",
            r"must\s+stop",
        ]
    ),
    re.IGNORECASE,
)
GENTLE_RX = re.compile(
    "|".join(
        [
            r"i invite",
            r"we invite",
            r"invitation",
            r"come\s+unto\s+christ",
            r"jesus\s+christ",
            r"our\s+savior",
            r"redeemer",
            r"his\s+love",
            r"peace\s+in\s+christ",
            r"gentle",
            r"tender",
            r"mercies",
            r"consider\s+how",
            r"ponder",
        ]
    ),
    re.IGNORECASE,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# If the corpus yields too few regex hits, pad poles with short hand-authored examples.
FALLBACK_PRESCRIPTIVE = [
    "Ye must repent and come unto Christ.",
    "Thou shalt not commit adultery.",
    "Beware of pride.",
    "Woe unto the wicked.",
    "Do not procrastinate the day of your repentance.",
]
FALLBACK_INVITATIONAL = [
    "I invite you to come unto Christ.",
    "We invite you to feel His redeeming.",
    "Ponder the tender mercies of the Lord.",
    "His love and peace are offered to all.",
]


def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    parts = SENT_SPLIT.split(str(text))
    return [p.strip() for p in parts if len(p.strip()) >= min_chars]


def _regex_order_one(text: str) -> list[tuple[str, bool, bool]]:
    """Per talk: sentence order + flags (picklable for multiprocessing)."""
    return [
        (s, bool(PRESC_RX.search(s)), bool(GENTLE_RX.search(s)))
        for s in split_sentences(text)
    ]


def _regex_batch_worker(texts: list[str]) -> list[list[tuple[str, bool, bool]]]:
    """Process a shard of talks in one worker (fewer IPC round-trips than one future per talk)."""
    return [_regex_order_one(t) for t in texts]


def collect_regex_exemplars(
    texts: list[str],
    max_per_pole: int,
    num_workers: int = 0,
    show_progress: bool = True,
) -> tuple[list[str], list[str]]:
    """Mine exemplars in corpus talk order; optionally parallel per-talk regex scan."""
    nw = int(num_workers) if num_workers else 0
    cpu = os.cpu_count() or 4
    if nw <= 1 or len(texts) < 32:
        it = tqdm_maybe(
            texts,
            total=len(texts),
            desc="Regex scan (talks)",
            disable=not show_progress,
        )
        ordered = [_regex_order_one(t) for t in it]
    else:
        nw = min(nw, cpu)
        shard_n = max(1, (len(texts) + nw - 1) // nw)
        shards = [texts[i : i + shard_n] for i in range(0, len(texts), shard_n)]
        with ProcessPoolExecutor(max_workers=len(shards)) as ex:
            parts = executor_map_as_completed(
                ex,
                _regex_batch_worker,
                shards,
                "Regex scan (shards)",
                disable=not show_progress,
            )
        ordered = [row for part in parts for row in part]

    presc: list[str] = []
    gentle: list[str] = []
    seen_p: set[str] = set()
    seen_g: set[str] = set()
    for rows in ordered:
        for s, hit_p, hit_g in rows:
            if len(presc) >= max_per_pole and len(gentle) >= max_per_pole:
                return presc, gentle
            key = s.lower()
            if hit_p and len(presc) < max_per_pole and key not in seen_p:
                presc.append(s)
                seen_p.add(key)
            if hit_g and len(gentle) < max_per_pole and key not in seen_g:
                gentle.append(s)
                seen_g.add(key)
        if len(presc) >= max_per_pole and len(gentle) >= max_per_pole:
            break
    return presc, gentle


def _year_py(y: Any) -> int | None:
    if y is None:
        return None
    try:
        if isinstance(y, float) and np.isnan(y):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(y)
    except (TypeError, ValueError):
        return None


def _chunk_shard_worker(payload: tuple[str, int, list[dict[str, Any]]]) -> list[dict]:
    """Per process: build Tokenizer+TextSplitter once, chunk a shard of talks."""
    model_name, max_chunk_tokens, dict_rows = payload
    from tokenizers import Tokenizer
    from semantic_text_splitter import TextSplitter

    tok_json = Tokenizer.from_pretrained(model_name)
    splitter = TextSplitter.from_huggingface_tokenizer(tok_json, max_chunk_tokens)
    out: list[dict] = []
    for r in dict_rows:
        talk_id, year, text = r["talk_id"], r["year"], r["text"]
        chs = splitter.chunks(str(text))
        if not isinstance(chs, (list, tuple)):
            chs = list(chs)
        for i, c in enumerate(chs):
            out.append(
                {
                    "talk_id": talk_id,
                    "year": _year_py(year),
                    "chunk_idx": i,
                    "text": str(c),
                }
            )
    return out


def chunk_all_talks(
    df: pd.DataFrame,
    model_name: str,
    max_chunk_tokens: int,
    num_workers: int,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Semantic chunks for every row; multiprocess over talk shards when num_workers>1."""
    records = df[["talk_id", "year", "text"]].to_dict("records")
    n = len(records)
    if n == 0:
        return pd.DataFrame(columns=["talk_id", "year", "chunk_idx", "text"])

    nw = int(num_workers) if num_workers else 0
    cpu = os.cpu_count() or 4

    if nw <= 1 or n < 16:
        from tokenizers import Tokenizer
        from semantic_text_splitter import TextSplitter

        tok_json = Tokenizer.from_pretrained(model_name)
        splitter = TextSplitter.from_huggingface_tokenizer(tok_json, max_chunk_tokens)
        chunk_rows: list[dict] = []
        rec_it = tqdm_maybe(
            records,
            total=n,
            desc="Semantic chunking (1 worker)",
            disable=not show_progress,
        )
        for r in rec_it:
            chs = splitter.chunks(str(r["text"]))
            if not isinstance(chs, (list, tuple)):
                chs = list(chs)
            for i, c in enumerate(chs):
                chunk_rows.append(
                    {
                        "talk_id": r["talk_id"],
                        "year": _year_py(r["year"]),
                        "chunk_idx": i,
                        "text": str(c),
                    }
                )
        return pd.DataFrame(chunk_rows)

    nw = min(nw, cpu, max(2, n // 8))
    shard_n = max(1, (n + nw - 1) // nw)
    shards: list[list[dict[str, Any]]] = [
        records[i : i + shard_n] for i in range(0, n, shard_n)
    ]
    payloads = [(model_name, max_chunk_tokens, sh) for sh in shards]
    with ProcessPoolExecutor(max_workers=len(shards)) as ex:
        parts = executor_map_as_completed(
            ex,
            _chunk_shard_worker,
            payloads,
            "Semantic chunking",
            disable=not show_progress,
        )
    chunk_rows = [row for part in parts for row in part]
    return pd.DataFrame(chunk_rows)


def pad_exemplars(mined: list[str], fallback: list[str], seen: set[str], target: int) -> list[str]:
    out = list(mined)
    for s in fallback:
        if len(out) >= target:
            break
        k = s.lower()
        if k not in seen:
            out.append(s)
            seen.add(k)
    return out


def try_fastembed_smoke(model_name: str) -> None:
    try:
        from fastembed import TextEmbedding

        emb = TextEmbedding(model_name=model_name)
        v = next(emb.embed(["cosine smoke test phrase"]))
    except Exception as exc:
        print(f"FastEmbed smoke skipped ({exc}). HF pooling still runs.", file=sys.stderr)
        return
    print(f"FastEmbed OK — model {model_name!r}, dim={len(v)}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Parquet: talk_id, year, text (+ optional columns)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="HF id; matches FastEmbed default BGE-small")
    ap.add_argument("--max-chunk-tokens", type=int, default=220)
    ap.add_argument("--max-length", type=int, default=512, help="Transformer encode max length")
    ap.add_argument("--max-exemplars", type=int, default=400, help="Max mined sentences per pole before fallback")
    ap.add_argument("--min-exemplars", type=int, default=12, help="Minimum sentences per pole after fallback")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Parallel preprocess workers (semantic chunking, IDF tokenization, regex scan). 1=off.",
    )
    ap.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable tokenizer/encoder overlap on CUDA/MPS (hot-path micro-optimization).",
    )
    ap.add_argument("--fastembed-smoke", action="store_true", help="Verify FastEmbed loads same model family")
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (parallel stages stay the same).",
    )
    args = ap.parse_args()

    if args.fastembed_smoke:
        try_fastembed_smoke(args.model)

    df = pd.read_parquet(args.input)
    for col in ("talk_id", "year", "text"):
        if col not in df.columns:
            raise SystemExit(f"Input parquet must contain column {col!r}")

    try:
        import semantic_text_splitter  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Install semantic-text-splitter (Python >= 3.10 recommended): "
            "pip install -r analysis/python/requirements-gc-embed.txt"
        ) from e

    workers = max(1, int(args.workers))
    prog = not args.no_progress
    print(
        f"Chunking {len(df)} talks (workers={workers}) …",
        file=sys.stderr,
    )
    chunks_df = chunk_all_talks(
        df, args.model, args.max_chunk_tokens, workers, show_progress=prog
    )
    if chunks_df.empty:
        raise SystemExit("No chunks produced (empty corpus?)")

    all_talk_text = df["text"].astype(str).tolist()
    print(f"Mining regex exemplars (workers={workers}) …", file=sys.stderr)
    presc_ex, gent_ex = collect_regex_exemplars(
        all_talk_text, args.max_exemplars, workers, show_progress=prog
    )
    seen_p = {s.lower() for s in presc_ex}
    seen_g = {s.lower() for s in gent_ex}
    presc_ex = pad_exemplars(presc_ex, FALLBACK_PRESCRIPTIVE, seen_p, args.min_exemplars)
    gent_ex = pad_exemplars(gent_ex, FALLBACK_INVITATIONAL, seen_g, args.min_exemplars)

    if len(presc_ex) < args.min_exemplars or len(gent_ex) < args.min_exemplars:
        raise SystemExit(f"Too few exemplars: prescriptive={len(presc_ex)}, invitational={len(gent_ex)}")

    pooler = HfTfidfPooler(args.model)
    chunk_texts = chunks_df["text"].tolist()

    print(
        f"Fitting IDF on {len(chunk_texts)} semantic chunks (workers={workers}) …",
        file=sys.stderr,
    )
    idf = pooler.compute_subword_idf(
        chunk_texts,
        max_length=args.max_length,
        num_workers=workers,
        show_progress=prog,
    )

    prefetch = not args.no_prefetch
    print(f"Encoding poles: {len(presc_ex)} prescriptive, {len(gent_ex)} invitational …", file=sys.stderr)
    emb_p = pooler.encode_tfidf_weighted_pool(
        presc_ex,
        idf,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prefetch=prefetch,
        show_progress=prog,
    )
    emb_g = pooler.encode_tfidf_weighted_pool(
        gent_ex,
        idf,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prefetch=prefetch,
        show_progress=prog,
    )
    u_p = unit_mean_direction(emb_p)
    u_g = unit_mean_direction(emb_g)
    if u_p is None or u_g is None:
        raise SystemExit("Failed to build exemplar pole vectors")

    print(f"Encoding {len(chunk_texts)} chunks (prefetch={prefetch}) …", file=sys.stderr)
    emb_c = pooler.encode_tfidf_weighted_pool(
        chunk_texts,
        idf,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prefetch=prefetch,
        show_progress=prog,
    )
    u_p = u_p / (np.linalg.norm(u_p) + 1e-12)
    u_g = u_g / (np.linalg.norm(u_g) + 1e-12)
    cos_p = emb_c @ u_p
    cos_g = emb_c @ u_g
    net = cos_p - cos_g

    chunks_df = chunks_df.assign(
        cos_presc=cos_p.astype(np.float64),
        cos_gentle=cos_g.astype(np.float64),
        net_presc=net.astype(np.float64),
    )

    talk_agg = (
        chunks_df.groupby(["talk_id", "year"], dropna=False, as_index=False)
        .agg(
            n_chunks=("chunk_idx", "count"),
            mean_net_presc=("net_presc", "mean"),
            mean_cos_presc=("cos_presc", "mean"),
            mean_cos_gentle=("cos_gentle", "mean"),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks_scored.parquet"
    talk_path = args.output_dir / "talk_scores.parquet"
    meta_path = args.output_dir / "pipeline_meta.json"

    chunks_df.to_parquet(chunks_path, index=False)
    talk_agg.to_parquet(talk_path, index=False)

    import json

    meta = {
        "model": args.model,
        "n_talks": int(len(df)),
        "n_chunks": int(len(chunks_df)),
        "n_presc_exemplars": len(presc_ex),
        "n_gentle_exemplars": len(gent_ex),
        "max_chunk_tokens": args.max_chunk_tokens,
        "preprocess_workers": workers,
        "encode_prefetch": prefetch,
        "tqdm_progress": prog,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {chunks_path} and {talk_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
