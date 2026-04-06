#!/usr/bin/env python3
"""
For a small set of talks, find the semantic chunk that best matches a two-phrase
contrast (maximize or minimize cos(A) − cos(B) within each talk).

Used by the Shiny contrast/showpiece exemplars — local "RAG" without a vector DB:
re-encode chunk texts with the same BGE + tf–idf pooling as the main pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hf_tfidf_pool import HfTfidfPooler  # noqa: E402


def _trunc(s: str, n: int = 650) -> str:
    s = str(s).strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "\u2026"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phrase-aligned chunk picks for contrast exemplars.")
    ap.add_argument("--chunks", type=Path, required=True, help="chunks_scored.parquet")
    ap.add_argument("--idf", type=Path, required=True, help="subword_idf.npy")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ua-json", type=Path, required=True, help="JSON array 384 floats phrase A")
    ap.add_argument("--ub-json", type=Path, required=True, help="JSON array 384 floats phrase B")
    ap.add_argument("--queries-json", type=Path, required=True, help='JSON [{talk_id, kind}, ...] kind=toward_a|toward_b')
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if not args.chunks.is_file():
        print(f"Missing chunks parquet: {args.chunks}", file=sys.stderr)
        sys.exit(2)
    if not args.idf.is_file():
        print(f"Missing idf: {args.idf}", file=sys.stderr)
        sys.exit(2)

    ua = np.asarray(json.loads(args.ua_json.read_text(encoding="utf-8")), dtype=np.float64).ravel()
    ub = np.asarray(json.loads(args.ub_json.read_text(encoding="utf-8")), dtype=np.float64).ravel()
    if ua.shape != (384,) or ub.shape != (384,):
        print("ua and ub must be length-384 JSON arrays", file=sys.stderr)
        sys.exit(2)

    queries = json.loads(args.queries_json.read_text(encoding="utf-8"))
    if not queries:
        json.dump([], sys.stdout)
        sys.stdout.write("\n")
        return

    talk_kind: dict[str, str] = {}
    for q in queries:
        tid = str(q["talk_id"])
        kind = str(q["kind"])
        if kind not in ("toward_a", "toward_b"):
            print(f"Invalid kind {kind!r}", file=sys.stderr)
            sys.exit(2)
        talk_kind[tid] = kind

    need_ids = list(talk_kind.keys())
    df = pd.read_parquet(args.chunks)
    if "talk_id" not in df.columns or "text" not in df.columns:
        print("chunks parquet needs talk_id, text", file=sys.stderr)
        sys.exit(2)
    df["talk_id"] = df["talk_id"].astype(str)
    sub = df[df["talk_id"].isin(need_ids)].copy()
    if sub.empty:
        json.dump([], sys.stdout)
        sys.stdout.write("\n")
        return

    idf = np.load(args.idf)
    pooler = HfTfidfPooler(args.model)
    texts = sub["text"].astype(str).tolist()
    emb = pooler.encode_tfidf_weighted_pool(
        texts,
        idf,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prefetch=False,
        show_progress=False,
    )
    ca = emb @ ua
    cb = emb @ ub
    delta = ca - cb
    sub = sub.assign(_ca=ca, _cb=cb, _delta=delta)

    out: list[dict] = []
    chunk_idx_col = "chunk_idx" if "chunk_idx" in sub.columns else None

    for tid in need_ids:
        kind = talk_kind[tid]
        rows = sub[sub["talk_id"] == tid]
        if len(rows) == 0:
            continue
        if kind == "toward_a":
            j = int(rows["_delta"].values.argmax())
        else:
            j = int(rows["_delta"].values.argmin())
        r = rows.iloc[j]
        cidx = int(r[chunk_idx_col]) if chunk_idx_col else j
        out.append(
            {
                "talk_id": tid,
                "kind": kind,
                "chunk_idx": cidx,
                "text_excerpt": _trunc(r["text"]),
                "cos_a": float(r["_ca"]),
                "cos_b": float(r["_cb"]),
                "delta": float(r["_delta"]),
            }
        )

    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
