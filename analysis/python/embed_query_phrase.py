#!/usr/bin/env python3
"""
Embed one user phrase with corpus subword IDF + tf–idf pooling (same as gc_chunk_embed_pipeline).
Prints a JSON array of floats (unit L2 row) to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hf_tfidf_pool import HfTfidfPooler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed query phrase for custom-pole Shiny tab.")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--idf", type=Path, required=True, help="subword_idf.npy from pipeline output dir")
    phrase_grp = ap.add_mutually_exclusive_group(required=True)
    phrase_grp.add_argument(
        "--phrase",
        help="Literal phrase (fine for one word; on Windows multi-word phrases belong in --phrase-file from callers).",
    )
    phrase_grp.add_argument(
        "--phrase-file",
        type=Path,
        dest="phrase_file",
        metavar="PATH",
        help="UTF-8 text file with the phrase (recommended when the phrase has spaces or newlines).",
    )
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if not args.idf.is_file():
        print(f"Missing IDF file: {args.idf}", file=sys.stderr)
        sys.exit(2)

    if args.phrase_file is not None:
        if not args.phrase_file.is_file():
            print(f"Missing phrase file: {args.phrase_file}", file=sys.stderr)
            sys.exit(2)
        phrase = args.phrase_file.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
    else:
        phrase = args.phrase or ""

    if not phrase.strip():
        print("Phrase is empty after reading --phrase or --phrase-file.", file=sys.stderr)
        sys.exit(2)

    idf = np.load(args.idf)
    pooler = HfTfidfPooler(args.model)
    arr = pooler.encode_tfidf_weighted_pool(
        [phrase],
        idf,
        max_length=args.max_length,
        batch_size=args.batch_size,
        prefetch=False,
        show_progress=False,
    )
    u = np.asarray(arr[0], dtype=np.float64)
    nu = np.linalg.norm(u)
    if nu > 1e-12:
        u = u / nu
    json.dump(u.tolist(), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
