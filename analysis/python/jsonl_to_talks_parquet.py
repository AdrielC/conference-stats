#!/usr/bin/env python3
"""
Build pipeline-ready talk Parquet without R / generalconference.

gc_chunk_embed_pipeline.py only requires columns: talk_id, year, text.

Input: JSON Lines (.jsonl), one object per line. Typical keys:
  - year (int, required)
  - text (str, required)
  - talk_id (str, optional) — if omitted, a stable hash from metadata is used
  - month, speaker, title, url (optional; included in hash when talk_id is omitted)

This does not fetch General Conference from the web. You supply text from sources
you're allowed to use (manual transcription, licensed dumps, your own scraping
subject to site terms).

Usage:
  python jsonl_to_talks_parquet.py --input talks.jsonl --output gc_talks_normalized.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


def _stable_talk_id(row: dict) -> str:
    if row.get("talk_id") is not None and str(row["talk_id"]).strip():
        return str(row["talk_id"]).strip()
    parts = [
        str(row.get("year", "")),
        str(row.get("month", "")),
        str(row.get("speaker", "")),
        str(row.get("title", "")),
        str(row.get("url", "")),
    ]
    key = "|".join(parts).encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="JSONL file (UTF-8)")
    ap.add_argument("--output", type=Path, required=True, help="Output .parquet")
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Missing input: {args.input}", file=sys.stderr)
        sys.exit(2)

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {i}: invalid JSON: {e}", file=sys.stderr)
                sys.exit(1)
            rows.append(obj)

    if not rows:
        print("No JSON objects read.", file=sys.stderr)
        sys.exit(1)

    out: list[dict] = []
    for r in rows:
        if "year" not in r or r["year"] is None:
            print("Each row needs a year.", file=sys.stderr)
            sys.exit(1)
        if "text" not in r or not str(r["text"]).strip():
            print("Each row needs non-empty text.", file=sys.stderr)
            sys.exit(1)
        year = int(r["year"])
        text = str(r["text"]).strip()
        tid = _stable_talk_id(r)
        rec = {
            "talk_id": tid,
            "year": year,
            "text": text,
        }
        for opt in ("month", "speaker", "title", "url"):
            if opt in r and r[opt] is not None:
                rec[opt] = r[opt]
        out.append(rec)

    df = pd.DataFrame(out)
    dup = df["talk_id"].duplicated()
    if dup.any():
        n = int(dup.sum())
        print(f"Warning: {n} duplicate talk_id values (last row wins if you dedupe).", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
