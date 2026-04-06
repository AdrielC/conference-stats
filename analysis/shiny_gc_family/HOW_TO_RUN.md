# General Conference — semantic trends Shiny explorer

## 1. R packages (once)

```r
install.packages(c(
  "shiny", "bslib", "ggplot2", "plotly", "dplyr",
  "DT", "jsonlite", "scales", "mgcv"
))
```

## 2. Run the app

From the **conference-stats repo root**:

```r
shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)
```

Or in a shell:

```bash
R -e 'shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)'
```

## 3. Refresh figures and `data/` after the Python embed pipeline

From repo root:

```bash
Rscript analysis/plot_gc_chunk_embed_results.R
```

That writes PNGs under `analysis/output/gc_chunk_embed/` and copies **`talk_scores.rds`**, **`summary_stats.json`**, and figure PNGs into this app folder. If `chunks_scored.parquet` sits next to the pipeline’s `talk_scores.parquet`, the script also builds **`chunk_highlights.rds`** (Chunk insights tab) and copies **`chunks_scored.parquet`** (phrase-aligned exemplars on Showpiece / Contrast tabs).

Set **`CONFERENCESTATS_PYTHON`** if `python3` is not the interpreter with `requirements-gc-embed.txt` installed.

## 4. What works without Python

Gallery, Explore, Methods, and most of Chunk insights: **offline** from tracked `www/` + `data/talk_scores.rds` (and `chunk_highlights.rds` when present).

## 5. What needs Python

| Feature | Required files (under this folder’s `data/`) |
|--------|-----------------------------------------------|
| **Custom pole** — embed a phrase, plot vs year | `talk_emb_sums.rds`, `subword_idf.npy`, `pipeline_meta.json` (synced by the plot script) |
| **Two-phrase contrast** | Same as custom pole |
| **Phrase-aligned exemplar** quotes | `chunks_scored.parquet` + the sidecars above; calls `analysis/python/best_contrast_chunks.py` at runtime |

If those Parquet/RDS/NPY files are missing, run the Python pipeline and the plot script (see repo **`README.md`**). Generated sidecars are listed in **`.gitignore`** so local syncs do not clutter `git status`; rebuild them after pulling.

## 6. Alternate talk corpus (full Python path)

To score talks not from `generalconference`, produce a Parquet with columns **`talk_id`**, **`year`**, **`text`** (e.g. `analysis/python/jsonl_to_talks_parquet.py` from JSONL), then run **`gc_chunk_embed_pipeline.py`** and point `plot_gc_chunk_embed_results.R` at the new `talk_scores.parquet` directory.
