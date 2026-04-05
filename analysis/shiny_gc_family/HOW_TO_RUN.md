# General Conference — family Shiny explorer

---

1. Install R packages (once):

```R
install.packages(c("shiny", "bslib", "ggplot2", "plotly", "dplyr", 
                   "DT", "jsonlite", "scales", "mgcv"))
```

1. From the conference-stats repo root:

```R
shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)
```

Or in a terminal:

```bash
R -e 'shiny::runApp("analysis/shiny_gc_family", launch.browser=TRUE)'
```

1. Refreshing charts + bundled data after re-running the Python pipeline:

```bash
Rscript analysis/plot_gc_chunk_embed_results.R
```

That writes PNGs to `analysis/output/gc_chunk_embed/` and copies them
(plus `talk_scores.rds` and `summary_stats.json`) into this folder.

If analysis/data/gc_chunk_embed/chunks_scored.parquet exists next
to talk_scores.parquet, the same script also builds
analysis/shiny_gc_family/data/chunk_highlights.rds — used by the
**Chunk insights** tab (top prescriptive / most invitational /
highest-leverage chunks per talk).

The app runs offline: images live in www/ and scores in data/.
No Python or Parquet is required to open most of the Shiny UI.

**Custom pole tab:** Needs local **Python** (same stack as `analysis/python/gc_chunk_embed_pipeline.py`) plus `data/talk_emb_sums.rds` and `data/subword_idf.npy` synced by the plot script after an updated pipeline run. Set `CONFERENCESTATS_PYTHON` if `python3` is not the right interpreter.