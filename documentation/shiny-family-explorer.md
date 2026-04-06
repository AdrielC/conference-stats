# General Conference semantic explorer (Shiny app)

The Shiny app lives in **`analysis/shiny_gc_family/`**: **Gallery** (static PNGs), **Explore** (interactive trend), **Methods**, **Chunk insights** (per-talk chunk quotes when `chunk_highlights.rds` is present), **Custom pole** (live phrase embedding), and **Contrast** (two phrases, A − B). Defaults use the **1971–2021** `generalconference`/`genconf` corpus scored by the Python embedding pipeline.

## View the handout HTML on a phone

File: [`shiny-family-explorer.html`](shiny-family-explorer.html)

GitHub “raw” URLs do not render as a web page. Options:

1. **HTMLPreview** — open  
   `https://htmlpreview.github.io/?https://raw.githubusercontent.com/AdrielC/conference-stats/main/documentation/shiny-family-explorer.html`
2. **GitHub Pages** — publish `documentation/` (or equivalent) per your repo settings.
3. **Download** the HTML from GitHub and open it locally.

## Run Shiny (R required)

Step-by-step: [`analysis/shiny_gc_family/HOW_TO_RUN.md`](../analysis/shiny_gc_family/HOW_TO_RUN.md).

```r
shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)
```

Rebuild bundled scores and figures:

```bash
Rscript analysis/plot_gc_chunk_embed_results.R
```

That syncs **`talk_emb_sums.rds`**, **`subword_idf.npy`**, **`pipeline_meta.json`**, and when available **`chunks_scored.parquet`**, into **`analysis/shiny_gc_family/data/`** for **Custom pole**, **Contrast**, and **phrase-aligned exemplars**. Those large/generated files are **gitignored**; run the plot script locally after cloning.

**Python:** install `analysis/python/requirements-gc-embed.txt`; use `CONFERENCESTATS_PYTHON` if needed.

## App behavior (summary)

- **Chunk insights** highlight prescriptive / invitational / swing segments per talk; labels clarify “within talk” vs corpus-wide wording.
- **Custom pole** embeds your phrase with the **same BGE + tf–idf pooling** as offline chunks and plots mean cosine vs year (filters follow Explore).
- **Contrast** compares two phrases (A − B). **Showpiece / Contrast exemplar cards** prefer the chunk in each talk that **maximizes or minimizes** cos(A) − cos(B) among segments when `chunks_scored.parquet` and Python are available; otherwise they use **swing** snippets from Chunk insights.

Repository overview and thumbnails: [`README.md`](../README.md). Methods and inference: [`methods-and-statistical-inference.md`](methods-and-statistical-inference.md).
