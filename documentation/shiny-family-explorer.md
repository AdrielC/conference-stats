# General Conference semantic explorer (family Shiny app)

This repo includes a small **R Shiny** app under `analysis/shiny_gc_family/`: static plots in **Gallery**, an interactive year vs. score chart in **Explore**, methodology in **Methods**, and **Chunk insights** (when `data/chunk_highlights.rds` is present) to inspect which text segments drove each talk’s score.

## View the handout page in a mobile browser

A single-page **HTML** summary (no R required) lives in this folder:

- **File:** [`shiny-family-explorer.html`](shiny-family-explorer.html)

Because **GitHub “raw” links do not render HTML** as a web page (the file is shown or downloaded as text), use one of these:

1. **HTMLPreview (quickest share link)** — paste your raw URL into [htmlpreview.github.io](https://htmlpreview.github.io/) or open directly, for this repo on `main`:

   ```text
   https://htmlpreview.github.io/?https://raw.githubusercontent.com/AdrielC/conference-stats/main/documentation/shiny-family-explorer.html
   ```

   Anyone can open that link on a phone; it loads the HTML from GitHub and renders it in the browser.

2. **GitHub Pages** — If you enable Pages (e.g. “Deploy from branch”, folder `/documentation` or repo root), adjust the publish path so this file is exposed; the exact URL depends on your settings.

3. **Download and open** — On the normal GitHub file view for `documentation/shiny-family-explorer.html`, use “Download” or “Save as”, then open the file from your phone’s Files/Downloads app (works offline after download).

## Run the interactive Shiny app (R required)

Detailed steps also live in [`analysis/shiny_gc_family/HOW_TO_RUN.md`](../analysis/shiny_gc_family/HOW_TO_RUN.md).

**Short version** (from repo root):

```r
shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)
```

Rebuild figures and bundled data (including `chunk_highlights.rds` when `chunks_scored.parquet` exists):

```bash
Rscript analysis/plot_gc_chunk_embed_results.R
```

That copy step also ships **`talk_emb_sums.rds`**, **`subword_idf.npy`**, and **`pipeline_meta.json`** into `analysis/shiny_gc_family/data/` when the Python pipeline has written them next to `talk_scores.parquet`. Those power the **Custom pole** tab: you type a phrase, Python embeds it with the same BGE + tf–idf scheme, and the app plots mean cosine vs conference year (filters follow the **Explore** tab).

**Custom pole requirements:** Python 3 with `analysis/python/requirements-gc-embed.txt` installed; `python3` on your `PATH` or set `CONFERENCESTATS_PYTHON` to your venv interpreter. The first embedding call loads the model and can take ~1 minute.

## What we changed recently (high level)

- **Gallery images** are scaled with CSS so high-resolution PNGs fit the card and viewport without re-exporting files.
- **Chunk insights** use cleaned text, skip stiff “amen” closings and the **last chunk** of each talk when possible, avoid duplicate segments across categories, and label “highest net in this talk” so negative scores are not misread as globally “prescriptive.”
- **Custom pole** tab: optional interactive trend for any short phrase you type, using the same embedding geometry as the main pipeline (requires Python + synced `talk_emb_sums` / `subword_idf` in `data/`).

For the full figure set and README thumbnails, see the [repository `README.md`](../README.md).
