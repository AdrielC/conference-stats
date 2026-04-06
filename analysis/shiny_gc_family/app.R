# ---- General Conference — semantic trends Shiny explorer (NLP / exploratory stats) ----
# Run from R:  shiny::runApp("analysis/shiny_gc_family", launch.browser = TRUE)
# Or: setwd to this folder then shiny::runApp(".")

suppressPackageStartupMessages({
  library(shiny)
  library(bslib)
  library(ggplot2)
  library(plotly)
  library(dplyr)
  library(DT)
  library(jsonlite)
  library(scales)
  library(mgcv)
})

app_dir <- getwd()

talk_scores <- readRDS(file.path(app_dir, "data", "talk_scores.rds"))
summ <- jsonlite::fromJSON(file.path(app_dir, "data", "summary_stats.json"))

ch_hi_path <- file.path(app_dir, "data", "chunk_highlights.rds")
has_chunk_highlights <- file.exists(ch_hi_path)
chunk_highlights <- if (has_chunk_highlights) {
  readRDS(ch_hi_path)
} else {
  NULL
}

years_for_pick <- sort(unique(talk_scores$year))

emb_sum_path <- file.path(app_dir, "data", "talk_emb_sums.rds")
idf_bundle_path <- file.path(app_dir, "data", "subword_idf.npy")
pipe_meta_path <- file.path(app_dir, "data", "pipeline_meta.json")
has_custom_pole_bundle <- file.exists(emb_sum_path) && file.exists(idf_bundle_path)
talk_emb_sums_tbl <- if (isTRUE(has_custom_pole_bundle)) readRDS(emb_sum_path) else NULL
pipe_meta <- if (file.exists(pipe_meta_path)) {
  jsonlite::fromJSON(pipe_meta_path, simplifyVector = TRUE)
} else {
  list(model = "BAAI/bge-small-en-v1.5")
}
if (is.null(pipe_meta$model) || !nzchar(as.character(pipe_meta$model))) {
  pipe_meta$model <- "BAAI/bge-small-en-v1.5"
}
py_embed_script <- normalizePath(file.path(app_dir, "..", "python", "embed_query_phrase.py"), mustWork = FALSE)
has_py_embed_script <- nzchar(py_embed_script) && file.exists(py_embed_script)
s_cols_custom <- if (isTRUE(has_custom_pole_bundle)) {
  nm <- names(talk_emb_sums_tbl)
  sc <- nm[grepl("^s_[0-9]+$", nm)]
  sc <- sc[order(as.integer(sub("^s_", "", sc)))]
  if (length(sc) != 384L) character(0) else sc
} else {
  character(0)
}
has_custom_dim_ok <- length(s_cols_custom) == 384L

talk_scores <- talk_scores |>
  mutate(
    decade = floor(.data$year / 10) * 10,
    era = cut(
      .data$year,
      breaks = c(1970, 1985, 2000, 2015, 2022),
      labels = c("1971–1985", "1986–2000", "2001–2015", "2016–2021")
    )
  ) |>
  filter(!is.na(.data$era))

era_levels <- levels(droplevels(talk_scores$era))
cust_yr_min <- min(talk_scores$year, na.rm = TRUE)
cust_yr_max <- max(talk_scores$year, na.rm = TRUE)
tw_tab_yr_lo_default <- max(1996L, as.integer(cust_yr_min))

p_gam_txt <- if (summ$gam_p_smooth < 1e-6 || summ$gam_p_smooth == 0) {
  "< 0.001"
} else {
  as.character(signif(summ$gam_p_smooth, 2))
}

chunk_kind_hex <- function(kind) {
  switch(
    as.character(kind),
    prescriptive = "#9b2c2c",
    invitational = "#276749",
    swing = "#c05621",
    "#4a5568"
  )
}

## One semantic-chunk card (1-row highlight tbl_df / data.frame)
fmt_p_num <- function(p) {
  p <- suppressWarnings(as.numeric(p))
  if (length(p) != 1L || !is.finite(p)) {
    return("NA")
  }
  if (p < 1e-6) {
    return("< 0.000001")
  }
  if (p < 1e-3) {
    return(sprintf("%.2e", p))
  }
  sprintf("%.4f", p)
}

## OLS line + 95% confidence band on unique x (robust in ggplotly vs geom_smooth).
lm_line_ribbon_df <- function(d, x_col, y_col) {
  if (!all(c(x_col, y_col) %in% names(d))) {
    return(NULL)
  }
  dd <- d[is.finite(d[[x_col]]) & is.finite(d[[y_col]]), , drop = FALSE]
  if (nrow(dd) < 3L) {
    return(NULL)
  }
  if (length(unique(dd[[x_col]])) < 2L) {
    return(NULL)
  }
  f <- stats::reformulate(x_col, response = y_col)
  m <- tryCatch(stats::lm(f, data = dd), error = function(e) NULL)
  if (is.null(m)) {
    return(NULL)
  }
  xu <- sort(unique(dd[[x_col]]))
  nd <- stats::setNames(data.frame(xu), x_col)
  pr <- tryCatch(
    stats::predict(m, newdata = nd, interval = "confidence", level = 0.95),
    error = function(e) NULL
  )
  if (is.null(pr) || ncol(pr) < 3L) {
    return(NULL)
  }
  data.frame(x = xu, fit = pr[, 1L], ymin = pr[, 2L], ymax = pr[, 3L])
}

## GAM smooth on a dense year grid + ~95% band from se.fit (Explore scatter overlay).
gam_line_ribbon_df <- function(d, x_col, y_col, k = 10L, n_grid = 150L) {
  if (!all(c(x_col, y_col) %in% names(d))) {
    return(NULL)
  }
  dd <- d[is.finite(d[[x_col]]) & is.finite(d[[y_col]]), , drop = FALSE]
  if (nrow(dd) < 8L) {
    return(NULL)
  }
  if (length(unique(dd[[x_col]])) < 4L) {
    return(NULL)
  }
  k <- max(4L, min(as.integer(k), nrow(dd) - 1L))
  fo <- stats::as.formula(paste0(y_col, " ~ s(", x_col, ", k = ", k, ")"))
  m <- tryCatch(
    mgcv::gam(fo, data = dd, method = "REML"),
    error = function(e) NULL,
    warning = function(w) NULL
  )
  if (is.null(m)) {
    return(NULL)
  }
  x_lo <- min(dd[[x_col]])
  x_hi <- max(dd[[x_col]])
  if (!is.finite(x_lo) || !is.finite(x_hi) || x_hi <= x_lo) {
    return(NULL)
  }
  xgrid <- seq(x_lo, x_hi, length.out = n_grid)
  nd <- stats::setNames(data.frame(xgrid), x_col)
  pr <- tryCatch(
    stats::predict(m, newdata = nd, se.fit = TRUE),
    error = function(e) NULL
  )
  if (is.null(pr)) {
    return(NULL)
  }
  fit <- as.numeric(pr$fit)
  se <- as.numeric(pr$se.fit)
  if (any(!is.finite(fit)) || any(!is.finite(se))) {
    return(NULL)
  }
  data.frame(x = xgrid, fit = fit, ymin = fit - 1.96 * se, ymax = fit + 1.96 * se)
}

## Chunk-count weighted early vs late mean (linear model with weights = n_chunks).
weighted_late_indicator_lm <- function(d, y_col, split_year) {
  need <- c(y_col, "year", "n_chunks")
  if (!all(need %in% names(d))) {
    return(NULL)
  }
  dd <- d[is.finite(d[[y_col]]) & is.finite(d$year) & is.finite(d$n_chunks), , drop = FALSE]
  dd <- dd[dd$n_chunks > 0, , drop = FALSE]
  if (nrow(dd) < 6L) {
    return(NULL)
  }
  if (sum(dd$year < split_year) < 2L || sum(dd$year >= split_year) < 2L) {
    return(NULL)
  }
  dd$late <- as.integer(dd$year >= split_year)
  f <- stats::as.formula(paste0(y_col, " ~ late"))
  fit <- tryCatch(stats::lm(f, data = dd, weights = dd$n_chunks), error = function(e) NULL)
  if (is.null(fit)) {
    return(NULL)
  }
  sm <- summary(fit)
  rn <- rownames(sm$coefficients)
  if (!"late" %in% rn) {
    return(NULL)
  }
  wmean <- function(rows) {
    stats::weighted.mean(rows[[y_col]], rows$n_chunks, na.rm = TRUE)
  }
  e_rows <- dd[dd$year < split_year, , drop = FALSE]
  l_rows <- dd[dd$year >= split_year, , drop = FALSE]
  list(
    estimate = unname(sm$coefficients["late", "Estimate"]),
    p_value = unname(sm$coefficients["late", "Pr(>|t|)"]),
    early_wm = wmean(e_rows),
    late_wm = wmean(l_rows),
    split_year = split_year
  )
}

## Markdown bullets: Welch *t* on early/late and chunk-weighted OLS (for contrasts or single-phrase scores).
early_late_significance_md <- function(d, y_col, split_yr) {
  if (!all(c(y_col, "year", "n_chunks") %in% names(d))) {
    return("")
  }
  y1 <- d[[y_col]][d$year < split_yr]
  y2 <- d[[y_col]][d$year >= split_yr]
  welch <- if (length(y1) > 1L && length(y2) > 1L) {
    stats::t.test(y1, y2)
  } else {
    NULL
  }
  wl <- weighted_late_indicator_lm(d, y_col, split_yr)
  parts <- character(0)
  if (!is.null(welch)) {
    m_e <- mean(y1, na.rm = TRUE)
    m_l <- mean(y2, na.rm = TRUE)
    parts <- c(
      parts,
      paste0(
        "**Welch *t* test** (unweighted talks: *year* < **", split_yr, "** vs ≥ **", split_yr, "**): ",
        "means **", sprintf("%.4f", m_e), "** (early) vs **", sprintf("%.4f", m_l),
        "** (late); late − early = **", sprintf("%+.4f", m_l - m_e),
        "**, two-sided *p* ", fmt_p_num(welch$p.value), "."
      )
    )
  }
  if (!is.null(wl)) {
    parts <- c(
      parts,
      paste0(
        "**Chunk-weighted OLS** (same split; weights = *n_chunks* per talk): ",
        "late − early = ", sprintf("%+.4f", wl$estimate),
        ", *p* ", fmt_p_num(wl$p_value),
        ". Weighted means **", sprintf("%.4f", wl$early_wm), "** (early) → **",
        sprintf("%.4f", wl$late_wm), "** (late)."
      )
    )
  }
  paste(parts, collapse = "\n\n")
}

## Run embed_query_phrase.py once; returns list(ok, vec | NULL, err).
call_embed_phrase <- function(phrase, py, scr_abs, idf_abs, model) {
  phrase <- trimws(paste(phrase, collapse = "\n"))
  if (!nzchar(phrase)) {
    return(list(ok = FALSE, vec = NULL, err = "Empty phrase."))
  }
  if (nchar(phrase) > 2000L) {
    return(list(ok = FALSE, vec = NULL, err = "Phrase exceeds 2000 characters."))
  }
  phf <- tempfile(fileext = ".txt")
  errf <- tempfile(fileext = ".log")
  on.exit(unlink(c(phf, errf)), add = FALSE)
  con_p <- file(phf, open = "wb")
  writeBin(charToRaw(enc2utf8(phrase)), con_p)
  close(con_p)
  phf_abs <- normalizePath(phf, winslash = "/", mustWork = TRUE)
  args <- c(scr_abs, "--model", model, "--idf", idf_abs, "--phrase-file", phf_abs)
  res <- system2(py, args = args, stdout = TRUE, stderr = errf)
  st <- attr(res, "status")
  errtxt <- paste(readLines(errf, warn = FALSE), collapse = "\n")
  if (!is.null(st) && !is.na(st) && st != 0L) {
    return(list(
      ok = FALSE,
      vec = NULL,
      err = paste0("Python exited with status ", st, ".\n", errtxt)
    ))
  }
  u <- tryCatch(
    jsonlite::fromJSON(paste(res, collapse = "")),
    error = function(e) NULL
  )
  if (is.null(u) || length(u) != 384L) {
    return(list(
      ok = FALSE,
      vec = NULL,
      err = paste("Could not parse 384-d JSON from Python.\n", errtxt, sep = "")
    ))
  }
  list(ok = TRUE, vec = as.numeric(u), err = "")
}

## One swing (or fallback) highlight row for a talk_id — for showpiece exemplars.
pick_one_highlight <- function(talk_id) {
  talk_id <- as.character(talk_id)[1L]
  if (!isTRUE(has_chunk_highlights)) {
    return(NULL)
  }
  hi <- chunk_highlights |>
    filter(as.character(.data$talk_id) == talk_id, .data$kind == "swing")
  if (nrow(hi) < 1L) {
    hi <- chunk_highlights |>
      filter(as.character(.data$talk_id) == talk_id) |>
      slice_head(n = 1L)
  }
  if (nrow(hi) < 1L) {
    return(NULL)
  }
  hi[1L, , drop = FALSE]
}

showpiece_excerpt_card <- function(talk_id, year, pole_contrast, border_hex) {
  row <- pick_one_highlight(talk_id)
  tags$div(
    class = "card mb-3 border-0 shadow-sm",
    style = sprintf("border-left: 4px solid %s !important;", border_hex),
    tags$div(
      class = "card-body",
      tags$p(
        class = "text-muted small mb-2",
        tags$code(class = "user-select-all", as.character(talk_id)),
        " · ",
        year,
        " · Δ = ",
        tags$strong(sprintf("%+.4f", pole_contrast))
      ),
      if (is.null(row)) {
        tags$p(
          class = "small fst-italic text-muted mb-0",
          "No bundled chunk excerpt (build chunk_highlights.rds via plot_gc_chunk_embed_results.R)."
        )
      } else {
        tagList(
          tags$p(
            class = "small text-muted mb-0",
            "Illustrative passage: a ",
            tags$strong("swing"),
            " highlight from the Chunk insights tab — ",
            tags$em("not"),
            " chosen by cosine to your phrases."
          ),
          tags$blockquote(
            class = "mt-2 mb-0 ps-3",
            style = "border-left: 3px solid #cbd5e1; font-family: Georgia, serif; font-size: 0.95rem;",
            tags$p(class = "mb-0", as.character(row$text_excerpt))
          )
        )
      }
    )
  )
}

chunk_card_ui <- function(row1) {
  r <- as.list(row1[1L, , drop = FALSE])
  col <- chunk_kind_hex(r$kind)
  tags$div(
    class = "card mb-3 border-0 shadow-sm",
    style = sprintf("border-left: 4px solid %s !important;", col),
    tags$div(
      class = "card-body",
      tags$h6(class = "mb-2", style = sprintf("color:%s;font-weight:600;", col), as.character(r$kind_title)),
      tags$p(
        class = "text-muted small mb-0",
        tags$strong("Segment index "), r$chunk_idx,
        " · **Chunk net** ", sprintf("%.4f", as.numeric(r$net_presc)),
        " (*pull vs this talk’s mean:* ", sprintf("%+.4f", as.numeric(r$vs_talk_mean)), ")",
        " · cos→presc ", sprintf("%.3f", as.numeric(r$cos_presc)),
        " · cos→invit ", sprintf("%.3f", as.numeric(r$cos_gentle))
      ),
      tags$blockquote(
        class = "mt-3 mb-0 ps-3",
        style = "border-left: 3px solid #cbd5e1; font-family: Georgia, 'Times New Roman', serif; font-size: 0.95rem;",
        tags$p(class = "mb-0", as.character(r$text_excerpt))
      )
    )
  )
}

img <- function(src, alt) {
  tags$figure(
    class = "figure my-2 gallery-figure mx-auto",
    tags$img(
      src = src,
      alt = alt,
      class = "img-fluid rounded shadow border gallery-plot-img",
      loading = "lazy",
      decoding = "async"
    ),
    tags$figcaption(class = "figure-caption text-center mt-2 px-2", alt)
  )
}

ui <- tagList(
  tags$head(
    tags$style(
      HTML("
/* fillable=FALSE on page_navbar is the main fix; this backs up against nested flex shrink */
.gallery-figure {
  max-width: 100%;
  margin-bottom: 0.5rem;
}
/* Scale high-res PNGs to fit the card + viewport (no re-export needed) */
.gallery-plot-img {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: auto;
  height: auto;
  max-width: 100%;
  max-height: min(72vh, 900px);
  object-fit: contain;
  min-height: 0;
  flex-shrink: 0;
}
/* Two-up row: a bit shorter so pairs fit laptop screens */
@media (min-width: 768px) {
  .gallery-card-row-split .gallery-plot-img {
    max-height: min(58vh, 520px);
  }
}
.gallery-card .card-body { overflow-x: auto; overflow-y: visible; }
.js-plotly-plot .plotly { min-height: 70vh; }
#plt_scatter { min-height: 72vh; }
")
    )
  ),
  page_navbar(
    title = tagList(
      tags$span(style = "color:#1a365d;", "\u2606 "),
      tags$strong("General Conference"),
      tags$span(" — semantic trends explorer", class = "text-muted")
    ),
    theme = bs_theme(bootswatch = "flatly", primary = "#2c5282"),
    fillable = FALSE,
  footer = tags$footer(
    class = "text-center text-muted py-4 mt-5 border-top small px-3",
    tags$p(
      "Exploratory analysis of wording patterns over time in a public corpus (embeddings / statistics). ",
      "It is ",
      tags$strong("not"),
      " affiliated with or endorsed by The Church of Jesus Christ of Latter-day Saints."
    ),
    tags$p("Embeddings: BAAI/bge-small-en-v1.5 • Chunking: semantic-text-splitter • Corpus: generalconference R package.")
  ),

  nav_panel(
    tags$span(icon("home"), " Welcome"),
    layout_columns(
      col_widths = c(12),
      card(
        card_header(class = "bg-primary text-white", "What you are looking at"),
        card_body(
          markdown(paste0(
            "We analyzed **every General Conference talk** in a public corpus (1971–2021) using ",
            "**computer language models** (the same *kind* of AI behind ChatGPT, but used here only ",
            "to measure patterns in wording).\n\n",
            "**Question we asked:** Have talks sounded *more prescriptive* ( commanding, warning, “thou shalt not” ",
            "energy) or *more invitational* (gentle, “come unto Christ,” pondering) over time — ",
            "in a **measurable** way?\n\n",
            "**Short answer:** On our carefully defined scale, talks drift slightly toward the **invitational** ",
            "side over the decades. The effect is real but modest — this is science-flavored exploration, ",
            "not prophecy or church policy.\n\n",
            "Use the tabs above: **Gallery** for the big charts, **Explore** for the interactive scatter, ",
            "**Custom pole** for your own phrase: decadal + era charts and **trend *p*-values** (needs Python + synced embedding data), ",
            "**Chunk insights** to read passages that moved each talk’s score, and **Methods** for how it was built."
          ))
        )
      )
    ),
    layout_columns(
      col_widths = c(4, 4, 4),
      card(
        card_header("Talks in corpus"),
        card_body(h1(comma(nrow(talk_scores)), class = "display-5 text-primary text-center"))
      ),
      card(
        card_header("Semantic chunks"),
        card_body(
          h1(comma(sum(talk_scores$n_chunks)), class = "display-5 text-primary text-center"),
          tags$p(class = "small text-muted text-center", "Pieces of text the model read per talk")
        )
      ),
      card(
        card_header("Correlation year ↔ score"),
        card_body(
          h1(format(summ$cor_year_net, digits = 2), class = "display-5 text-primary text-center"),
          tags$p(class = "small text-muted text-center", "Negative ⇒ later years score lower on “net prescriptive”")
        )
      )
    ),
    card(
      card_header("What the main number means"),
      card_body(
        markdown(paste0(
          "Each talk gets a **mean net score** built from many **chunks** of text. For each chunk we measured ",
          "similarity to example sentences mined from the corpus for “prescriptive” vs “gentle / invitational” ",
          "language, then averaged. **`mean_net_presc`** = (prescriptive similarity) minus (invitational similarity). ",
          "**Below zero** means that talk’s chunks, on average, sit closer to the gentle pole than the prescriptive pole.\n\n",
          "**Smoothed change (decades):** early vs late years in the model land around ",
          sprintf("**%.3f** vs **%.3f** (change **%.3f**). ", summ$gam_fit_early_decile, summ$gam_fit_late_decile, summ$delta_late_minus_early),
          "The statistical curve (GAM) is highly unlikely to be a fluke (approx. *p* ",
          p_gam_txt,
          ")."
        ))
      )
    )
  ),

  nav_panel(
    tags$span(icon("images"), " Gallery"),
    div(
      class = "container-fluid pb-4 gallery-scroll-pane",
      card(
        class = "gallery-card mb-4",
        card_header("Full-width story panel"),
        card_body(
          class = "p-3",
          img("00_panel_trajectory_and_decades.png", "Trajectory and decadal summary combined")
        )
      ),
      card(
        class = "gallery-card mb-4",
        card_header("Every talk as a dot — curve is the trend"),
        card_body(
          class = "p-3",
          img("01_talk_trajectory_gam.png", "GAM smooth with 95% band over talk-level net scores")
        )
      ),
      layout_columns(
        class = "gallery-card-row-split",
        col_widths = c(12, 12),
        card(
          class = "gallery-card mb-4",
          card_header("By decade"),
          card_body(
            class = "p-3",
            img("02_decadal_lollipop.png", "Decadal means with error bars")
          )
        ),
        card(
          class = "gallery-card mb-4",
          card_header("Two language “poles” in embedding space"),
          card_body(
            class = "p-3",
            img("03_cosine_plane_by_year.png", "Prescriptive vs invitational cosine; color = year")
          )
        )
      ),
      card(
        class = "gallery-card mb-4",
        card_header("How distributions shift across eras"),
        card_body(
          class = "p-3",
          img("04_era_violins.png", "Violin plots by time period")
        )
      )
    )
  ),

  nav_panel(
    tags$span(icon("sliders-h"), " Explore"),
    layout_columns(
      col_widths = c(12, 12),
      card(
        card_header("Filters"),
        tags$p(
          class = "small text-muted",
          "Explore scatter uses **only** talks in this **year** window (and eras below). **Green band** = GAM smooth; **red band** = OLS line (both refit when you move the sliders)."
        ),
        sliderInput(
          "yr",
          "Year range",
          min = min(talk_scores$year, na.rm = TRUE),
          max = max(talk_scores$year, na.rm = TRUE),
          value = c(min(talk_scores$year, na.rm = TRUE), max(talk_scores$year, na.rm = TRUE)),
          sep = "",
          width = "100%"
        ),
        checkboxGroupInput(
          "era_f",
          "Era bands",
          choices = era_levels,
          selected = era_levels,
          inline = TRUE
        )
      ),
      card(
        card_header("Brush points on the scatter"),
        card_body(markdown("Click and drag on the chart to zoom; double-click to reset. Hover for talk id."))
      )
    ),
    card(
      class = "explore-plot-card",
      card_header("Interactive: year vs net prescriptive score"),
      plotlyOutput("plt_scatter", height = "72vh")
    ),
    card(
      card_header("Sample of filtered talks (first 500 rows)"),
      DTOutput("tbl")
    )
  ),

  nav_panel(
    tags$span(icon("star"), " Custom pole"),
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok)) {
      card(
        card_header("Embedding sums not bundled"),
        card_body(
          markdown(paste0(
            "This tab needs **`talk_emb_sums.rds`** and **`subword_idf.npy`** in `analysis/shiny_gc_family/data/`. ",
            "They are produced when you run the **Python** chunk-embedding pipeline (see `analysis/python/gc_chunk_embed_pipeline.py`) ",
            "and then refresh Shiny data:\n\n",
            "```\nRscript analysis/plot_gc_chunk_embed_results.R\n```\n\n",
            "That copies the new Parquet/NumPy artifacts next to **`talk_scores.parquet`** into this app folder. ",
            "If `talk_emb_sums.rds` exists but columns are wrong, re-run the **updated** Python pipeline so `talk_emb_sums.parquet` is regenerated."
          ))
        )
      )
    } else if (!isTRUE(has_py_embed_script)) {
      card(
        card_header("Python script not found"),
        card_body(
          markdown(paste0(
            "Expected `embed_query_phrase.py` at:\n\n`",
            file.path(app_dir, "..", "python", "embed_query_phrase.py"),
            "`\n\nRun the app from the repo so `analysis/python/` is one level above this folder."
          ))
        )
      )
    } else {
      tagList(
        accordion(
          open = FALSE,
          accordion_panel(
            "What does this do?",
            markdown(paste0(
              "**What you’re measuring:** For your phrase we build one **384-d “direction”** in the same embedding space as every talk’s semantic chunks. ",
              "Each talk’s score is the **average cosine similarity** between chunk vectors and that direction (range is roughly −1 to 1; typical values are modest).\n\n",
              "**How to read the charts:** (1) **Every talk** — scatter + smooth curve vs calendar year. ",
              "(2) **By decade** — mean ± error bar per decade so the stair-step is obvious. ",
              "(3) **By era band** — full distribution shape, not just the mean. ",
              "(4) **Trend tests** — Pearson linear correlation and a **GAM** smooth test (is a curved trend better than a flat line?).\n\n",
              "**Not keyword counts** — semantic similarity. **First embed** may take a minute while PyTorch loads. ",
              "Python: `pip install -r analysis/python/requirements-gc-embed.txt`. Filters below apply **only** to this tab."
            ))
          )
        ),
        card(
          card_header("Filters (Custom pole only)"),
          tags$p(
            class = "small text-muted px-3 pt-2 mb-0",
            "Every chart and **OLS/GAM** statistic on this tab uses only talks matching **year** + **era** here."
          ),
          layout_columns(
            col_widths = c(12, 12),
            sliderInput(
              "cust_yr",
              "Year range",
              min = cust_yr_min,
              max = cust_yr_max,
              value = c(cust_yr_min, cust_yr_max),
              sep = "",
              width = "100%"
            ),
            checkboxGroupInput(
              "cust_era_f",
              "Era bands",
              choices = era_levels,
              selected = era_levels,
              inline = TRUE
            )
          )
        ),
        card(
          card_header("Your phrase → embed → plots below"),
          card_body(
            textAreaInput(
              "custom_phrase",
              label = "Phrase or sentence to embed",
              value = "",
              rows = 3L,
              placeholder = "e.g. covenant, or a full short sentence you care about",
              width = "100%"
            ),
            layout_columns(
              col_widths = c(6, 6),
              actionButton(
                "run_custom_embed",
                label = "Embed & update charts",
                class = "btn-primary",
                width = "100%",
                icon = icon("play")
              ),
              tags$p(
                class = "small text-muted mb-0 align-self-center",
                "Optional: set env ",
                tags$code("CONFERENCESTATS_PYTHON"),
                " to your venv’s Python if ",
                tags$code("python3"),
                " is wrong."
              )
            ),
            verbatimTextOutput("custom_embed_err", placeholder = TRUE)
          )
        ),
        uiOutput("custom_stats_card"),
        card(
          class = "explore-plot-card",
          card_header(uiOutput("custom_plot_title")),
          plotlyOutput("plt_custom_pole", height = "60vh")
        ),
        layout_columns(
          col_widths = c(12, 12),
          card(
            card_header("Decadal means (±1 SE)"),
            plotlyOutput("plt_custom_decade", height = "420px")
          ),
          card(
            card_header("Distribution by era band"),
            plotlyOutput("plt_custom_era", height = "420px")
          )
        ),
        card(
          card_header("Showpiece — two-phrase contrast"),
          card_body(
            markdown(paste0(
              "**Story arc:** Phrase **A** and phrase **B** each define a direction in embedding space. ",
              "Every talk gets two scores (mean chunk cosine to each direction). We plot **A − B** vs year: ",
              "positive ⇒ chunks sit **closer to A**; negative ⇒ **closer to B**. This is geometry, not theology — but the **spread** is where dinner-table arguments live.\n\n",
              "At the bottom, **exemplar talks** are the strongest tilts either way in your current filters. ",
              "Quoted passages are **swing-lemma highlights** from the Chunk insights bundle (not passages selected by your phrases). ",
              "Two Python runs per click (warm model ≈ quick)."
            )),
            layout_columns(
              col_widths = c(12, 12),
              textAreaInput(
                "sp_phrase_a",
                label = "Phrase A (minuend / “toward the red side” in exemplars)",
                value = "obedience covenant faithfulness",
                rows = 2L,
                width = "100%"
              ),
              textAreaInput(
                "sp_phrase_b",
                label = "Phrase B (subtrahend / “toward the green side” in exemplars)",
                value = "grace compassion inclusion healing",
                rows = 2L,
                width = "100%"
              )
            ),
            actionButton(
              "run_showpiece_embed",
              label = "Embed both phrases & build showpiece",
              class = "btn-outline-primary",
              width = "100%",
              icon = icon("arrows-alt-h")
            ),
            verbatimTextOutput("showpiece_embed_err", placeholder = TRUE)
          )
        ),
        uiOutput("showpiece_stats_card"),
        card(
          class = "explore-plot-card",
          card_header(uiOutput("showpiece_plot_hdr")),
          plotlyOutput("plt_showpiece", height = "56vh")
        ),
        uiOutput("showpiece_exemplars")
      )
    }
  ),

  nav_panel(
    tags$span(icon("clock"), " 1996–today / contrast"),
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok)) {
      card(
        card_header("Embedding bundle required"),
        card_body(
          markdown("Same setup as **Custom pole** — sync `talk_emb_sums.rds` and `subword_idf.npy` into `analysis/shiny_gc_family/data/`.")
        )
      )
    } else if (!isTRUE(has_py_embed_script)) {
      card(
        card_header("Python script not found"),
        card_body(markdown(paste0("Expected `embed_query_phrase.py` under `analysis/python/`.")))
      )
    } else {
      tagList(
        card(
          card_header("Contrastive lens (research use)"),
          card_body(
            markdown(paste0(
              "This tab fixes a **modern window** (defaults **1996 → latest year**) and focuses on **two phrases at once**: ",
              "each talk gets mean cosines to **A** and **B**, and we analyze **A − B** (relative geometry, not morality). ",
              "That is the right framing for “how does usage **lean** between two related wordings over time?” — ",
              "including pairs you choose yourself (you type the text; nothing is hard-coded here).\n\n",
              "**Statistics** respect your sliders: **correlation** uses only talks in the year band; **early vs late** splits on the **split year** ",
              "(Welch *t* + chunk-weighted OLS). **Plots** use an explicit **OLS line + 95% band** (same as refreshed Explore / Custom plots) so the fit is always visible in Plotly."
            ))
          )
        ),
        card(
          card_header("Year window (this tab only)"),
          layout_columns(
            col_widths = c(12),
            sliderInput(
              "tw_yr",
              "Include talks with conference year in this range",
              min = cust_yr_min,
              max = cust_yr_max,
              value = c(tw_tab_yr_lo_default, cust_yr_max),
              sep = "",
              width = "100%"
            ),
            sliderInput(
              "tw_split_yr",
              "Split year for early vs late tests (same window)",
              min = cust_yr_min,
              max = cust_yr_max,
              value = round(mean(c(tw_tab_yr_lo_default, cust_yr_max))),
              sep = "",
              width = "100%"
            )
          )
        ),
        card(
          card_header("Two phrases → embed → contrast plot"),
          card_body(
            layout_columns(
              col_widths = c(12, 12),
              textAreaInput(
                "tw_sp_phrase_a",
                label = "Phrase A (reference / ‘numerator’ cosine)",
                value = "",
                rows = 2L,
                placeholder = "first wording to compare",
                width = "100%"
              ),
              textAreaInput(
                "tw_sp_phrase_b",
                label = "Phrase B (comparison / ‘denominator’ cosine)",
                value = "",
                rows = 2L,
                placeholder = "second wording (e.g. neutral counterpart)",
                width = "100%"
              )
            ),
            actionButton(
              "tw_run_showpiece_embed",
              label = "Embed A & B (uses same Python stack as Custom pole)",
              class = "btn-primary",
              width = "100%",
              icon = icon("arrows-alt-h")
            ),
            verbatimTextOutput("tw_showpiece_embed_err", placeholder = TRUE)
          )
        ),
        uiOutput("tw_show_stats_card"),
        card(
          class = "explore-plot-card",
          card_header(uiOutput("tw_show_plot_hdr")),
          plotlyOutput("plt_tw_showpiece", height = "56vh")
        ),
        uiOutput("tw_showpiece_exemplars")
      )
    }
  ),

  nav_panel(
    tags$span(icon("search-plus"), " Chunk insights"),
    if (!has_chunk_highlights) {
      card(
        card_header("Inspector data not bundled"),
        card_body(
          markdown(paste0(
            "This tab needs **`chunk_highlights.rds`**, built from the full **`chunks_scored.parquet`** output of the Python pipeline.\n\n",
            "**From the repo root**, after `chunks_scored.parquet` exists next to `talk_scores.parquet`, run:\n\n",
            "```\nRscript analysis/plot_gc_chunk_embed_results.R\n```\n\n",
            "That refreshes plots and writes **`analysis/shiny_gc_family/data/chunk_highlights.rds`** (then restart the app)."
          ))
        )
      )
    } else {
      tagList(
        accordion(
          open = FALSE,
          accordion_panel(
            "Why these passages? (read me)",
            markdown(paste0(
              "**The headline score for a talk** is the *average* of many **semantic chunks** (~220 words each). ",
              "Every chunk gets a **net** score: how similar that chunk sounds to our **prescriptive** example sentences ",
              "(commands, warnings, “beware,” etc.) *minus* how similar it sounds to **gentle / invitational** examples ",
              "(“I invite,” “ponder,” tenderness toward Christ).\n\n",
              "**“Meaningful” here means three different things** — all *within the same talk*, so you are comparing apples to apples:\n\n",
              "1. **Highest net in this talk** — the chunks with the **largest** (least negative) net scores *among this talk’s segments*. ",
              "That is **not** the same as “sounds prescriptive on an absolute scale”: if every chunk in the talk sits on the gentle side of the corpus, these winners can **still have negative net** — they’re just the *least* gentle slices *here*.\n",
              "2. **Lowest net in this talk** — the **most** invitational-leaning segments relative to this talk’s other chunks.\n",
              "3. **Leverage** — chunks farthest from **this talk’s mean** chunk score (big pull on the headline average). ",
              "We skip segments already listed in (1) or (2) so you don’t see the same index twice.\n\n",
              "**Light cleaning:** ultra-short closings that mostly end with “amen” (etc.) are usually dropped from these picks so they don’t crowd out substantive paragraphs. ",
              "The **last semantic chunk of each talk** is also skipped when possible — that’s almost always the peroration (“in the name of…”, “amen”), and it isn’t comparable to body paragraphs. ",
              "Very short talks automatically fall back to using every chunk so you still get highlights. ",
              "Quoted text is whitespace-normalized. Rebuild **`chunk_highlights.rds`** with `Rscript analysis/plot_gc_chunk_embed_results.R` after pipeline changes.\n\n",
              "**Not a verdict on holiness** — only geometry next to hand-picked regex-mined examples, using one embedding model. ",
              "A chunk can be theologically rich but score “invitational” if its wording matches our gentle prototypes more than our prohibitive ones."
            ))
          )
        ),
        layout_columns(
          col_widths = c(6, 6),
          card(
            card_header("Pick a session year"),
            selectInput(
              "chunk_year",
              NULL,
              choices = years_for_pick,
              selected = max(years_for_pick)
            )
          ),
          card(
            card_header("Pick a talk (id hash)"),
            selectInput(
              "chunk_talk",
              NULL,
              choices = NULL,
              selectize = TRUE
            )
          )
        ),
        uiOutput("chunk_talk_summary"),
        uiOutput("chunk_cards")
      )
    }
  ),

  nav_panel(
    tags$span(icon("flask"), " Methods"),
    card(
      card_header("Pipeline (high level)"),
      card_body(
        markdown(paste0(
          "1. **Corpus:** `generalconference` R package — one row per talk with full text.\n",
          "2. **Chunking:** *Semantic text splitter* with the same tokenizer as the embedding model (~220 tokens per chunk).\n",
          "3. **Model:** `BAAI/bge-small-en-v1.5` (384 numbers per chunk).\n",
          "4. **Pooling:** *Subword tf–idf weights* on the transformer’s last layer — “sicko mode” attention to rare-ish words.\n",
          "5. **Poles:** Real sentences from talks matching regex bundles for prescriptive vs gentle language; average their vectors.\n",
          "6. **Score:** For each chunk: cosine to prescriptive pole minus cosine to gentle pole. **Talk score = mean over chunks.**\n",
          "7. **Trend:** `mgcv` GAM smooth on year in R; figures saved under `analysis/output/gc_chunk_embed/`.\n",
          "8. **Custom pole (Shiny tab):** Precomputed **sums of chunk embedding vectors** per talk plus frozen **subword IDF** let you embed any short phrase with Python and plot **mean cosine** vs year (same pooling as the main pipeline).\n",
          "9. **Showpiece:** embed **two** phrases and plot **(cosine to A) − (cosine to B)** vs year, with exemplar talks (quotes from the chunk inspector, not phrase-selected).\n",
          "10. **1996–today / contrast tab:** same two-phrase machinery with a **dedicated year window** (default 1996–latest), **split-year** controls for early/late tests, and **explicit OLS lines** on scatter plots.\n\n",
          "The Python driver is `analysis/python/gc_chunk_embed_pipeline.py`. ",
          "A plain-language report lives in `analysis/prescriptive_chunks_embed_report.Rmd`."
        ))
      )
    ),
    card(
      card_header("Caveats"),
      card_body(
        markdown(paste0(
          "- **Not theology** — geometry of language in one embedding model.\n",
          "- **Regex exemplars** steer what “prescriptive” means; different examples move the needle.\n",
          "- **Chunk size** matters; we picked a defensible default for BGE’s context length.\n",
          "- **Aggregate vs detail** — individual talks, speakers, and years can contradict any corpus-wide smooth; qualitative reading always matters."
        ))
      )
    )
  ),

  nav_panel(
    tags$span(icon("heart"), " About"),
    card(
      card_header("About"),
      card_body(
        markdown(paste0(
          "This app bundles **NLP-style embeddings** and simple **time-series / exploratory statistics** on General Conference talks in a public corpus — ",
          "interesting patterns in language over time, not theology or institutional claims.\n\n",
          "**Repo:** `conference-stats` on GitHub (same project as the Parquet corpus helpers).\n\n",
          "Made with R **Shiny**, **ggplot2**, **plotly**, and **mgcv**. Methods and statistical caveats: ",
          "[`documentation/methods-and-statistical-inference.md`]",
          "(https://github.com/AdrielC/conference-stats/blob/main/documentation/methods-and-statistical-inference.md)."
        ))
      )
    )
  )
  )
)

server <- function(input, output, session) {
  custom_state <- reactiveValues(mean_cos = NULL, phrase = NULL, err = "")
  showpiece_state <- reactiveValues(
    mean_cos_a = NULL,
    mean_cos_b = NULL,
    phrase_a = NULL,
    phrase_b = NULL,
    err = ""
  )

  resolve_embed_python <- function() {
    py <- Sys.getenv("CONFERENCESTATS_PYTHON", unset = "")
    if (!nzchar(py)) {
      py <- Sys.which("python3")
    }
    if (!nzchar(py)) {
      py <- Sys.which("python")
    }
    py
  }

  two_pole_finish_showpiece <- function(pha, phb) {
    st <- new.env(parent = emptyenv())
    st$ok <- FALSE
    st$err <- ""
    py <- resolve_embed_python()
    if (!nzchar(py)) {
      st$err <- "No python3 or python on PATH. Set CONFERENCESTATS_PYTHON to your interpreter."
      return(list(ok = FALSE, err = st$err))
    }
    idf_abs <- normalizePath(idf_bundle_path, winslash = "/", mustWork = TRUE)
    scr_abs <- normalizePath(py_embed_script, winslash = "/", mustWork = TRUE)
    model <- as.character(pipe_meta$model)
    withProgress(message = "Two-phrase embedding", min = 0, max = 1, {
      incProgress(0.12, detail = "Phrase A…")
      ra <- call_embed_phrase(pha, py, scr_abs, idf_abs, model)
      if (!isTRUE(ra$ok)) {
        st$err <- paste0("Phrase A: ", ra$err)
      } else {
        incProgress(0.48, detail = "Phrase B…")
        rb <- call_embed_phrase(phb, py, scr_abs, idf_abs, model)
        if (!isTRUE(rb$ok)) {
          st$err <- paste0("Phrase B: ", rb$err)
        } else {
          incProgress(0.82, detail = "Computing per-talk contrast…")
          S <- as.matrix(talk_emb_sums_tbl[, s_cols_custom])
          mca <- as.numeric(S %*% matrix(ra$vec, ncol = 1L)) / talk_emb_sums_tbl$n_chunks
          mcb <- as.numeric(S %*% matrix(rb$vec, ncol = 1L)) / talk_emb_sums_tbl$n_chunks
          showpiece_state$mean_cos_a <- mca
          showpiece_state$mean_cos_b <- mcb
          showpiece_state$phrase_a <- pha
          showpiece_state$phrase_b <- phb
          st$ok <- TRUE
        }
      }
    })
    list(ok = isTRUE(st$ok), err = as.character(st$err))
  }

  finalize_showpiece_attempt <- function(res) {
    if (isTRUE(res$ok)) {
      showpiece_state$err <- ""
      showNotification("Two-phrase contrast ready.", type = "message")
    } else {
      showpiece_state$mean_cos_a <- NULL
      showpiece_state$mean_cos_b <- NULL
      showpiece_state$phrase_a <- NULL
      showpiece_state$phrase_b <- NULL
      showpiece_state$err <- res$err
      showNotification("Embedding failed — see message below the button.", type = "error")
    }
  }

  observeEvent(input$run_custom_embed, {
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok) || !isTRUE(has_py_embed_script)) {
      return(invisible(NULL))
    }
    phrase <- paste(trimws(input$custom_phrase), collapse = "\n")
    phrase <- trimws(phrase)
    if (!nzchar(phrase)) {
      showNotification("Enter text to embed.", type = "warning")
      return(invisible(NULL))
    }
    if (nchar(phrase) > 2000L) {
      showNotification("Please keep the phrase under 2000 characters.", type = "warning")
      return(invisible(NULL))
    }
    custom_state$err <- ""
    py <- resolve_embed_python()
    if (!nzchar(py)) {
      custom_state$err <- "No python3 or python on PATH. Set CONFERENCESTATS_PYTHON to your interpreter."
      showNotification(custom_state$err, type = "error")
      return(invisible(NULL))
    }
    idf_abs <- normalizePath(idf_bundle_path, winslash = "/", mustWork = TRUE)
    scr_abs <- normalizePath(py_embed_script, winslash = "/", mustWork = TRUE)
    model <- as.character(pipe_meta$model)
    res <- withProgress(
      message = "Embedding phrase",
      detail = "First run loads the model into memory (~1 min).",
      value = 0.35,
      expr = {
        call_embed_phrase(phrase, py, scr_abs, idf_abs, model)
      }
    )
    if (!isTRUE(res$ok)) {
      custom_state$err <- res$err
      showNotification("Embedding failed — see message below the button.", type = "error")
      return(invisible(NULL))
    }
    u <- res$vec
    S <- as.matrix(talk_emb_sums_tbl[, s_cols_custom])
    mc <- as.numeric(S %*% matrix(u, ncol = 1L)) / talk_emb_sums_tbl$n_chunks
    custom_state$mean_cos <- mc
    custom_state$phrase <- phrase
  })

  observeEvent(input$run_showpiece_embed, {
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok) || !isTRUE(has_py_embed_script)) {
      return(invisible(NULL))
    }
    pha <- trimws(paste(trimws(input$sp_phrase_a), collapse = "\n"))
    phb <- trimws(paste(trimws(input$sp_phrase_b), collapse = "\n"))
    if (!nzchar(pha) || !nzchar(phb)) {
      showNotification("Enter both phrase A and phrase B.", type = "warning")
      return(invisible(NULL))
    }
    if (nchar(pha) > 2000L || nchar(phb) > 2000L) {
      showNotification("Keep each phrase under 2000 characters.", type = "warning")
      return(invisible(NULL))
    }
    if (pha == phb) {
      showNotification("Use two different phrases for a contrast.", type = "warning")
      return(invisible(NULL))
    }
    res <- two_pole_finish_showpiece(pha, phb)
    finalize_showpiece_attempt(res)
  })

  observeEvent(input$tw_run_showpiece_embed, {
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok) || !isTRUE(has_py_embed_script)) {
      return(invisible(NULL))
    }
    pha <- trimws(paste(trimws(input$tw_sp_phrase_a), collapse = "\n"))
    phb <- trimws(paste(trimws(input$tw_sp_phrase_b), collapse = "\n"))
    if (!nzchar(pha) || !nzchar(phb)) {
      showNotification("Enter both phrases on this tab.", type = "warning")
      return(invisible(NULL))
    }
    if (nchar(pha) > 2000L || nchar(phb) > 2000L) {
      showNotification("Keep each phrase under 2000 characters.", type = "warning")
      return(invisible(NULL))
    }
    if (pha == phb) {
      showNotification("Use two different phrases for a contrast.", type = "warning")
      return(invisible(NULL))
    }
    res <- two_pole_finish_showpiece(pha, phb)
    finalize_showpiece_attempt(res)
  })

  output$custom_embed_err <- renderText({
    e <- custom_state$err
    if (is.null(e) || !nzchar(e)) {
      return("")
    }
    e
  })

  custom_talks_d <- reactive({
    req(custom_state$mean_cos)
    req(length(input$cust_era_f) > 0L)
    te <- talk_emb_sums_tbl
    te$mean_cos_custom <- custom_state$mean_cos
    talk_scores |>
      inner_join(
        te |> select("talk_id", "year", "mean_cos_custom"),
        by = c("talk_id", "year")
      ) |>
      filter(
        .data$year >= input$cust_yr[[1L]],
        .data$year <= input$cust_yr[[2L]],
        .data$era %in% input$cust_era_f
      )
  })

  showpiece_talks_d <- reactive({
    req(showpiece_state$mean_cos_a)
    req(showpiece_state$mean_cos_b)
    req(length(input$cust_era_f) > 0L)
    te <- talk_emb_sums_tbl
    mca <- showpiece_state$mean_cos_a
    mcb <- showpiece_state$mean_cos_b
    if (length(mca) != nrow(te) || length(mcb) != nrow(te)) {
      return(talk_scores[FALSE, ])
    }
    te$pole_a <- mca
    te$pole_b <- mcb
    te$pole_contrast <- mca - mcb
    talk_scores |>
      inner_join(
        te |> select("talk_id", "year", "pole_contrast", "pole_a", "pole_b"),
        by = c("talk_id", "year")
      ) |>
      filter(
        .data$year >= input$cust_yr[[1L]],
        .data$year <= input$cust_yr[[2L]],
        .data$era %in% input$cust_era_f
      )
  })

  showpiece_talks_tw <- reactive({
    req(showpiece_state$mean_cos_a)
    req(showpiece_state$mean_cos_b)
    req(!is.null(input$tw_yr))
    te <- talk_emb_sums_tbl
    mca <- showpiece_state$mean_cos_a
    mcb <- showpiece_state$mean_cos_b
    if (length(mca) != nrow(te) || length(mcb) != nrow(te)) {
      return(talk_scores[FALSE, ])
    }
    te$pole_a <- mca
    te$pole_b <- mcb
    te$pole_contrast <- mca - mcb
    talk_scores |>
      inner_join(
        te |> select("talk_id", "year", "pole_contrast", "pole_a", "pole_b"),
        by = c("talk_id", "year")
      ) |>
      filter(
        .data$year >= input$tw_yr[[1L]],
        .data$year <= input$tw_yr[[2L]]
      )
  })

  output$custom_plot_title <- renderUI({
    ph <- custom_state$phrase
    if (is.null(ph) || !nzchar(ph)) {
      return(tags$h6(class = "text-muted mb-0", "Run a query to plot year vs mean cosine"))
    }
    tags$div(
      class = "mb-0",
      "Year vs mean cosine to ",
      tags$strong(style = "color:#1a365d;", ph)
    )
  })

  output$custom_stats_card <- renderUI({
    if (is.null(custom_state$mean_cos)) {
      return(NULL)
    }
    d <- tryCatch(custom_talks_d(), error = function(e) NULL)
    if (is.null(d) || nrow(d) < 12L) {
      return(
        card(
          card_header("Trend & significance (after filters)"),
          card_body(
            markdown(
              paste0(
                "After you **embed a phrase**, statistics appear here. Need more talks in view ",
                "(widen **year range** or select more **era** boxes). Currently **",
                if (is.null(d)) 0L else nrow(d), "** talks."
              )
            )
          )
        )
      )
    }
    ct <- tryCatch(
      stats::cor.test(d$year, d$mean_cos_custom, method = "pearson"),
      error = function(e) NULL
    )
    r_line <- if (is.null(ct)) {
      "*Could not compute Pearson correlation (check variation in year/score).*"
    } else {
      paste0(
        "**Linear (Pearson):** *r* = ", sprintf("%.3f", unname(ct$estimate)),
        ", two-sided *p* ", fmt_p_num(ct$p.value), ". ",
        if (ct$p.value < 0.05) {
          "Conventionally **significant** at α = 0.05 (correlation is unlikely if there were no association). "
        } else {
          "Not significant at α = 0.05 (weak or inconsistent linear trend in this window). "
        }
      )
    }
    mg <- tryCatch(
      mgcv::gam(mean_cos_custom ~ s(year, k = 10), data = d, method = "REML"),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    gam_line <- if (is.null(mg)) {
      "\n\n**GAM smooth:** could not fit (try more talks)."
    } else {
      p_g <- summary(mg)$s.table[1L, "p-value"]
      edf <- summary(mg)$s.table[1L, "edf"]
      paste0(
        "\n\n**Flexible curve (GAM):** effective df ≈ ", sprintf("%.2f", edf),
        ", approximate *p* ", fmt_p_num(p_g), " for the smooth term. ",
        "Small *p* suggests the trajectory isn’t flat **if** you trust the smooth model in this window."
      )
    }
    qlo <- stats::quantile(d$year, 0.1)
    qhi <- stats::quantile(d$year, 0.9)
    early <- mean(d$mean_cos_custom[d$year <= qlo])
    late <- mean(d$mean_cos_custom[d$year >= qhi])
    span <- sprintf(
      "**Scale check:** talk scores in this window run from about **%.3f** to **%.3f**; ",
      min(d$mean_cos_custom, na.rm = TRUE),
      max(d$mean_cos_custom, na.rm = TRUE)
    )
    decile_line <- paste0(
      "**Early vs late (10th vs 90th percentile years in this filter):** mean cosine ",
      sprintf("%.4f", early), " → ", sprintf("%.4f", late),
      " (Δ = ", sprintf("%+.4f", late - early), ")."
    )
    spl <- round(stats::median(d$year))
    el_md <- early_late_significance_md(d, "mean_cos_custom", spl)
    el_block <- if (nzchar(el_md)) {
      paste0(
        "\n\n**Temporal split at median year **", spl,
        "** (Welch *t* + chunk-weighted OLS on *mean cosine*):**\n\n",
        el_md
      )
    } else {
      ""
    }
    card(
      card_header("How strong is the trend?"),
      card_body(
        markdown(paste0(
          "**Talks in view:** ", nrow(d), " · **Metric:** mean cosine between your phrase and each talk’s chunks.\n\n",
          r_line, gam_line, "\n\n", decile_line, el_block, "\n\n", span,
          " Statistical *p*-values assume **independent** talks (ignores repeated speakers / series correlation) — interpret as exploratory."
        ))
      )
    )
  })

  output$plt_custom_pole <- renderPlotly({
    d <- custom_talks_d()
    req(nrow(d) > 0L)
    ct <- tryCatch(
      stats::cor.test(d$year, d$mean_cos_custom, method = "pearson"),
      error = function(e) NULL
    )
    lmdf <- lm_line_ribbon_df(d, "year", "mean_cos_custom")
    g <- ggplot(d, aes(year, mean_cos_custom, text = talk_id, size = n_chunks)) +
      geom_point(alpha = 0.25, color = "#276749")
    if (!is.null(lmdf)) {
      g <- g +
        geom_ribbon(
          data = lmdf,
          aes(x = x, ymin = ymin, ymax = ymax),
          inherit.aes = FALSE,
          alpha = 0.22,
          fill = "#9b2c2c"
        ) +
        geom_line(
          data = lmdf,
          aes(x = x, y = fit),
          inherit.aes = FALSE,
          color = "#9b2c2c",
          linewidth = 0.95
        )
    }
    g <- g +
      scale_size(range = c(1, 6), guide = "none") +
      labs(
        x = "Conference year",
        y = "Mean cosine (semantic alignment with your phrase)"
      ) +
      theme_minimal(base_size = 13)
    plt <- plotly::ggplotly(g, tooltip = "text")
    if (!is.null(ct)) {
      plt <- plotly::layout(
        plt,
        hovermode = "closest",
        annotations = list(
          list(
            text = sprintf(
              paste0(
                "<b>OLS line + 95% CI</b><br>",
                "Pearson <i>r</i> (year, score) = %.3f<br>",
                "two-sided <i>p</i> = %s"
              ),
              unname(ct$estimate),
              fmt_p_num(ct$p.value)
            ),
            xref = "paper",
            yref = "paper",
            x = 0.02,
            y = 0.98,
            xanchor = "left",
            yanchor = "top",
            showarrow = FALSE,
            align = "left",
            font = list(size = 12, color = "#1a365d"),
            bgcolor = "rgba(255,255,255,0.92)",
            bordercolor = "#cbd5e1",
            borderwidth = 1,
            borderpad = 5
          )
        )
      )
    } else {
      plt <- plotly::layout(plt, hovermode = "closest")
    }
    plt
  })

  output$plt_custom_decade <- renderPlotly({
    d <- custom_talks_d()
    req(nrow(d) > 0L)
    dec <- d |>
      mutate(decade = floor(.data$year / 10) * 10) |>
      group_by(.data$decade) |>
      summarise(
        n = dplyr::n(),
        m = mean(.data$mean_cos_custom),
        se = stats::sd(.data$mean_cos_custom) / sqrt(.data$n),
        .groups = "drop"
      )
    req(nrow(dec) > 0L)
    lmdf <- lm_line_ribbon_df(dec, "decade", "m")
    if (!is.null(lmdf)) {
      lmdf$dx <- factor(lmdf$x, levels = sort(unique(dec$decade)))
    }
    ## Custom cosines are usually positive; lollipops anchored at y=0 wasted vertical space.
    g <- ggplot(dec, aes(factor(decade), m, text = paste0("Decade ", decade, "\nn=", n))) +
      geom_point(aes(size = n), color = "#9b2c2c", alpha = 0.95) +
      geom_errorbar(
        aes(ymin = m - se, ymax = m + se),
        width = 0.22,
        color = "#2c5282",
        linewidth = 0.45,
        alpha = 0.75
      )
    if (!is.null(lmdf)) {
      g <- g +
        geom_ribbon(
          data = lmdf,
          aes(x = dx, ymin = ymin, ymax = ymax, group = 1),
          inherit.aes = FALSE,
          alpha = 0.18,
          fill = "#9b2c2c"
        ) +
        geom_line(
          data = lmdf,
          aes(x = dx, y = fit, group = 1),
          inherit.aes = FALSE,
          color = "#742d2d",
          linewidth = 0.85
        )
    }
    g <- g +
      scale_size(range = c(4, 14), breaks = pretty_breaks(), name = "Talks") +
      labs(x = "Decade", y = "Mean cosine ± 1 SE") +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom")
    ggplotly(g, tooltip = "text") |> layout(hovermode = "closest")
  })

  output$plt_custom_era <- renderPlotly({
    d <- custom_talks_d()
    req(nrow(d) > 5L)
    g <- ggplot(d, aes(factor(era), mean_cos_custom, fill = factor(era))) +
      geom_violin(alpha = 0.88, color = NA, scale = "width", width = 0.95) +
      stat_summary(fun = median, geom = "point", color = "white", size = 1.6, shape = 18) +
      scale_fill_manual(values = c("#5c7cba", "#c9a227", "#c05621", "#742d2d"), guide = "none") +
      labs(x = NULL, y = "Mean cosine (per talk)") +
      theme_minimal(base_size = 13)
    ggplotly(g) |> layout(hovermode = "closest")
  })

  output$showpiece_embed_err <- renderText({
    e <- showpiece_state$err
    if (is.null(e) || !nzchar(e)) {
      return("")
    }
    e
  })

  output$showpiece_plot_hdr <- renderUI({
    pa <- showpiece_state$phrase_a
    pb <- showpiece_state$phrase_b
    if (is.null(pa) || is.null(pb) || !nzchar(pa) || !nzchar(pb)) {
      return(tags$h6(class = "text-muted mb-0", "Run the showpiece button to plot Δ cosine vs year"))
    }
    tags$div(
      class = "mb-0 small",
      "Δ mean cosine ",
      tags$strong(style = "color:#742d2d;", "A"),
      " − ",
      tags$strong(style = "color:#276749;", "B"),
      " vs year · ",
      tags$span(style = "color:#742d2d;", tags$strong("A:"), " ", pa),
      " · ",
      tags$span(style = "color:#276749;", tags$strong("B:"), " ", pb)
    )
  })

  output$showpiece_stats_card <- renderUI({
    if (is.null(showpiece_state$mean_cos_a)) {
      return(NULL)
    }
    d <- tryCatch(showpiece_talks_d(), error = function(e) NULL)
    if (is.null(d) || nrow(d) < 12L) {
      return(
        card(
          card_header("Showpiece — trend & significance"),
          card_body(
            markdown(
              paste0(
                "After you **build the showpiece**, statistics appear when at least **12** talks match your filters. ",
                "Currently **", if (is.null(d)) 0L else nrow(d), "** talks."
              )
            )
          )
        )
      )
    }
    ct <- tryCatch(
      stats::cor.test(d$year, d$pole_contrast, method = "pearson"),
      error = function(e) NULL
    )
    r_line <- if (is.null(ct)) {
      "*Could not compute Pearson correlation.*"
    } else {
      paste0(
        "**Linear (Pearson)** on **(year, A − B):** *r* = ", sprintf("%.3f", unname(ct$estimate)),
        ", two-sided *p* ", fmt_p_num(ct$p.value), ". ",
        if (ct$p.value < 0.05) {
          "Conventionally **significant** at α = 0.05 (exploratory; talks not independent). "
        } else {
          "Not significant at α = 0.05. "
        }
      )
    }
    mg <- tryCatch(
      mgcv::gam(pole_contrast ~ s(year, k = 10), data = d, method = "REML"),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    gam_line <- if (is.null(mg)) {
      "\n\n**GAM:** could not fit."
    } else {
      p_g <- summary(mg)$s.table[1L, "p-value"]
      edf <- summary(mg)$s.table[1L, "edf"]
      paste0(
        "\n\n**GAM smooth:** edf ≈ ", sprintf("%.2f", edf),
        ", approximate *p* ", fmt_p_num(p_g), " for the smooth term."
      )
    }
    qlo <- stats::quantile(d$year, 0.1)
    qhi <- stats::quantile(d$year, 0.9)
    early <- mean(d$pole_contrast[d$year <= qlo])
    late <- mean(d$pole_contrast[d$year >= qhi])
    decile_line <- paste0(
      "**Early vs late** (10th vs 90th percentile years in filter): mean (A − B) ",
      sprintf("%+.4f", early), " → ", sprintf("%+.4f", late),
      " (Δ = ", sprintf("%+.4f", late - early), ")."
    )
    spl <- round(stats::median(d$year))
    el_md <- early_late_significance_md(d, "pole_contrast", spl)
    el_block <- if (nzchar(el_md)) {
      paste0(
        "\n\n**Median-year split on (A − B) at **", spl,
        "** (Welch *t* + chunk-weighted OLS):**\n\n",
        el_md
      )
    } else {
      ""
    }
    card(
      card_header("Showpiece — how the gap moves"),
      card_body(
        markdown(paste0(
          "**Talks in view:** ", nrow(d), " · **Metric:** mean cosine to **A** minus mean cosine to **B** (per talk).\n\n",
          r_line, gam_line, "\n\n", decile_line, el_block
        ))
      )
    )
  })

  output$plt_showpiece <- renderPlotly({
    d <- showpiece_talks_d()
    req(nrow(d) > 0L)
    ct <- tryCatch(
      stats::cor.test(d$year, d$pole_contrast, method = "pearson"),
      error = function(e) NULL
    )
    lmdf <- lm_line_ribbon_df(d, "year", "pole_contrast")
    g <- ggplot(d, aes(year, pole_contrast, text = talk_id, size = n_chunks)) +
      geom_hline(yintercept = 0, linetype = 3, linewidth = 0.35, color = "gray55") +
      geom_point(alpha = 0.28, color = "#553c9a")
    if (!is.null(lmdf)) {
      g <- g +
        geom_ribbon(
          data = lmdf,
          aes(x = x, ymin = ymin, ymax = ymax),
          inherit.aes = FALSE,
          alpha = 0.22,
          fill = "#9b2c2c"
        ) +
        geom_line(
          data = lmdf,
          aes(x = x, y = fit),
          inherit.aes = FALSE,
          color = "#9b2c2c",
          linewidth = 0.95
        )
    }
    g <- g +
      scale_size(range = c(1, 6), guide = "none") +
      labs(
        x = "Conference year",
        y = "Δ mean cosine (phrase A − phrase B)"
      ) +
      theme_minimal(base_size = 13)
    plt <- plotly::ggplotly(g, tooltip = "text")
    if (!is.null(ct)) {
      plt <- plotly::layout(
        plt,
        hovermode = "closest",
        annotations = list(
          list(
            text = sprintf(
              paste0(
                "<b>OLS line + 95% CI</b><br>",
                "Pearson <i>r</i> (year, Δ) = %.3f<br>",
                "two-sided <i>p</i> = %s"
              ),
              unname(ct$estimate),
              fmt_p_num(ct$p.value)
            ),
            xref = "paper",
            yref = "paper",
            x = 0.02,
            y = 0.98,
            xanchor = "left",
            yanchor = "top",
            showarrow = FALSE,
            align = "left",
            font = list(size = 12, color = "#1a365d"),
            bgcolor = "rgba(255,255,255,0.92)",
            bordercolor = "#cbd5e1",
            borderwidth = 1,
            borderpad = 5
          )
        )
      )
    } else {
      plt <- plotly::layout(plt, hovermode = "closest")
    }
    plt
  })

  output$showpiece_exemplars <- renderUI({
    if (is.null(showpiece_state$mean_cos_a)) {
      return(NULL)
    }
    pa <- showpiece_state$phrase_a
    pb <- showpiece_state$phrase_b
    d <- tryCatch(showpiece_talks_d(), error = function(e) NULL)
    if (is.null(d) || nrow(d) < 3L) {
      return(
        card(
          card_header("Showpiece — exemplar talks"),
          card_body(tags$p(class = "text-muted small mb-0", "Need more talks in view for exemplars."))
        )
      )
    }
    toward_a <- d |>
      arrange(desc(.data$pole_contrast), .data$year) |>
      slice_head(n = 3L)
    toward_b <- d |>
      arrange(.data$pole_contrast, .data$year) |>
      slice_head(n = 3L)
    col_a <- "#742d2d"
    col_b <- "#276749"
    card(
      card_header("Showpiece — exemplar talks (strongest tilts in your filters)"),
      card_body(
        tags$p(
          class = "small text-muted",
          "Δ = mean cosine to ",
          tags$strong(style = sprintf("color:%s;", col_a), "A"),
          " minus mean cosine to ",
          tags$strong(style = sprintf("color:%s;", col_b), "B"),
          ". Higher ⇒ diction/chunk semantics lean toward **A** in this embedding geometry."
        ),
        layout_columns(
          col_widths = c(12, 12),
          tags$div(
            tags$h6(class = "mt-2 mb-2", style = sprintf("color:%s;font-weight:600;", col_a), "Toward phrase A"),
            tags$p(class = "small text-muted", tags$code(pa)),
            lapply(seq_len(nrow(toward_a)), function(i) {
              r <- toward_a[i, , drop = FALSE]
              showpiece_excerpt_card(r$talk_id, r$year, r$pole_contrast, col_a)
            })
          ),
          tags$div(
            tags$h6(class = "mt-2 mb-2", style = sprintf("color:%s;font-weight:600;", col_b), "Toward phrase B"),
            tags$p(class = "small text-muted", tags$code(pb)),
            lapply(seq_len(nrow(toward_b)), function(i) {
              r <- toward_b[i, , drop = FALSE]
              showpiece_excerpt_card(r$talk_id, r$year, r$pole_contrast, col_b)
            })
          )
        )
      )
    )
  })

  output$tw_showpiece_embed_err <- renderText({
    e <- showpiece_state$err
    if (is.null(e) || !nzchar(e)) {
      return("")
    }
    e
  })

  output$tw_show_plot_hdr <- renderUI({
    pa <- showpiece_state$phrase_a
    pb <- showpiece_state$phrase_b
    if (is.null(pa) || is.null(pb) || !nzchar(pa) || !nzchar(pb)) {
      return(tags$h6(class = "text-muted mb-0", "Embed two phrases above (this tab’s year sliders apply)."))
    }
    y1 <- input$tw_yr[[1L]]
    y2 <- input$tw_yr[[2L]]
    tags$div(
      class = "mb-0 small",
      tags$strong("Δ mean cosine (A − B) vs year"),
      " · talks in ",
      tags$strong(paste0(y1, "–", y2)),
      " · ",
      tags$span(style = "color:#742d2d;", tags$strong("A:"), " ", pa),
      " · ",
      tags$span(style = "color:#276749;", tags$strong("B:"), " ", pb)
    )
  })

  output$tw_show_stats_card <- renderUI({
    if (is.null(showpiece_state$mean_cos_a)) {
      return(NULL)
    }
    req(!is.null(input$tw_split_yr))
    d <- tryCatch(showpiece_talks_tw(), error = function(e) NULL)
    if (is.null(d) || nrow(d) < 12L) {
      return(
        card(
          card_header("Contrast tab — tests"),
          card_body(
            markdown(
              paste0(
                "Need at least **12** talks in the **year window** above. Currently **",
                if (is.null(d)) 0L else nrow(d), "** talks."
              )
            )
          )
        )
      )
    }
    spl <- as.integer(input$tw_split_yr)
    spl <- max(min(spl, max(d$year)), min(d$year))
    ct <- tryCatch(
      stats::cor.test(d$year, d$pole_contrast, method = "pearson"),
      error = function(e) NULL
    )
    r_line <- if (is.null(ct)) {
      "*Could not compute Pearson correlation.*"
    } else {
      paste0(
        "**Pearson** *(year, A − B)* on this window: *r* = ", sprintf("%.3f", unname(ct$estimate)),
        ", two-sided *p* ", fmt_p_num(ct$p.value), "."
      )
    }
    mg <- tryCatch(
      mgcv::gam(pole_contrast ~ s(year, k = 10), data = d, method = "REML"),
      error = function(e) NULL,
      warning = function(w) NULL
    )
    gam_line <- if (is.null(mg)) {
      "\n\n**GAM:** could not fit."
    } else {
      p_g <- summary(mg)$s.table[1L, "p-value"]
      edf <- summary(mg)$s.table[1L, "edf"]
      paste0(
        "\n\n**GAM smooth:** edf ≈ ", sprintf("%.2f", edf),
        ", approximate *p* ", fmt_p_num(p_g), "."
      )
    }
    el_md <- early_late_significance_md(d, "pole_contrast", spl)
    el_block <- if (nzchar(el_md)) {
      paste0(
        "\n\n**Early vs late split at year **", spl,
        "** (your slider; Welch *t* + chunk-weighted OLS on A − B):**\n\n",
        el_md
      )
    } else {
      ""
    }
    card(
      card_header("Contrast tab — correlation & mean shift"),
      card_body(
        markdown(paste0(
          "**Talks in window:** ", nrow(d), ". All tests use **only** this window.\n\n",
          r_line, gam_line, el_block,
          "\n\n*Chunk-weighted OLS* treats each talk as repeated **n_chunks** times at the talk’s score (exploratory)."
        ))
      )
    )
  })

  output$plt_tw_showpiece <- renderPlotly({
    d <- showpiece_talks_tw()
    req(nrow(d) > 0L)
    ct <- tryCatch(
      stats::cor.test(d$year, d$pole_contrast, method = "pearson"),
      error = function(e) NULL
    )
    lmdf <- lm_line_ribbon_df(d, "year", "pole_contrast")
    g <- ggplot(d, aes(year, pole_contrast, text = talk_id, size = n_chunks)) +
      geom_hline(yintercept = 0, linetype = 3, linewidth = 0.35, color = "gray55") +
      geom_point(alpha = 0.28, color = "#553c9a")
    if (!is.null(lmdf)) {
      g <- g +
        geom_ribbon(
          data = lmdf,
          aes(x = x, ymin = ymin, ymax = ymax),
          inherit.aes = FALSE,
          alpha = 0.22,
          fill = "#9b2c2c"
        ) +
        geom_line(
          data = lmdf,
          aes(x = x, y = fit),
          inherit.aes = FALSE,
          color = "#9b2c2c",
          linewidth = 0.95
        )
    }
    g <- g +
      scale_size(range = c(1, 6), guide = "none") +
      labs(
        x = "Conference year",
        y = "Δ mean cosine (phrase A − phrase B)"
      ) +
      theme_minimal(base_size = 13)
    plt <- plotly::ggplotly(g, tooltip = "text")
    if (!is.null(ct)) {
      plt <- plotly::layout(
        plt,
        hovermode = "closest",
        annotations = list(
          list(
            text = sprintf(
              paste0(
                "<b>OLS line + 95% CI</b><br>",
                "Pearson <i>r</i> (year, Δ) = %.3f<br>",
                "two-sided <i>p</i> = %s"
              ),
              unname(ct$estimate),
              fmt_p_num(ct$p.value)
            ),
            xref = "paper",
            yref = "paper",
            x = 0.02,
            y = 0.98,
            xanchor = "left",
            yanchor = "top",
            showarrow = FALSE,
            align = "left",
            font = list(size = 12, color = "#1a365d"),
            bgcolor = "rgba(255,255,255,0.92)",
            bordercolor = "#cbd5e1",
            borderwidth = 1,
            borderpad = 5
          )
        )
      )
    } else {
      plt <- plotly::layout(plt, hovermode = "closest")
    }
    plt
  })

  output$tw_showpiece_exemplars <- renderUI({
    if (is.null(showpiece_state$mean_cos_a)) {
      return(NULL)
    }
    pa <- showpiece_state$phrase_a
    pb <- showpiece_state$phrase_b
    d <- tryCatch(showpiece_talks_tw(), error = function(e) NULL)
    if (is.null(d) || nrow(d) < 3L) {
      return(
        card(
          card_header("Contrast tab — exemplar talks"),
          card_body(tags$p(class = "text-muted small mb-0", "Widen the year window or embed again."))
        )
      )
    }
    toward_a <- d |>
      arrange(desc(.data$pole_contrast), .data$year) |>
      slice_head(n = 3L)
    toward_b <- d |>
      arrange(.data$pole_contrast, .data$year) |>
      slice_head(n = 3L)
    col_a <- "#742d2d"
    col_b <- "#276749"
    card(
      card_header("Contrast tab — exemplar talks"),
      card_body(
        tags$p(
          class = "small text-muted",
          "Extremes within **this tab’s** year window. Quotes are inspector **swing** chunks (not phrase-picked)."
        ),
        layout_columns(
          col_widths = c(12, 12),
          tags$div(
            tags$h6(class = "mt-2 mb-2", style = sprintf("color:%s;font-weight:600;", col_a), "Toward phrase A"),
            tags$p(class = "small text-muted", tags$code(pa)),
            lapply(seq_len(nrow(toward_a)), function(i) {
              r <- toward_a[i, , drop = FALSE]
              showpiece_excerpt_card(r$talk_id, r$year, r$pole_contrast, col_a)
            })
          ),
          tags$div(
            tags$h6(class = "mt-2 mb-2", style = sprintf("color:%s;font-weight:600;", col_b), "Toward phrase B"),
            tags$p(class = "small text-muted", tags$code(pb)),
            lapply(seq_len(nrow(toward_b)), function(i) {
              r <- toward_b[i, , drop = FALSE]
              showpiece_excerpt_card(r$talk_id, r$year, r$pole_contrast, col_b)
            })
          )
        )
      )
    )
  })

  filtered <- reactive({
    req(length(input$era_f) > 0)
    d <- talk_scores |>
      filter(
        .data$year >= input$yr[1],
        .data$year <= input$yr[2],
        .data$era %in% input$era_f
      )
    req(nrow(d) > 0)
    d
  })

  output$plt_scatter <- renderPlotly({
    d <- filtered()
    gamdf <- gam_line_ribbon_df(d, "year", "mean_net_presc", k = 10L)
    lmdf <- lm_line_ribbon_df(d, "year", "mean_net_presc")
    g <- ggplot(d, aes(year, mean_net_presc, text = talk_id, size = n_chunks)) +
      geom_hline(yintercept = 0, linetype = 2, color = "gray50") +
      geom_point(alpha = 0.25, color = "#1a365d")
    if (!is.null(gamdf)) {
      g <- g +
        geom_ribbon(
          data = gamdf,
          aes(x = x, ymin = ymin, ymax = ymax),
          inherit.aes = FALSE,
          alpha = 0.16,
          fill = "#276749"
        ) +
        geom_line(
          data = gamdf,
          aes(x = x, y = fit),
          inherit.aes = FALSE,
          color = "#276749",
          linewidth = 0.85
        )
    }
    if (!is.null(lmdf)) {
      g <- g +
        geom_ribbon(
          data = lmdf,
          aes(x = x, ymin = ymin, ymax = ymax),
          inherit.aes = FALSE,
          alpha = 0.18,
          fill = "#c53030"
        ) +
        geom_line(
          data = lmdf,
          aes(x = x, y = fit),
          inherit.aes = FALSE,
          color = "#9b2c2c",
          linewidth = 0.95
        )
    }
    g <- g +
      scale_size(range = c(1, 6), guide = "none") +
      labs(x = "Conference year", y = "Mean net score (chunks)") +
      theme_minimal(base_size = 13)
    ggplotly(g, tooltip = "text") |> layout(hovermode = "closest")
  })

  output$tbl <- renderDT({
    filtered() |>
      arrange(desc(year)) |>
      transmute(
        year = year,
        talk_id = substr(as.character(talk_id), 1, 12),
        n_chunks = n_chunks,
        net = round(mean_net_presc, 4),
        cos_p = round(mean_cos_presc, 3),
        cos_g = round(mean_cos_gentle, 3)
      ) |>
      head(500) |>
      datatable(rownames = FALSE, options = list(scrollX = TRUE, pageLength = 12))
  })

  observe({
    if (!isTRUE(has_chunk_highlights)) {
      return(invisible(NULL))
    }
    req(input$chunk_year)
    ids <- talk_scores |>
      filter(.data$year == as.integer(input$chunk_year)) |>
      pull(.data$talk_id) |>
      unique() |>
      as.character() |>
      sort()
    if (!length(ids)) {
      return(invisible(NULL))
    }
    cur <- input$chunk_talk
    sel <- if (!is.null(cur) && nzchar(cur) && cur %in% ids) cur else ids[[1L]]
    updateSelectInput(session, "chunk_talk", choices = ids, selected = sel)
  })

  output$chunk_talk_summary <- renderUI({
    if (!isTRUE(has_chunk_highlights)) {
      return(invisible(NULL))
    }
    req(input$chunk_talk)
    ts <- talk_scores |>
      filter(as.character(.data$talk_id) == as.character(input$chunk_talk))
    req(nrow(ts) == 1L)
    card(
      card_header("Talk-level context"),
      card_body(
        markdown(paste0(
          "**Conference year:** ", ts$year[[1L]],
          " · **Number of semantic chunks in this talk:** ", ts$n_chunks[[1L]],
          " · **Talk net score** (simple average of chunk nets): **",
          sprintf("%.4f", ts$mean_net_presc[[1L]]), "**\n\n",
          "**Mean** chunk similarity to prescriptive pole: ", sprintf("%.3f", ts$mean_cos_presc[[1L]]),
          " · to invitational pole: ", sprintf("%.3f", ts$mean_cos_gentle[[1L]]), ".\n\n",
          "*Below, each quoted passage is one chunk (~220 tokens). Numbers are relative to the whole corpus, ",
          "not to General Conference as a moral category.*"
        ))
      )
    )
  })

  output$chunk_cards <- renderUI({
    if (!isTRUE(has_chunk_highlights)) {
      return(invisible(NULL))
    }
    req(input$chunk_talk)
    rows <- chunk_highlights |>
      filter(as.character(.data$talk_id) == as.character(input$chunk_talk))
    req(nrow(rows) > 0L)
    kinds <- c("prescriptive", "invitational", "swing")
    pieces <- lapply(kinds, function(k) {
      sub <- rows |> filter(.data$kind == k)
      if (nrow(sub) == 0L) {
        return(NULL)
      }
      if (k == "invitational") {
        sub <- sub |> arrange(.data$net_presc)
      } else if (k == "swing") {
        sub <- sub |> arrange(desc(abs(.data$vs_talk_mean)))
      } else {
        sub <- sub |> arrange(desc(.data$net_presc))
      }
      tagList(
        tags$h5(class = "mt-4 mb-3", style = "color:#2c5282;font-weight:600;", unique(as.character(sub$kind_title))[1]),
        lapply(seq_len(nrow(sub)), function(i) chunk_card_ui(sub[i, , drop = FALSE]))
      )
    })
    tagList(Filter(Negate(is.null), pieces))
  })
}

shinyApp(ui, server)
