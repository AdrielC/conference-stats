# ---- General Conference — family-friendly Shiny explorer ----
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
      "This app is a personal research demo for friends and family. ",
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
            "**Custom pole** to type your own phrase and see how talks align to it over time (needs Python + synced embedding data), ",
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
              "You type a **short phrase** (a word, quotation, or sentence). The app calls **Python** to embed it with the ",
              "same **BGE + tf–idf pooling** and **subword IDF** as the main corpus pipeline, then measures **mean cosine similarity** ",
              "between that direction and each talk’s semantic chunks (using precomputed **per-talk sums** of chunk vectors — ",
              "exactly consistent with averaging cosines because chunk vectors are length-normalized).\n\n",
              "**Not keyword counts** — semantic similarity. **First click** may take a minute while PyTorch loads the model. ",
              "You need a working Python env with `pip install -r analysis/python/requirements-gc-embed.txt`.\n\n",
              "**Year / era filters** match the **Explore** tab sliders and checkboxes."
            ))
          )
        ),
        card(
          card_header("Your phrase → trend over time"),
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
                label = "Embed & update chart",
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
        card(
          class = "explore-plot-card",
          card_header(uiOutput("custom_plot_title")),
          plotlyOutput("plt_custom_pole", height = "68vh")
        )
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
          "8. **Custom pole (Shiny tab):** Precomputed **sums of chunk embedding vectors** per talk plus frozen **subword IDF** let you embed any short phrase with Python and plot **mean cosine** vs year (same pooling as the main pipeline).\n\n",
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
          "- **Family demo** — if an uncle says “but Elder so-and-so in ’82…”, that’s qualitative texture our average line will never capture."
        ))
      )
    )
  ),

  nav_panel(
    tags$span(icon("heart"), " About"),
    card(
      card_header("For family & friends"),
      card_body(
        markdown(paste0(
          "This app bundles outputs from a research side project on **language change** in General Conference. ",
          "If slides or Thanksgiving conversation ever need a chart — you’ve got it.\n\n",
          "**Repo:** `conference-stats` on GitHub (same project as the Parquet corpus helpers).\n\n",
          "Made with R **Shiny**, **ggplot2**, **plotly**, and a lot of punctuation typed next to embeddings."
        ))
      )
    )
  )
  )
)

server <- function(input, output, session) {
  custom_state <- reactiveValues(mean_cos = NULL, phrase = NULL, err = "")

  observeEvent(input$run_custom_embed, {
    if (!isTRUE(has_custom_pole_bundle) || !isTRUE(has_custom_dim_ok) || !isTRUE(has_py_embed_script)) {
      return(invisible(NULL))
    }
    phrase <- trimws(input$custom_phrase)
    if (!nzchar(phrase)) {
      showNotification("Enter text to embed.", type = "warning")
      return(invisible(NULL))
    }
    if (nchar(phrase) > 2000L) {
      showNotification("Please keep the phrase under 2000 characters.", type = "warning")
      return(invisible(NULL))
    }
    custom_state$err <- ""
    py <- Sys.getenv("CONFERENCESTATS_PYTHON", unset = "")
    if (!nzchar(py)) {
      py <- Sys.which("python3")
    }
    if (!nzchar(py)) {
      py <- Sys.which("python")
    }
    if (!nzchar(py)) {
      custom_state$err <- "No python3 or python on PATH. Set CONFERENCESTATS_PYTHON to your interpreter."
      showNotification(custom_state$err, type = "error")
      return(invisible(NULL))
    }
    idf_abs <- normalizePath(idf_bundle_path, winslash = "/", mustWork = TRUE)
    scr_abs <- normalizePath(py_embed_script, winslash = "/", mustWork = TRUE)
    model <- as.character(pipe_meta$model)
    args <- c(scr_abs, "--model", model, "--idf", idf_abs, "--phrase", phrase)
    errf <- tempfile(fileext = ".log")
    on.exit(unlink(errf), add = TRUE)
    res <- withProgress(
      message = "Embedding phrase",
      detail = "First run loads the model into memory (~1 min).",
      value = 0.35,
      expr = {
        system2(py, args = args, stdout = TRUE, stderr = errf)
      }
    )
    st <- attr(res, "status")
    errtxt <- paste(readLines(errf, warn = FALSE), collapse = "\n")
    if (!is.null(st) && !is.na(st) && st != 0L) {
      custom_state$err <- paste0(
        "Python exited with status ", st, ".\n\nSTDERR:\n", errtxt,
        "\n\nSTDOUT:\n", paste(res, collapse = "\n")
      )
      showNotification("Embedding failed — see message below the button.", type = "error")
      return(invisible(NULL))
    }
    u <- tryCatch(
      jsonlite::fromJSON(paste(res, collapse = "")),
      error = function(e) NULL
    )
    if (is.null(u) || length(u) != 384L) {
      custom_state$err <- paste(
        "Could not parse 384-d JSON from Python.\n",
        "STDERR:\n", errtxt, "\nSTDOUT:\n", paste(res, collapse = "\n"),
        sep = ""
      )
      showNotification("Bad embedding output.", type = "error")
      return(invisible(NULL))
    }
    S <- as.matrix(talk_emb_sums_tbl[, s_cols_custom])
    mc <- as.numeric(S %*% matrix(u, ncol = 1L)) / talk_emb_sums_tbl$n_chunks
    custom_state$mean_cos <- mc
    custom_state$phrase <- phrase
  })

  output$custom_embed_err <- renderText({
    e <- custom_state$err
    if (is.null(e) || !nzchar(e)) {
      return("")
    }
    e
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

  output$plt_custom_pole <- renderPlotly({
    req(custom_state$mean_cos)
    req(length(input$era_f) > 0L)
    te <- talk_emb_sums_tbl
    te$mean_cos_custom <- custom_state$mean_cos
    d <- talk_scores |>
      inner_join(
        te |> select("talk_id", "year", "mean_cos_custom"),
        by = c("talk_id", "year")
      ) |>
      filter(
        .data$year >= input$yr[[1L]],
        .data$year <= input$yr[[2L]],
        .data$era %in% input$era_f
      )
    req(nrow(d) > 0L)
    g <- ggplot(d, aes(year, mean_cos_custom, text = talk_id, size = n_chunks)) +
      geom_hline(yintercept = 0, linetype = 2, color = "gray50") +
      geom_point(alpha = 0.25, color = "#276749") +
      geom_smooth(method = "gam", formula = y ~ s(x, k = 10), color = "#9b2c2c", fill = "#9b2c2c33", linewidth = 0.7) +
      scale_size(range = c(1, 6), guide = "none") +
      labs(x = "Conference year", y = "Mean cosine (your phrase ↔ chunks)") +
      theme_minimal(base_size = 13)
    ggplotly(g, tooltip = "text") |> layout(hovermode = "closest")
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
    g <- ggplot(d, aes(year, mean_net_presc, text = talk_id, size = n_chunks)) +
      geom_hline(yintercept = 0, linetype = 2, color = "gray50") +
      geom_point(alpha = 0.25, color = "#1a365d") +
      geom_smooth(method = "gam", formula = y ~ s(x, k = 10), color = "#c53030", fill = "#c5303044", linewidth = 0.7) +
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
