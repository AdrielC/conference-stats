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

img <- function(src, alt) {
  tags$figure(
    class = "figure my-3",
    tags$img(
      src = src,
      alt = alt,
      class = "img-fluid rounded shadow border",
      style = "width:100%;max-width:1100px;"
    ),
    tags$figcaption(class = "figure-caption text-center", alt)
  )
}

ui <- page_navbar(
  title = tagList(
    tags$span(style = "color:#1a365d;", "\u2606 "),
    tags$strong("General Conference"),
    tags$span(" — semantic trends explorer", class = "text-muted")
  ),
  theme = bs_theme(bootswatch = "flatly", primary = "#2c5282"),
  fillable = TRUE,
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
            "Use the tabs above: **Gallery** for the big charts, **Explore** to play with sliders and points, ",
            "**Methods** for how it was built."
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
    layout_columns(
      col_widths = c(12),
      card(
        card_header("Full-width story panel"),
        card_body(img("00_panel_trajectory_and_decades.png", "Trajectory and decadal summary combined"))
      )
    ),
    layout_columns(
      col_widths = c(12),
      card(
        card_header("Every talk as a dot — curve is the trend"),
        card_body(img("01_talk_trajectory_gam.png", "GAM smooth with 95% band over talk-level net scores"))
      )
    ),
    layout_columns(
      col_widths = c(12, 12),
      card(
        card_header("By decade"),
        card_body(img("02_decadal_lollipop.png", "Decadal means with error bars"))
      ),
      card(
        card_header("Two language “poles” in embedding space"),
        card_body(img("03_cosine_plane_by_year.png", "Prescriptive vs invitational cosine; color = year"))
      )
    ),
    layout_columns(
      col_widths = c(12),
      card(
        card_header("How distributions shift across eras"),
        card_body(img("04_era_violins.png", "Violin plots by time period"))
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
      card_header("Interactive: year vs net prescriptive score"),
      plotlyOutput("plt_scatter", height = "520px")
    ),
    card(
      card_header("Sample of filtered talks (first 500 rows)"),
      DTOutput("tbl")
    )
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
          "7. **Trend:** `mgcv` GAM smooth on year in R; figures saved under `analysis/output/gc_chunk_embed/`.\n\n",
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

server <- function(input, output, session) {
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
}

shinyApp(ui, server)
