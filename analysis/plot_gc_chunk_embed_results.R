#!/usr/bin/env Rscript
## Publication-style figures from gc_chunk_embed_pipeline outputs (BGE + semantic chunks).
## Usage:
##   Rscript analysis/plot_gc_chunk_embed_results.R [path/to/talk_scores.parquet]

argv_full <- commandArgs(trailingOnly = FALSE)
arg_file <- sub("^--file=", "", argv_full[grep("^--file=", argv_full)])[1]
script_dir <- if (is.na(arg_file) || !nzchar(arg_file)) {
  getwd()
} else {
  dirname(normalizePath(arg_file))
}
repo_root <- if (nzchar(Sys.getenv("CONFERENCESTATS_ROOT", ""))) {
  Sys.getenv("CONFERENCESTATS_ROOT")
} else {
  normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
default_parquet <- file.path(
  repo_root, "analysis", "data", "gc_chunk_embed", "talk_scores.parquet"
)
parquet_path <- if (length(args) >= 1L) normalizePath(args[[1]], mustWork = TRUE) else default_parquet

out_dir <- file.path(repo_root, "analysis", "output", "gc_chunk_embed")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(parquet_path)) {
  stop(
    "Missing ", parquet_path, "\nRun: analysis/python/gc_chunk_embed_pipeline.py ...",
    call. = FALSE
  )
}

needed <- c(
  "ggplot2", "dplyr", "scales", "mgcv", "tidyr", "stringr",
  "patchwork", "ggtext", "nanoparquet", "jsonlite"
)
miss <- needed[!vapply(needed, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(miss)) {
  stop("Install: install.packages(c(", paste0("\"", miss, "\"", collapse = ", "), "))", call. = FALSE)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(mgcv)
  library(tidyr)
  library(patchwork)
  library(ggtext)
  library(jsonlite)
})

raw <- nanoparquet::read_parquet(parquet_path)
d <- raw |>
  filter(!is.na(.data$year), !is.na(.data$mean_net_presc)) |>
  mutate(
    year = as.integer(.data$year),
    decade = floor(.data$year / 10) * 10
  )

stopifnot(nrow(d) > 50L)
n_talks_ok <- nrow(d)

## GAM on full talk-level series
m <- mgcv::gam(mean_net_presc ~ s(year, k = 14), data = d, method = "REML")
p_smooth <- summary(m)$s.table[1, 4]
edf <- sum(m$edf[-1])

grid <- tibble(year = seq(min(d$year), max(d$year), length.out = 400))
grid$fit <- as.numeric(predict(m, newdata = grid))
grid$se <- as.numeric(predict(m, newdata = grid, se.fit = TRUE)$se.fit)

q10 <- quantile(d$year, 0.1)
q90 <- quantile(d$year, 0.9)
early_fit <- mean(grid$fit[grid$year <= q10])
late_fit <- mean(grid$fit[grid$year >= q90])
delta <- late_fit - early_fit
r_pearson <- cor(d$year, d$mean_net_presc)

## Decadal aggregates
dec <- d |>
  group_by(.data$decade) |>
  summarise(
    n = n(),
    m = mean(.data$mean_net_presc),
    se = stats::sd(.data$mean_net_presc) / sqrt(n()),
    .groups = "drop"
  )

## --- Plots (cohesive palette: deep navy, burnt sienna, slate) ---
col_line <- "#9b2c2c"
col_rib <- "#9b2c2c33"
col_pt <- "#1a365d"
col_dec <- "#2c5282"
col_fit <- "#742d2d"
col_gent <- "#276749"
col_presc <- "#2b4c7e"

## 1) Main trajectory
p_main <- ggplot(d, aes(year, mean_net_presc)) +
  geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", color = "gray55") +
  geom_point(aes(size = n_chunks), alpha = 0.08, color = col_pt, stroke = 0) +
  geom_ribbon(
    data = grid,
    aes(x = year, ymin = fit - 1.96 * se, ymax = fit + 1.96 * se),
    alpha = 0.2,
    fill = col_line,
    inherit.aes = FALSE
  ) +
  geom_line(data = grid, aes(x = year, y = fit), linewidth = 0.95, color = col_fit, inherit.aes = FALSE) +
  scale_size(range = c(0.2, 4), guide = "none") +
  labs(
    title = "**Semantic prescriptiveness** of General Conference talks (chunked corpus)",
    subtitle = "Each point: one talk &mdash; mean over semantic chunks of (cos U<sub>presc</sub> &minus; cos U<sub>gentle</sub>) | BGE-small, tf&ndash;idf pooling",
    x = NULL,
    y = "Mean net cosine (prescriptive &minus; invitational pole)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_markdown(face = "bold", size = rel(1.08)),
    plot.subtitle = element_markdown(color = "gray35", size = rel(0.92)),
    panel.grid.minor = element_blank(),
    plot.margin = margin(12, 12, 8, 12)
  )

ggsave(
  file.path(out_dir, "01_talk_trajectory_gam.png"),
  p_main,
  width = 11,
  height = 5.8,
  dpi = 160,
  bg = "white"
)

## 2) Decades ribbon + lollipop
p_dec <- ggplot(dec, aes(factor(decade), m)) +
  geom_hline(yintercept = 0, linetype = 3, color = "gray60") +
  geom_segment(
    aes(x = factor(decade), xend = factor(decade), y = 0, yend = m),
    color = col_dec,
    linewidth = 0.55,
    alpha = 0.85
  ) +
  geom_point(aes(size = n), color = col_line, alpha = 0.95) +
  geom_errorbar(
    aes(ymin = m - se, ymax = m + se),
    width = 0.2,
    color = col_dec,
    linewidth = 0.45,
    alpha = 0.75
  ) +
  scale_size(range = c(4, 14), breaks = pretty_breaks(), name = "Talks") +
  labs(
    title = "**By decade:** invitational pole gains relative to prescriptive pole",
    subtitle = "Error bars &pm;1 SE of talk-level means | regex-mined exemplar poles on semantic chunks",
    x = "Decade",
    y = "Mean net cosine"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_markdown(face = "bold"),
    plot.subtitle = element_markdown(color = "gray35", size = rel(0.9)),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  file.path(out_dir, "02_decadal_lollipop.png"),
  p_dec,
  width = 9,
  height = 6,
  dpi = 160,
  bg = "white"
)

## 3) 2D embedding plane: prescriptive vs gentle cosine, colored by year
p_plane <- ggplot(d, aes(mean_cos_presc, mean_cos_gentle, color = year)) +
  geom_segment(
    x = 0.5, y = 0.5, xend = 0.85, yend = 0.85,
    linewidth = 0.25,
    linetype = "dotted",
    color = "gray75",
    inherit.aes = FALSE
  ) +
  geom_point(aes(size = n_chunks), alpha = 0.18) +
  scale_color_viridis_c(option = "mako", begin = 0.15, end = 0.95, name = "Year") +
  scale_size(range = c(0.25, 3.2), guide = "none") +
  coord_fixed(ratio = 1, xlim = range(d$mean_cos_presc), ylim = range(d$mean_cos_gentle), expand = TRUE) +
  labs(
    title = "Alignment to **two poles** in embedding space",
    subtitle = "Above diagonal &mdash; chunk-mean cos<sub>gentle</sub> &gt; cos<sub>presc</sub> (more invitational on this scale)",
    x = "Mean cos similarity to prescriptive exemplars",
    y = "Mean cos similarity to invitational exemplars"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_markdown(face = "bold"),
    plot.subtitle = element_markdown(color = "gray35", size = rel(0.88)),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(out_dir, "03_cosine_plane_by_year.png"),
  p_plane,
  width = 8,
  height = 6.8,
  dpi = 160,
  bg = "white"
)

## 4) Ridgeline-style density by era (approximate with violin by 15y bins)
d_viol <- d |> mutate(
  era = cut(
    .data$year,
    breaks = c(1970, 1985, 2000, 2015, 2022),
    labels = c("1971&ndash;1985", "1986&ndash;2000", "2001&ndash;2015", "2016&ndash;2021")
  )
)
d_viol <- d_viol |> filter(!is.na(.data$era))

p_violin <- ggplot(d_viol, aes(factor(era), mean_net_presc, fill = factor(era))) +
  geom_hline(yintercept = 0, color = "gray55", linetype = 2) +
  geom_violin(alpha = 0.88, color = NA, scale = "width", width = 0.95) +
  stat_summary(fun = median, geom = "point", color = "white", size = 1.6, shape = 18) +
  scale_fill_manual(
    values = c("#5c7cba", "#c9a227", "#c05621", "#742d2d"),
    guide = "none"
  ) +
  labs(
    title = "Distribution of talk scores **moves downward** in later eras",
    subtitle = "Net cosine across chunks; wider spread in recent decades",
    x = NULL,
    y = "Mean net (presc &minus; gentle)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_markdown(face = "bold"),
    plot.subtitle = element_markdown(color = "gray35", size = rel(0.9)),
    axis.text.x = element_markdown(),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(out_dir, "04_era_violins.png"),
  p_violin,
  width = 9.5,
  height = 5.5,
  dpi = 160,
  bg = "white"
)

## 5) Composite panel
p_combo <- p_main + p_dec + plot_layout(ncol = 1, heights = c(1.12, 1))
ggsave(
  file.path(out_dir, "00_panel_trajectory_and_decades.png"),
  p_combo,
  width = 10.5,
  height = 11,
  dpi = 150,
  bg = "white"
)

## Summary JSON for downstream
summ <- list(
  n_talks = n_talks_ok,
  year_min = min(d$year),
  year_max = max(d$year),
  gam_edf = unname(edf),
  gam_p_smooth = max(as.numeric(p_smooth), 2.2e-16),
  cor_year_net = unname(r_pearson),
  gam_fit_early_decile = unname(early_fit),
  gam_fit_late_decile = unname(late_fit),
  delta_late_minus_early = unname(delta)
)
jsonlite::write_json(
  summ,
  file.path(out_dir, "summary_stats.json"),
  pretty = TRUE,
  auto_unbox = TRUE
)

shiny_root <- file.path(repo_root, "analysis", "shiny_gc_family")
shiny_www <- file.path(shiny_root, "www")
shiny_data <- file.path(shiny_root, "data")

## Chunk-level highlights for Shiny “meaningful segments” inspector
ch_path <- file.path(dirname(normalizePath(parquet_path)), "chunks_scored.parquet")
if (file.exists(ch_path) && requireNamespace("stringr", quietly = TRUE)) {
  ch <- nanoparquet::read_parquet(ch_path)
  ch <- ch |>
    mutate(
      talk_id = as.character(.data$talk_id),
      year = as.integer(.data$year)
    )
  th <- ch |>
    group_by(.data$talk_id) |>
    summarise(
      talk_mean_net = mean(.data$net_presc, na.rm = TRUE),
      n_chunks_talk = dplyr::n(),
      .groups = "drop"
    )
  ch2 <- ch |>
    dplyr::left_join(th, by = "talk_id") |>
    mutate(
      text_clean = stringr::str_squish(as.character(.data$text)),
      ## Word count for filtering ritual closings / ultra-short segments
      n_words = lengths(
        regmatches(.data$text_clean, gregexpr("\\S+", .data$text_clean, perl = TRUE))
      ),
      ends_with_amen = stringr::str_detect(
        .data$text_clean,
        stringr::regex("\\bamen\\.?\\s*$", ignore_case = TRUE)
      ),
      ## Drop short closings and tiny fragments; base rule before final-segment rule below
      highlight_ok = .data$n_words >= 55L &
        !(.data$n_words < 70L & .data$ends_with_amen)
    ) |>
    dplyr::group_by(.data$talk_id) |>
    dplyr::mutate(
      ## Final segment is almost always formulaic (“in the name of…”, “amen.”);
      ## exclude from the *preferred* pool. pick_within_talk falls back to all
      ## chunks when too few are left (very short talks).
      is_last_in_talk = .data$chunk_idx == max(.data$chunk_idx, na.rm = TRUE)
    ) |>
    dplyr::ungroup() |>
    dplyr::mutate(highlight_ok = .data$highlight_ok & !.data$is_last_in_talk)

  pick_within_talk <- function(rows, slice_fn, n_pick) {
    dplyr::group_by(rows, .data$talk_id) |>
      dplyr::group_modify(function(d, key) {
        d_use <- if (sum(d$highlight_ok, na.rm = TRUE) >= n_pick) {
          dplyr::filter(d, .data$highlight_ok)
        } else {
          d
        }
        slice_fn(d_use, n_pick)
      }) |>
      dplyr::ungroup()
  }

  n_ext <- 3L
  n_swing <- 2L

  hi_presc <- ch2 |>
    pick_within_talk(function(d, n_pick) {
      dplyr::slice_max(d, order_by = .data$net_presc, n = n_pick, with_ties = FALSE)
    }, n_ext) |>
    mutate(kind = "prescriptive")

  hi_inv <- ch2 |>
    pick_within_talk(function(d, n_pick) {
      dplyr::slice_min(d, order_by = .data$net_presc, n = n_pick, with_ties = FALSE)
    }, n_ext) |>
    mutate(kind = "invitational")

  excl <- dplyr::bind_rows(
    hi_presc |> dplyr::transmute(talk_id = .data$talk_id, chunk_idx = .data$chunk_idx),
    hi_inv |> dplyr::transmute(talk_id = .data$talk_id, chunk_idx = .data$chunk_idx)
  ) |>
    dplyr::distinct()

  hi_swing <- ch2 |>
    dplyr::anti_join(excl, by = c("talk_id", "chunk_idx")) |>
    dplyr::group_by(.data$talk_id) |>
    dplyr::group_modify(function(d, key) {
      if (nrow(d) == 0L) {
        return(d[integer(0), , drop = FALSE])
      }
      d_use <- if (sum(d$highlight_ok, na.rm = TRUE) >= n_swing) {
        dplyr::filter(d, .data$highlight_ok)
      } else {
        d
      }
      if (nrow(d_use) < n_swing) {
        d_use <- d
      }
      d_use <- d_use |>
        dplyr::mutate(dev = abs(.data$net_presc - .data$talk_mean_net[[1L]]))
      dplyr::slice_max(d_use, order_by = .data$dev, n = n_swing, with_ties = FALSE) |>
        dplyr::select(-"dev")
    }) |>
    dplyr::ungroup() |>
    mutate(kind = "swing")

  chunk_hi <- dplyr::bind_rows(hi_presc, hi_inv, hi_swing) |>
    mutate(
      vs_talk_mean = .data$net_presc - .data$talk_mean_net,
      kind_title = dplyr::case_when(
        .data$kind == "prescriptive" ~
          "Highest net in this talk (least gentle vs this talk’s other chunks; can still be < 0 overall)",
        .data$kind == "invitational" ~
          "Lowest net in this talk (most invitational-leaning vs this talk’s other chunks)",
        .data$kind == "swing" ~
          "High leverage (vs talk mean; segments not listed above)",
        TRUE ~ as.character(.data$kind)
      ),
      text_excerpt = stringr::str_trunc(.data$text_clean, 650L, ellipsis = "\u2026")
    ) |>
    select(
      "talk_id", "year", "chunk_idx", "kind", "kind_title",
      "net_presc", "cos_presc", "cos_gentle",
      "talk_mean_net", "vs_talk_mean",
      "n_chunks_talk",
      "text_excerpt"
    )

  if (dir.exists(shiny_data)) {
    saveRDS(chunk_hi, file.path(shiny_data, "chunk_highlights.rds"), compress = "xz")
    message(
      sprintf(
        "Chunk inspector data: %d highlight rows → %s",
        nrow(chunk_hi),
        file.path(shiny_data, "chunk_highlights.rds")
      )
    )
  }
} else {
  message(
    "Tip: run Python pipeline to create chunks_scored.parquet next to talk_scores, ",
    "then re-run this script to build chunk_highlights.rds for the Shiny app."
  )
}

## Sync figures + stats into Shiny app folder (for family demo / repo)
if (dir.exists(shiny_www) && dir.exists(shiny_data)) {
  pngs <- list.files(out_dir, pattern = "\\.png$", full.names = TRUE)
  if (length(pngs)) {
    invisible(file.copy(pngs, shiny_www, overwrite = TRUE))
  }
  invisible(file.copy(
    file.path(out_dir, "summary_stats.json"),
    file.path(shiny_data, "summary_stats.json"),
    overwrite = TRUE
  ))
  d_shiny <- raw |>
    filter(!is.na(.data$year), !is.na(.data$mean_net_presc)) |>
    mutate(year = as.integer(.data$year))
  saveRDS(d_shiny, file.path(shiny_data, "talk_scores.rds"), compress = "xz")

  par_dir <- dirname(normalizePath(parquet_path))
  tes_parq <- file.path(par_dir, "talk_emb_sums.parquet")
  idf_npy <- file.path(par_dir, "subword_idf.npy")
  meta_js <- file.path(par_dir, "pipeline_meta.json")
  if (file.exists(tes_parq) && requireNamespace("nanoparquet", quietly = TRUE)) {
    tes_df <- nanoparquet::read_parquet(tes_parq)
    saveRDS(tes_df, file.path(shiny_data, "talk_emb_sums.rds"), compress = "xz")
    message("Synced talk_emb_sums.rds (Custom pole tab).")
  } else {
    message(
      "No talk_emb_sums.parquet next to talk_scores — re-run Python pipeline for Custom pole data."
    )
  }
  if (file.exists(idf_npy)) {
    invisible(file.copy(idf_npy, file.path(shiny_data, "subword_idf.npy"), overwrite = TRUE))
    message("Synced subword_idf.npy")
  }
  if (file.exists(meta_js)) {
    invisible(file.copy(meta_js, file.path(shiny_data, "pipeline_meta.json"), overwrite = TRUE))
    message("Synced pipeline_meta.json")
  }
  message("Synced → analysis/shiny_gc_family/{www,data}/")
}

message("\n=== GC chunk-embed summary ===")
message(sprintf("Talks: %d | years %d–%d", nrow(d), min(d$year), max(d$year)))
message(sprintf(
  "GAM mean_net ~ s(year): edf=%.2f, p≈%.2e",
  edf, p_smooth
))
message(sprintf("r(year, mean_net) = %.3f", r_pearson))
message(sprintf(
  "Smoothed mean (early yr decile vs late): %.4f → %.4f | Δ = %.4f",
  early_fit, late_fit, delta
))
message("\nFigures written to:\n", normalizePath(out_dir))
