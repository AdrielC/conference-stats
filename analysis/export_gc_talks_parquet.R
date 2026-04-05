#!/usr/bin/env Rscript
## Export normalized genconf talks to Parquet for the Python chunk+embedding pipeline.
## Usage (from package root):
##   Rscript analysis/export_gc_talks_parquet.R
##   Rscript analysis/export_gc_talks_parquet.R path/to/out.parquet

repo_root <- if (nzchar(Sys.getenv("CONFERENCESTATS_ROOT", ""))) {
  Sys.getenv("CONFERENCESTATS_ROOT")
} else {
  normalizePath(".", mustWork = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
default_out <- file.path(repo_root, "analysis", "data", "gc_talks_normalized.parquet")
out_path <- if (length(args) >= 1L) normalizePath(args[[1]], mustWork = FALSE) else default_out

if (!dir.exists(file.path(repo_root, "R"))) {
  stop("Run from conferencestats package root (or set CONFERENCESTATS_ROOT).")
}

if (!requireNamespace("generalconference", quietly = TRUE)) {
  stop("Install remotes::install_github('bryanwhiting/generalconference@v0.3.2')")
}
if (!requireNamespace("nanoparquet", quietly = TRUE)) stop("Install nanoparquet package.")
if (!requireNamespace("pkgload", quietly = TRUE)) stop("Install pkgload.")

suppressPackageStartupMessages({
  pkgload::load_all(repo_root, quiet = TRUE)
  library(dplyr)
})
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)

talks <- gc_normalize_talks(gc_as_tibble_generalconference())
tbl <- talks |>
  select(talk_id, year, month, speaker, title, text)

nanoparquet::write_parquet(tbl, out_path)
message("Wrote ", nrow(tbl), " talks to ", out_path)
