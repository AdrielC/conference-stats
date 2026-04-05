# Corpus: canonical Parquet store (~/.cache or user override)

.corpus_filename <- function() {
  "general_conference_talks.parquet"
}

.corpus_version_string <- function() {
  as.character(utils::packageVersion("conferencestats"))
}

#' Location of the default General Conference Parquet corpus
#'
#' Uses [rappdirs::user_cache_dir()]: typically
#' `~/.cache/conferencestats` on Linux,
#' `~/Library/Caches/conferencestats` on macOS.
#'
#' @param filename File name (default: package-standard Parquet name).
#' @return Character path (directory or file).
#' @export
gc_corpus_cache_dir <- function(filename = NULL) {
  root <- rappdirs::user_cache_dir("conferencestats")
  if (is.null(filename)) {
    return(root)
  }
  file.path(root, filename)
}

#' @rdname gc_corpus_cache_dir
#' @export
gc_corpus_cache_path <- function() {
  gc_corpus_cache_dir(.corpus_filename())
}

#' Default URL for a pre-built Parquet corpus (optional)
#'
#' Set `options(conferencestats.corpus_parquet_url = "https://..." )` or the
#' environment variable `CONFERENCESTATS_CORPUS_URL` so [gc_ensure_corpus()]
#' can download instead of building from source.
#'
#' @return `character(1)` or `NULL`.
#' @export
gc_default_corpus_download_url <- function() {
  env <- Sys.getenv("CONFERENCESTATS_CORPUS_URL", "")
  if (nzchar(env)) {
    return(env)
  }
  getOption("conferencestats.corpus_parquet_url", default = NULL)
}

#' Build the talk corpus and write Parquet
#'
#' Requires the Suggested package `generalconference` (flatten + normalize).
#' Writes a columnar, compressed file suitable for analysis and version control
#' of artifacts (not the R package itself).
#'
#' @param dest Output path (default: [gc_corpus_cache_path()]).
#' @param overwrite Replace an existing file.
#' @param compression Passed to [nanoparquet::write_parquet()] (e.g. `zstd`, `snappy`).
#' @return `dest`, invisibly.
#' @export
gc_build_corpus_parquet <- function(dest = gc_corpus_cache_path(),
                                    overwrite = FALSE,
                                    compression = "zstd") {
  if (!requireNamespace("generalconference", quietly = TRUE)) {
    rlang::abort(
      c(
        "Install {.pkg generalconference} to build the corpus from source.",
        i = "remotes::install_github('bryanwhiting/generalconference')",
        i = "Or set a download URL: {.code options(conferencestats.corpus_parquet_url = \"https://...\")}"
      )
    )
  }
  if (file.exists(dest) && !overwrite) {
    rlang::abort(c(
      "Corpus file already exists.",
      i = dest,
      i = "Use {.code overwrite = TRUE} to replace it."
    ))
  }
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  talks <- gc_normalize_talks(gc_as_tibble_generalconference())
  talks$corpus_package_version <- .corpus_version_string()
  talks$corpus_built_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  nanoparquet::write_parquet(talks, dest, compression = compression)
  invisible(dest)
}

#' Download a pre-built Parquet corpus
#'
#' @param url HTTP(S) URL to a `.parquet` file (e.g. a GitHub Release asset).
#' @param dest Path to write (default: cache path from [gc_corpus_cache_path()]).
#' @param overwrite Replace an existing file.
#' @param quiet Passed to [utils::download.file()].
#' @return `dest`, invisibly.
#' @export
gc_download_corpus_parquet <- function(url,
                                       dest = gc_corpus_cache_path(),
                                       overwrite = FALSE,
                                       quiet = FALSE) {
  if (missing(url) || !nzchar(as.character(url))) {
    rlang::abort("`url` must be a non-empty string.")
  }
  if (file.exists(dest) && !overwrite) {
    rlang::abort(c("File exists.", i = dest, i = "Set {.code overwrite = TRUE}."))
  }
  dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(dest)) {
    unlink(dest)
  }
  status <- utils::download.file(url, dest, mode = "wb", quiet = quiet)
  if (status != 0L) {
    unlink(dest)
    rlang::abort(c("Download failed.", i = url))
  }
  invisible(dest)
}

#' Ensure the Parquet corpus exists on disk
#'
#' Order of attempts when the file is missing or `force = TRUE`:
#' \enumerate{
#'   \item If `download_url` is non-NULL, download.
#'   \item Else build with [gc_build_corpus_parquet()] (needs `generalconference`).
#' }
#'
#' @param path Output path; default is [gc_corpus_cache_path()].
#' @param force If `TRUE`, rebuild or re-download even when `path` exists.
#' @param download_url `NULL`, or a URL; defaults to [gc_default_corpus_download_url()].
#' @param quiet Less console output.
#' @return `path`, invisibly.
#' @export
gc_ensure_corpus <- function(path = gc_corpus_cache_path(),
                             force = FALSE,
                             download_url = gc_default_corpus_download_url(),
                             quiet = FALSE) {
  if (file.exists(path) && !force) {
    return(invisible(path))
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  if (!is.null(download_url) && nzchar(download_url)) {
    if (!quiet) {
      message("Downloading General Conference corpus Parquet...")
    }
    gc_download_corpus_parquet(
      download_url,
      dest = path,
      overwrite = TRUE,
      quiet = quiet
    )
    return(invisible(path))
  }

  if (!quiet) {
    message("Building General Conference corpus Parquet (first run may take a while)...")
  }
  gc_build_corpus_parquet(dest = path, overwrite = TRUE)
  invisible(path)
}

#' Load the General Conference corpus from Parquet
#'
#' Calls [gc_ensure_corpus()] first when `sync = TRUE` so the first load
#' populates the cache (download or build).
#'
#' @param path Path to `.parquet`; default cache path.
#' @param sync Whether to run [gc_ensure_corpus()] before reading.
#' @param force Passed to [gc_ensure_corpus()] as `force`.
#' @param download_url Passed to [gc_ensure_corpus()].
#' @param quiet Passed to [gc_ensure_corpus()] / download.
#' @return A [tibble::tibble()] of normalized talks (same schema as [gc_normalize_talks()]).
#' @export
gc_load_corpus <- function(path = gc_corpus_cache_path(),
                             sync = TRUE,
                             force = FALSE,
                             download_url = gc_default_corpus_download_url(),
                             quiet = FALSE) {
  if (sync) {
    gc_ensure_corpus(
      path = path,
      force = force,
      download_url = download_url,
      quiet = quiet
    )
  } else if (!file.exists(path)) {
    rlang::abort(c("No corpus at path.", i = path, i = "Set {.code sync = TRUE} or run {.fn gc_ensure_corpus}."))
  }
  df <- nanoparquet::read_parquet(path)
  tibble::as_tibble(df)
}

#' Read the bundled sample Parquet (no network, tiny)
#'
#' For examples and tests when the full corpus is not installed.
#'
#' @return A small [tibble::tibble()].
#' @export
gc_load_corpus_sample <- function() {
  p <- system.file("extdata", "gc_corpus_sample.parquet", package = "conferencestats")
  if (!nzchar(p)) {
    rlang::abort("Sample package data is missing (extdata not installed).")
  }
  tibble::as_tibble(nanoparquet::read_parquet(p))
}
