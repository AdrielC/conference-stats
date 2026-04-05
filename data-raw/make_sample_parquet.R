# Creates inst/extdata/gc_corpus_sample.parquet for examples and unit tests.
# Run from package root: Rscript data-raw/make_sample_parquet.R

if (!requireNamespace("nanoparquet", quietly = TRUE)) {
  install.packages("nanoparquet", repos = "https://cloud.r-project.org")
}

dir.create("inst/extdata", recursive = TRUE, showWarnings = FALSE)

sample <- tibble::tibble(
  talk_id = c("sample_talk_001", "sample_talk_002"),
  text = c(
    "Faith and repentance draw us nearer to the Savior.",
    "We covenant to follow Him with patience and charity."
  ),
  speaker = c("Example Speaker", "Another Leader"),
  title = c("On Faith", "The Covenant Path"),
  year = c(2020L, 2021L),
  month = c(10L, 4L),
  conference_date = as.Date(c("2020-10-01", "2021-04-01")),
  session = c("Saturday Morning", "Sunday Morning"),
  url = c(
    "https://example.org/study/general-conference/2020/10/1talk",
    "https://example.org/study/general-conference/2021/04/2talk"
  ),
  language = c("en", "en"),
  word_count = c(10L, 11L),
  conference_index = c(2020.1, 2021.04),
  season = c("October", "April"),
  corpus_package_version = "0.2.0",
  corpus_built_at = c("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
)

nanoparquet::write_parquet(sample, "inst/extdata/gc_corpus_sample.parquet", compression = "zstd")
message("Wrote inst/extdata/gc_corpus_sample.parquet")
