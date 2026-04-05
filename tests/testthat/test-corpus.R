test_that("read_gc_talks ingests json lines and normalizes metadata", {
  dir <- tempfile("gc")
  dir.create(dir)

  writeLines(c(
    jsonlite::toJSON(list(
      text = "Faith and repentance.",
      speaker = "Example Speaker",
      title = "On Faith",
      year = 2020L,
      month = 10L
    ), auto_unbox = TRUE),
    jsonlite::toJSON(list(
      text = "Second talk body.",
      author = "Another Author",
      title1 = "Covenant Path",
      year = 2021L,
      month = 4L
    ), auto_unbox = TRUE)
  ), file.path(dir, "tiny.jsonl"), useBytes = FALSE)

  talks <- read_gc_talks(dir)
  expect_equal(nrow(talks), 2L)
  expect_true(all(c("talk_id", "word_count", "conference_date", "season") %in% names(talks)))
  expect_equal(sort(unique(talks$year)), c(2020L, 2021L))
})

test_that("token pipeline builds counts and sparse matrix", {
  talks <- tibble::tibble(
    talk_id = c("a", "b"),
    conference_date = as.Date(c("2020-10-01", "2021-04-01")),
    text = c("faith hope charity", "faith and patience"),
    year = c(2020L, 2021L),
    month = c(10L, 4L),
    speaker = NA_character_,
    title = NA_character_,
    session = NA_character_,
    url = NA_character_,
    language = NA_character_,
    word_count = 3L,
    conference_index = c(2020.1, 2021.04),
    season = c("October", "April")
  )

  toks <- gc_tokenize(talks)
  expect_true(all(c("talk_id", "word") %in% names(toks)))
  cm <- gc_count_tokens(toks, sort = FALSE)
  mat <- gc_document_term_matrix(cm)
  expect_s4_class(mat, "dgCMatrix")
  tf <- gc_bind_tf_idf(cm)
  expect_true(all(c("tf", "idf", "tf_idf") %in% names(tf)))
})

test_that("bundled sample Parquet loads", {
  skip_if_not(file.exists(
    system.file("extdata", "gc_corpus_sample.parquet", package = "conferencestats")
  ))
  s <- gc_load_corpus_sample()
  expect_equal(nrow(s), 2L)
  expect_true("corpus_package_version" %in% names(s))
})

test_that("Parquet roundtrip preserves columns", {
  tmp <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp), add = TRUE)
  talks <- gc_normalize_talks(tibble::tibble(
    text = c("one two", "three four five"),
    year = c(2019L, 2020L),
    month = c(4L, 10L),
    speaker = c("x", "y"),
    title = c("t1", "t2"),
    session = NA_character_,
    url = NA_character_,
    language = "en"
  ))
  talks$corpus_package_version <- "0"
  talks$corpus_built_at <- "2026-01-01T00:00:00Z"
  nanoparquet::write_parquet(talks, tmp, compression = "zstd")
  back <- tibble::as_tibble(nanoparquet::read_parquet(tmp))
  expect_equal(nrow(back), nrow(talks))
  expect_true(setequal(names(talks), names(back)))
})
