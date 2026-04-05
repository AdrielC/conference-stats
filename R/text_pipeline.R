#' Tokenize talk texts for tidytext workflows
#'
#' Thin wrapper around [tidytext::unnest_tokens()] that keeps all document
#' metadata columns present in `talks`.
#'
#' @param talks A data frame with talk-level rows (e.g. from [read_gc_talks()]).
#' @param input Name of the column holding full text.
#' @param output Name of the token column to create (default `"word"`).
#' @param token Passed to [tidytext::unnest_tokens()] (`"words"` by default).
#' @param ... Additional arguments to [tidytext::unnest_tokens()].
#' @return A tibble with one row per token.
#' @export
gc_tokenize <- function(talks, input = "text", output = "word", token = "words", ...) {
  if (!input %in% names(talks)) {
    rlang::abort(paste0('Column "', input, '" is not in `talks`.'))
  }
  col <- rlang::sym(input)
  tidytext::unnest_tokens(
    talks,
    !!rlang::sym(output),
    !!col,
    token = token,
    ...
  )
}

#' Remove stopwords from a token table
#'
#' Uses [tidytext::get_stopwords()] by default (Snowball English). Supply a
#' character vector of custom stopwords instead if needed.
#'
#' @param token_tbl Output of [gc_tokenize()].
#' @param word_col Name of the token column.
#' @param custom_stopwords Optional character vector; if set, `source` and
#'   `language` are ignored.
#' @param source,language Passed to [tidytext::get_stopwords()] when
#'   `custom_stopwords` is `NULL`.
#' @export
gc_remove_stopwords <- function(token_tbl,
                               word_col = "word",
                               custom_stopwords = NULL,
                               source = "snowball",
                               language = "en") {
  stops <- if (!is.null(custom_stopwords)) {
    custom_stopwords
  } else {
    tidytext::get_stopwords(source = source, language = language)$word
  }
  stop_tbl <- tibble::tibble(!!word_col := stops)
  dplyr::anti_join(token_tbl, stop_tbl, by = word_col)
}

#' Count token frequencies within each document
#'
#' @param token_tbl Token table from [gc_tokenize()] (and optional filtering).
#' @param document Name of the document id column (typically `talk_id`).
#' @param term Name of the token column (typically `word`).
#' @param sort Whether to sort by document then descending count.
#' @export
gc_count_tokens <- function(token_tbl,
                            document = "talk_id",
                            term = "word",
                            sort = TRUE) {
  out <- token_tbl |>
    dplyr::count(.data[[document]], .data[[term]], name = "n")
  if (sort) {
    out <- dplyr::arrange(out, .data[[document]], dplyr::desc(.data$n))
  }
  out
}

#' Build a sparse document–term matrix
#'
#' Wrapper around [tidytext::cast_sparse()]. If `value` is absent, token counts
#' are computed with [gc_count_tokens()].
#'
#' @param token_tbl Token-level tibble.
#' @param document,term,value Column names (defaults: `talk_id`, `word`, `n`).
#' @export
gc_document_term_matrix <- function(token_tbl,
                                    document = "talk_id",
                                    term = "word",
                                    value = "n") {
  if (!value %in% names(token_tbl)) {
    token_tbl <- gc_count_tokens(
      token_tbl,
      document = document,
      term = term,
      sort = FALSE
    )
  }
  tidytext::cast_sparse(
    token_tbl,
    !!rlang::sym(document),
    !!rlang::sym(term),
    !!rlang::sym(value)
  )
}

#' Compute TF-IDF using tidytext
#'
#' Expects one row per document–term pair with a count column. Counts are
#' computed automatically when missing.
#'
#' @inheritParams gc_document_term_matrix
#' @export
gc_bind_tf_idf <- function(token_tbl,
                           document = "talk_id",
                           term = "word",
                           n = "n") {
  if (!n %in% names(token_tbl)) {
    token_tbl <- gc_count_tokens(
      token_tbl,
      document = document,
      term = term,
      sort = FALSE
    )
  }
  tidytext::bind_tf_idf(
    token_tbl,
    !!rlang::sym(term),
    !!rlang::sym(document),
    !!rlang::sym(n)
  )
}

#' Aggregate token statistics by conference date
#'
#' Convenience helper for time-series plots (lexical rates, topical shares, etc.).
#'
#' @param token_tbl Token table that still contains `conference_date`, `talk_id`,
#'   and the token column.
#' @param word_col Token column name.
#' @param fun Summary function applied to per-talk counts within each conference.
#' @export
gc_summarize_tokens_by_conference <- function(token_tbl,
                                               word_col = "word",
                                               fun = sum) {
  req <- c("conference_date", "talk_id", word_col)
  if (!all(req %in% names(token_tbl))) {
    rlang::abort(
      "`token_tbl` must include conference_date, talk_id, and the token column."
    )
  }
  if (!"n" %in% names(token_tbl)) {
    token_tbl <- dplyr::count(
      token_tbl,
      .data$conference_date,
      .data$talk_id,
      !!rlang::sym(word_col),
      name = "n"
    )
  }
  token_tbl |>
    dplyr::group_by(.data$conference_date, !!rlang::sym(word_col)) |>
    dplyr::summarise(n_tokens = fun(.data$n), .groups = "drop")
}
