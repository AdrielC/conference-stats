#' Expand `genconf` to one row per paragraph
#'
#' Works with `generalconference` v0.3.2-style `paragraphs` list columns.
#' Use for chunk-level embeddings or span detection before pasting into a full talk.
#'
#' @inheritParams gc_as_tibble_generalconference
#' @return A [tibble::tibble()] with `year`, `month`, `speaker`, `title`, `url`,
#'   `session`, `paragraph_index`, `text`.
#' @export
gc_as_paragraph_tibble_generalconference <- function(genconf = NULL) {
  if (is.null(genconf)) {
    if (!requireNamespace("generalconference", quietly = TRUE)) {
      rlang::abort("Install the generalconference package to use this helper.")
    }
    env <- new.env(parent = emptyenv())
    utils::data("genconf", package = "generalconference", envir = env)
    genconf <- env$genconf
  }

  flat <- genconf |>
    tidyr::unnest(cols = dplyr::all_of("sessions")) |>
    tidyr::unnest(cols = dplyr::all_of("talks"))

  if (!"paragraphs" %in% names(flat)) {
    rlang::abort("Unexpected structure: missing `paragraphs`.")
  }

  out <- vector("list", nrow(flat))
  for (i in seq_len(nrow(flat))) {
    p <- flat$paragraphs[[i]]
    if (is.null(p)) {
      next
    }
    p <- as.character(unlist(p, use.names = FALSE))
    p <- p[nzchar(stringr::str_squish(p))]
    if (!length(p)) {
      next
    }
    ssn <- if ("session_name" %in% names(flat)) flat$session_name[[i]] else NA_character_
    out[[i]] <- tibble::tibble(
      year = flat$year[[i]],
      month = flat$month[[i]],
      speaker = as.character(flat$author1[[i]]),
      title = as.character(flat$title1[[i]]),
      url = as.character(flat$url[[i]]),
      session = ssn,
      paragraph_index = seq_along(p),
      text = p
    )
  }
  dplyr::bind_rows(out)
}
