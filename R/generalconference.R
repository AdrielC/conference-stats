#' Convert `generalconference::genconf` to a flat talk corpus
#'
#' The community-maintained `generalconference` package ships a nested
#' `genconf` object. This helper unnests it to one row per talk with full
#' `text` suitable for [gc_normalize_talks()].
#'
#' @param genconf Nested tibble as in `generalconference::genconf`. If `NULL`,
#'   the data set is loaded from that package.
#' @return A tibble; pipe to [gc_normalize_talks()] for stable `talk_id`s and
#'   derived columns.
#' @export
gc_as_tibble_generalconference <- function(genconf = NULL) {
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
    rlang::abort("Unexpected structure: missing `paragraphs` after unnesting talks.")
  }

  flat <- dplyr::mutate(
    flat,
    text = purrr::map_chr(.data$paragraphs, function(p) {
      if (is.null(p) || (is.atomic(p) && !length(p))) {
        return("")
      }
      if (is.atomic(p)) {
        return(paste(as.character(p), collapse = " "))
      }
      paste(as.character(unlist(p, use.names = FALSE)), collapse = " ")
    }),
    speaker = .data$author1,
    title = .data$title1
  )

  if ("date" %in% names(flat)) {
    flat <- dplyr::mutate(flat, conference_date = as.Date(.data$date))
  } else {
    flat <- dplyr::mutate(
      flat,
      conference_date = gc_make_conference_date(.data$year, .data$month)
    )
  }

  if ("session_name" %in% names(flat)) {
    flat <- dplyr::rename(flat, session = dplyr::all_of("session_name"))
  } else if (!"session" %in% names(flat)) {
    flat$session <- NA_character_
  }

  cols <- c(
    "year", "month", "conference_date", "session",
    "speaker", "title", "url", "text"
  )
  dplyr::select(flat, dplyr::any_of(cols))
}
