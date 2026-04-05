#' Normalize talk corpora to a standard tibble
#'
#' Ensures `talk_id`, parsed calendar fields, conference index (year + month),
#' and word counts. Rows with missing `text` are dropped.
#'
#' @param talks A data frame with at least `text` (character).
#' @param language Default language code if missing (e.g. `"en"`).
#' @return A [tibble::tibble()] sorted by `conference_date`, `talk_id`.
#' @export
gc_normalize_talks <- function(talks, language = NA_character_) {
  if (!is.data.frame(talks)) {
    rlang::abort("`talks` must be a data frame.")
  }
  talks <- tibble::as_tibble(talks)
  if (!"text" %in% names(talks)) {
    rlang::abort('Data must contain a column named "text".')
  }
  for (nm in c("speaker", "title", "session", "url", "language")) {
    if (!nm %in% names(talks)) {
      talks[[nm]] <- NA_character_
    }
  }
  for (nm in c("year", "month")) {
    if (!nm %in% names(talks)) {
      talks[[nm]] <- NA_integer_
    }
  }

  talks <- talks |>
    dplyr::mutate(
      text = as.character(.data$text),
      dplyr::across(
        dplyr::all_of(c("speaker", "title", "session", "url", "language")),
        ~ as.character(.x)
      ),
      dplyr::across(dplyr::all_of(c("year", "month")), ~ suppressWarnings(as.integer(.x)))
    ) |>
    dplyr::filter(!is.na(.data$text), nzchar(.data$text))

  if (!"language" %in% names(talks) || all(is.na(talks$language))) {
    talks$language <- language
  }
  if (!"conference_date" %in% names(talks)) {
    talks$conference_date <- as.Date(NA)
  } else {
    talks$conference_date <- as.Date(talks$conference_date)
  }

  talks <- gc_fill_conference_dates(talks)
  dplyr::mutate(
    talks,
    talk_id = .make_talk_id(
      .data$year,
      .data$month,
      .data$speaker,
      .data$title,
      .data$url
    ),
    word_count = .count_words(.data$text),
    conference_index = .conference_index(.data$year, .data$month),
    season = .season_label(.data$month)
  ) |>
    dplyr::arrange(.data$conference_date, .data$talk_id)
}

#' Build a conference date from year and month
#'
#' April and October General Conferences are encoded as the first day of that
#' month (a stable index for plotting and joins).
#'
#' @param year,month Integer vectors for April (`4`) and October (`10`) sessions.
#' @return A vector of [Date] values.
#' @export
gc_make_conference_date <- function(year, month) {
  as.Date(sprintf("%04d-%02d-01", as.integer(year), as.integer(month)))
}

gc_fill_conference_dates <- function(talks) {
  if (!all(c("year", "month") %in% names(talks))) {
    return(talks)
  }
  need <- is.na(talks$conference_date) &
    !is.na(talks$year) & !is.na(talks$month)
  if (any(need)) {
    talks$conference_date[need] <- gc_make_conference_date(
      talks$year[need],
      talks$month[need]
    )
  }
  talks
}

.make_talk_id <- function(year, month, speaker, title, url) {
  yr <- ifelse(is.na(year), "NA", sprintf("%04d", as.integer(year)))
  mo <- ifelse(is.na(month), "NA", sprintf("%02d", as.integer(month)))
  sp <- ifelse(is.na(speaker), "", speaker)
  tt <- ifelse(is.na(title), "", title)
  u <- ifelse(is.na(url), "", url)
  keys <- paste(yr, mo, sp, tt, u, sep = "|")
  vapply(keys, rlang::hash, character(1), USE.NAMES = FALSE)
}

.count_words <- function(text) {
  sq <- stringr::str_squish(text)
  wc <- stringr::str_count(sq, "\\s+") + 1L
  wc[sq == "" | is.na(sq)] <- 0L
  wc
}

.conference_index <- function(year, month) {
  ifelse(
    is.na(year) | is.na(month),
    NA_real_,
    as.numeric(year) + as.numeric(month) / 100
  )
}

.season_label <- function(month) {
  m <- as.integer(month)
  lab <- rep(NA_character_, length(m))
  lab[m == 4L] <- "April"
  lab[m == 10L] <- "October"
  lab
}

#' Subset talks by conference year, season, or speaker
#'
#' @param talks A normalized corpus from [gc_normalize_talks()] or [read_gc_talks()].
#' @param years Optional integer vector of conference years to keep.
#' @param months Optional integer vector (typically `4` or `10`).
#' @param speakers Optional character vector (substring match, case-insensitive).
#' @export
gc_filter_talks <- function(talks, years = NULL, months = NULL, speakers = NULL) {
  out <- talks
  if (!is.null(years)) {
    out <- dplyr::filter(out, .data$year %in% years)
  }
  if (!is.null(months)) {
    out <- dplyr::filter(out, .data$month %in% months)
  }
  if (!is.null(speakers)) {
    patt <- paste(stringr::str_escape(speakers), collapse = "|")
    out <- dplyr::filter(
      out,
      stringr::str_detect(.data$speaker, stringr::regex(patt, ignore_case = TRUE))
    )
  }
  out
}
