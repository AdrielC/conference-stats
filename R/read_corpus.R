#' Read General Conference–style talks from local files
#'
#' Expects one row per talk with at least a full-text column. Optional columns
#' are preserved and standardized when present (see Details).
#'
#' **JSON / JSON Lines**: either an array of objects or one JSON object per
#' line. Recognised field names (first match wins for text and speaker):
#' - Text: `text`, `body`, `content`, `transcript`
#' - Speaker: `speaker`, `author`, `author1`, `presenter`
#' - Title: `title`, `title1`, `name`
#' - Year / session: `year`, `month`, `conference_date`, `date`, `session`, `url`, `language`
#'
#' **CSV / TSV**: column names as above; delimiter is chosen from the file
#' extension (comma or tab). For other table sources, import yourself and pass
#' a data frame to [gc_normalize_talks()].
#'
#' @param path Path to a file (`.json`, `.jsonl`, `.ndjson`, `.csv`, `.tsv`)
#'   or directory of such files.
#' @param pattern When `path` is a directory, regex passed to [list.files()]
#'   (default: JSON and common table extensions).
#' @param recursive Passed to [list.files()].
#' @return A [tibble::tibble()] with standardized columns; see [gc_normalize_talks()].
#' @export
#' @examples
#' \dontrun{
#' talks <- read_gc_talks("/data/gc/talks.jsonl")
#' }
read_gc_talks <- function(path, pattern = "\\.(jsonl|ndjson|json|csv|tsv)$",
                          recursive = TRUE) {
  path <- as.character(path)
  if (!length(path) || anyNA(path)) {
    rlang::abort("`path` must be a non-missing character path.")
  }
  path <- path[[1L]]
  if (!file.exists(path)) {
    rlang::abort(c("Path does not exist.", i = path))
  }

  files <- if (dir.exists(path)) {
    list.files(
      path,
      pattern = pattern,
      full.names = TRUE,
      recursive = recursive,
      ignore.case = TRUE
    )
  } else {
    path
  }

  if (!length(files)) {
    rlang::abort(c("No matching files found.", i = path))
  }

  ext <- tolower(tools::file_ext(files))
  out <- vector("list", length(files))
  for (i in seq_along(files)) {
    out[[i]] <- switch(ext[i],
      json = .read_gc_one_json(files[[i]]),
      jsonl = ,
      ndjson = .read_gc_json_lines(files[[i]]),
      csv = .read_gc_table(files[[i]], delim = ","),
      tsv = .read_gc_table(files[[i]], delim = "\t"),
      rlang::abort(c(
        "Unsupported file extension.",
        i = files[[i]],
        i = paste("Extension:", ext[i])
      ))
    )
  }
  gc_normalize_talks(dplyr::bind_rows(out))
}

.read_gc_one_json <- function(file) {
  raw <- jsonlite::read_json(file, simplifyVector = FALSE)
  if (!is.list(raw)) {
    rlang::abort(c("JSON root must be an object or array.", i = file))
  }
  if (!length(raw)) {
    return(tibble::tibble())
  }
  if (.is_bare_json_array(raw)) {
    dplyr::bind_rows(purrr::map(raw, .as_talk_row))
  } else {
    .as_talk_row(raw)
  }
}

.is_bare_json_array <- function(raw) {
  is.null(names(raw))
}

.read_gc_json_lines <- function(file) {
  con <- file(file, open = "r", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  lines <- readLines(con, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nzchar(lines)]
  if (!length(lines)) {
    return(tibble::tibble())
  }
  dplyr::bind_rows(purrr::map(lines, function(ln) {
    obj <- jsonlite::fromJSON(ln, simplifyVector = FALSE)
    .as_talk_row(obj)
  }))
}

.read_gc_table <- function(file, delim) {
  tbl <- utils::read.table(
    file,
    header = TRUE,
    sep = delim,
    quote = "\"",
    fill = TRUE,
    stringsAsFactors = FALSE,
    fileEncoding = "UTF-8"
  )
  tibble::as_tibble(tbl)
}

.as_talk_row <- function(x) {
  if (!is.list(x)) {
    rlang::abort("Each talk must be a JSON object or list.")
  }
  text <- .first_chr(x, c("text", "body", "content", "transcript"))
  if (is.na(text) || !nzchar(text)) {
    rlang::abort("Talk is missing a text field (text/body/content/transcript).")
  }
  tibble::tibble(
    text = text,
    speaker = .first_chr(x, c("speaker", "author", "author1", "presenter")),
    title = .first_chr(x, c("title", "title1", "name")),
    year = .first_int(x, c("year")),
    month = .first_int(x, c("month")),
    conference_date = .first_date(x, c("conference_date", "date")),
    session = .first_chr(x, c("session_name", "session")),
    url = .first_chr(x, c("url", "talk_url", "href")),
    language = .first_chr(x, c("language", "lang")),
    source_file = NA_character_
  )
}

.first_chr <- function(x, nms) {
  for (nm in nms) {
    if (!is.null(x[[nm]])) {
      v <- x[[nm]]
      if (length(v) == 1L) {
        return(as.character(v))
      }
    }
  }
  NA_character_
}

.first_int <- function(x, nms) {
  for (nm in nms) {
    if (!is.null(x[[nm]])) {
      v <- suppressWarnings(as.integer(x[[nm]]))
      if (length(v) == 1L && !is.na(v)) {
        return(v)
      }
    }
  }
  NA_integer_
}

.first_date <- function(x, nms) {
  for (nm in nms) {
    if (!is.null(x[[nm]])) {
      v <- x[[nm]]
      if (inherits(v, "Date")) {
        return(v)
      }
      if (is.character(v) && length(v) == 1L) {
        d <- suppressWarnings(as.Date(v))
        if (!is.na(d)) {
          return(d)
        }
      }
    }
  }
  as.Date(NA)
}
