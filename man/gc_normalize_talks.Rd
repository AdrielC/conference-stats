\name{gc_normalize_talks}
\alias{gc_normalize_talks}
\title{Normalize talk corpora to a standard tibble}
\usage{
gc_normalize_talks(talks, language = NA_character_)
}
\arguments{
\item{talks}{Data frame with at least a \code{text} column.}
\item{language}{Default language code when missing.}
}
\value{
A tibble sorted by \code{conference_date} and \code{talk_id}.
}
\description{
Adds \code{talk_id}, conference dates, seasons, and word counts, dropping empty texts.
}
