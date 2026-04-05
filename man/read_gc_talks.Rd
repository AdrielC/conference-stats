\name{read_gc_talks}
\alias{read_gc_talks}
\title{Read General Conference--style talks from local files}
\usage{
read_gc_talks(path, pattern = "\\\\.(jsonl|ndjson|json|csv|tsv)$",
  recursive = TRUE)
}
\arguments{
\item{path}{Path to a file or directory of supported files.}
\item{pattern}{File pattern when \code{path} is a directory.}
\item{recursive}{Passed to \code{\link{list.files}}.}
}
\value{
A tibble with standardized columns; see \code{\link{gc_normalize_talks}}.
}
\description{
Read JSON, JSON Lines, CSV, or TSV sources and normalize them into an analysis-ready corpus.
}
