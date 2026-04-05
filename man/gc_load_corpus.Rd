\name{gc_load_corpus}
\alias{gc_load_corpus}
\title{Load the General Conference corpus}
\usage{
gc_load_corpus(path = gc_corpus_cache_path(), sync = TRUE,
  force = FALSE, download_url = gc_default_corpus_download_url(),
  quiet = FALSE)
}
\arguments{
\item{path}{Parquet file to read.}
\item{sync}{Call \code{\link{gc_ensure_corpus}} first.}
\item{force, download_url, quiet}{Forwarded to \code{\link{gc_ensure_corpus}} when
\code{sync} is true.}
}
\value{
A \code{\link[tibble]{tibble}} of normalized talks.
}
\description{
Primary entry point: populate the cache if needed, then return all talks as a
data frame.
}
