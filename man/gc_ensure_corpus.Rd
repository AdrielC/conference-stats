\name{gc_ensure_corpus}
\alias{gc_ensure_corpus}
\title{Ensure the Parquet corpus exists locally}
\usage{
gc_ensure_corpus(path = gc_corpus_cache_path(), force = FALSE,
  download_url = gc_default_corpus_download_url(), quiet = FALSE)
}
\arguments{
\item{path}{Target \code{.parquet} file.}
\item{force}{Re-download or rebuild if the file already exists.}
\item{download_url}{If non-\code{NULL} and non-empty, download; else build via
\code{\link{gc_build_corpus_parquet}}.}
\item{quiet}{Less console output.}
}
\value{
\code{path}, invisibly.
}
\description{
Creates the cache directory if needed. When the file is missing or
\code{force} is \code{TRUE}, either downloads or builds the corpus.
}
