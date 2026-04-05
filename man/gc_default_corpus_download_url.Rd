\name{gc_default_corpus_download_url}
\alias{gc_default_corpus_download_url}
\title{Optional pre-built corpus URL}
\usage{
gc_default_corpus_download_url()
}
\value{
A single string or \code{NULL}.
}
\description{
Reads \code{CONFERENCESTATS_CORPUS_URL} or the R option
\code{conferencestats.corpus_parquet_url} so \code{\link{gc_ensure_corpus}}
can download a release artifact instead of building from
\pkg{generalconference}.
}
