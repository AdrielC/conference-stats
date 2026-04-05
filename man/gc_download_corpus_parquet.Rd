\name{gc_download_corpus_parquet}
\alias{gc_download_corpus_parquet}
\title{Download a pre-built Parquet corpus}
\usage{
gc_download_corpus_parquet(url, dest = gc_corpus_cache_path(),
  overwrite = FALSE, quiet = FALSE)
}
\arguments{
\item{url}{HTTPS URL to a \code{.parquet} file.}
\item{dest}{Where to save it.}
\item{overwrite}{Allow replacing an existing file.}
\item{quiet}{Passed to \code{\link[utils]{download.file}}.}
}
\value{
\code{dest}, invisibly.
}
\description{
Fetches a corpus artifact (for example attached to a GitHub Release).
}
