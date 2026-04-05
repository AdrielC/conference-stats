\name{gc_build_corpus_parquet}
\alias{gc_build_corpus_parquet}
\title{Build the talk corpus and write Parquet}
\usage{
gc_build_corpus_parquet(dest = gc_corpus_cache_path(),
  overwrite = FALSE, compression = "zstd")
}
\arguments{
\item{dest}{Output \code{.parquet} path.}
\item{overwrite}{Replace existing file.}
\item{compression}{Codec for \code{\link[nanoparquet]{write_parquet}}.}
}
\value{
\code{dest}, invisibly.
}
\description{
Flattens \code{generalconference::genconf}, normalizes rows, adds provenance
columns, and writes a compressed Parquet file. Requires Suggested package
\pkg{generalconference}.
}
