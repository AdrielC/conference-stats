\name{gc_corpus_cache_dir}
\alias{gc_corpus_cache_dir}
\alias{gc_corpus_cache_path}
\title{Corpus cache location}
\usage{
gc_corpus_cache_dir(filename = NULL)
gc_corpus_cache_path()
}
\arguments{
\item{filename}{If \code{NULL}, returns the cache directory. Otherwise returns
\code{file.path(cache_dir, filename)}.}
}
\value{
Character path.
}
\description{
Default directory for the General Conference Parquet file, via
\code{\link[rappdirs]{user_cache_dir}} (\pkg{conferencestats} app name).
}
