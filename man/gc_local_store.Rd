\name{gc_local_store}
\alias{gc_local_store}
\alias{gc_store_read}
\alias{gc_store_read.gc_local_store}
\title{Local corpus storage handles}
\usage{
gc_local_store(root)
gc_store_read(store, ...)
\method{gc_store_read}{gc_local_store}(store, ...)
}
\arguments{
\item{root}{Directory containing ingestible files (see \code{\link{read_gc_talks}}).}
\item{store}{Handle produced by \code{gc_local_store}.}
\item{...}{Arguments forwarded to \code{\link{read_gc_talks}}.}
}
\value{
\code{gc_store_read} returns a normalized corpus tibble.
}
\description{
Lightweight abstraction so cloud backends can mirror the same interface later.
}
