\name{gc_filter_talks}
\alias{gc_filter_talks}
\title{Subset talks by conference or speaker}
\usage{
gc_filter_talks(talks, years = NULL, months = NULL, speakers = NULL)
}
\arguments{
\item{talks}{Normalized corpus from \code{\link{read_gc_talks}} or \code{\link{gc_normalize_talks}}.}
\item{years, months}{Optional integer vectors.}
\item{speakers}{Optional character vector; matched case-insensitively as a regex alternation.}
}
\value{
Filtered tibble.
}
\description{
Convenience filters for selecting slices of the corpus.
}
