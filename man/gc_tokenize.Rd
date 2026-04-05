\name{gc_tokenize}
\alias{gc_tokenize}
\title{Tokenize talk texts for tidytext workflows}
\usage{
gc_tokenize(talks, input = "text", output = "word", token = "words", ...)
}
\arguments{
\item{talks}{Talk-level data frame.}
\item{input, output}{Column names for source text and token output.}
\item{token}{Tokenizer passed to \code{\link[tidytext]{unnest_tokens}}.}
\item{...}{Additional arguments to \code{\link[tidytext]{unnest_tokens}}.}
}
\value{
One row per token.
}
\description{
Thin wrapper around \code{\link[tidytext]{unnest_tokens}} preserving metadata columns.
}
