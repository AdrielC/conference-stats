\name{gc_bind_tf_idf}
\alias{gc_bind_tf_idf}
\title{Compute TF--IDF using tidytext}
\usage{
gc_bind_tf_idf(token_tbl, document = "talk_id", term = "word", n = "n")
}
\arguments{
\item{token_tbl}{Token or count tibble.}
\item{document, term, n}{Column names. Counts are computed when \code{n} is missing.}
}
\value{
Same rows with \code{tf}, \code{idf}, and \code{tf_idf} columns added.
}
\description{
Convenience wrapper around \code{\link[tidytext]{bind_tf_idf}}.
}
