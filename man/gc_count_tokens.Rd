\name{gc_count_tokens}
\alias{gc_count_tokens}
\title{Count token frequencies within each document}
\usage{
gc_count_tokens(token_tbl, document = "talk_id", term = "word", sort = TRUE)
}
\arguments{
\item{token_tbl}{Token table.}
\item{document, term}{Column names.}
\item{sort}{Whether to order by document and descending count.}
}
\value{
Tibble with columns \code{document}, \code{term}, \code{n}.
}
\description{
Per-document term counts prior to matrix factorization or TF--IDF.
}
