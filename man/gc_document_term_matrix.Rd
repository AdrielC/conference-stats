\name{gc_document_term_matrix}
\alias{gc_document_term_matrix}
\title{Build a sparse document--term matrix}
\usage{
gc_document_term_matrix(token_tbl, document = "talk_id", term = "word",
  value = "n")
}
\arguments{
\item{token_tbl}{Token- or count-level tibble.}
\item{document, term, value}{Column names; counts are computed when \code{value} is absent.}
}
\value{
A \code{\link[Matrix]{sparseMatrix}} in compressed column form (\code{dgCMatrix}).
}
\description{
Wraps \code{\link[tidytext]{cast_sparse}} for downstream topic models or embeddings export.
}
