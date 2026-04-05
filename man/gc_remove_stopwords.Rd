\name{gc_remove_stopwords}
\alias{gc_remove_stopwords}
\title{Remove stopwords from a token table}
\usage{
gc_remove_stopwords(token_tbl, word_col = "word",
  custom_stopwords = NULL, source = "snowball", language = "en")
}
\arguments{
\item{token_tbl}{Token table from \code{\link{gc_tokenize}}.}
\item{word_col}{Token column name.}
\item{custom_stopwords}{Optional character vector of stopwords.}
\item{source, language}{Passed to \code{\link[tidytext]{get_stopwords}} when \code{custom_stopwords} is \code{NULL}.}
}
\value{
Filtered token table.
}
\description{
Wraps \code{\link[tidytext]{get_stopwords}} defaults or a custom list.
}
