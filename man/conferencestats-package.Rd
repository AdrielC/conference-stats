\name{conferencestats-package}
\alias{conferencestats-package}
\alias{conferencestats}
\docType{package}
\title{Conferencestats: General Conference corpus (Parquet) and tidytext pipelines}
\description{
The main workflow is \code{\link{gc_load_corpus}}: materialize a compressed
Apache Parquet file (user cache by default), then analyse with \pkg{tidytext}
(tokens, sparse document--term matrices, TF--IDF). The corpus is built from
\pkg{generalconference} when installed, or downloaded from a URL you configure.
Ad-hoc JSON/CSV exports still work via \code{\link{read_gc_talks}}.
}
\seealso{
\code{\link{gc_load_corpus}}, \code{\link{gc_tokenize}}, \code{\link{gc_document_term_matrix}}
}
