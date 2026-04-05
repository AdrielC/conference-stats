\name{gc_as_paragraph_tibble_generalconference}
\alias{gc_as_paragraph_tibble_generalconference}
\title{Expand genconf to one row per paragraph}
\usage{
gc_as_paragraph_tibble_generalconference(genconf = NULL)
}
\arguments{
\item{genconf}{Nested \code{genconf} object; \code{NULL} loads from \pkg{generalconference}.}
}
\value{
Tibble with paragraph-level \code{text} and talk metadata.
}
\description{
For chunk-level TF--IDF weighting and mean-pooled word embeddings aligned with
a word-vector vocabulary (e.g. GloVe).
}
