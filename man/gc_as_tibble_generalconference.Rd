\name{gc_as_tibble_generalconference}
\alias{gc_as_tibble_generalconference}
\title{Flatten \code{generalconference::genconf} into talk rows}
\usage{
gc_as_tibble_generalconference(genconf = NULL)
}
\arguments{
\item{genconf}{Nested tibble; \code{NULL} loads the packaged \code{genconf} object.}
}
\value{
Wide tibble with \code{text} suitable for \code{\link{gc_normalize_talks}}.
}
\description{
Requires the external \pkg{generalconference} package.
}
