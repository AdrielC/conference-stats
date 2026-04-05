\name{gc_make_conference_date}
\alias{gc_make_conference_date}
\title{Build a conference date from year and month}
\usage{
gc_make_conference_date(year, month)
}
\arguments{
\item{year}{\code{integer} conference year.}
\item{month}{\code{integer} month (\code{4} April, \code{10} October).}
}
\value{
A \code{\link{Date}} vector anchored on the first of the month.
}
\description{
Stable calendar index for joins and plotting.
}
