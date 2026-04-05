\name{gc_summarize_tokens_by_conference}
\alias{gc_summarize_tokens_by_conference}
\title{Aggregate token statistics by conference date}
\usage{
gc_summarize_tokens_by_conference(token_tbl, word_col = "word", fun = sum)
}
\arguments{
\item{token_tbl}{Must include \code{conference_date}, \code{talk_id}, and the token column.}
\item{word_col}{Token column name.}
\item{fun}{Summary applied to per-talk counts within a conference.}
}
\value{
Tibble grouped by conference and token.
}
\description{
Helper for diachronic plots, for example prescriptive language rates over time.
}
