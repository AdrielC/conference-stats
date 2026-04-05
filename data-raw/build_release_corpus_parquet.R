# Maintainer script: build full corpus Parquet for a GitHub Release asset.
# Prerequisites: devtools/remotes installed; generalconference from GitHub.
#
#   Rscript data-raw/build_release_corpus_parquet.R
#
# Then upload `general_conference_talks.parquet` to a release (e.g. tag corpus-v1).

if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("generalconference", quietly = TRUE)) {
  devtools::install_github("bryanwhiting/generalconference")
}
if (!requireNamespace("nanoparquet", quietly = TRUE)) {
  install.packages("nanoparquet", repos = "https://cloud.r-project.org")
}

pkg_root <- normalizePath(".", winslash = "/")
devtools::load_all(pkg_root)

out <- file.path(pkg_root, "general_conference_talks.parquet")
gc_build_corpus_parquet(dest = out, overwrite = TRUE)
message("Release artifact: ", out, " (upload this to GitHub Releases)")
