#' Local corpus storage reference
#'
#' Lightweight handle for a directory of corpus files. Cloud backends (for
#' example S3 via `paws.storage`) can follow the same pattern in future
#' releases.
#'
#' @param root Absolute or project-relative directory containing talk files
#'   consumable by [read_gc_talks()].
#' @return An object of class `gc_local_store`.
#' @export
gc_local_store <- function(root) {
  root <- as.character(root)[[1L]]
  if (!nzchar(root)) {
    rlang::abort("`root` must be a non-empty path.")
  }
  structure(list(root = root), class = c("gc_local_store", "gc_store"))
}

#' Read all talks known to a store
#'
#' @param store Result of [gc_local_store()].
#' @param ... Passed to [read_gc_talks()] (`pattern`, `recursive`, etc.).
#' @export
gc_store_read <- function(store, ...) {
  UseMethod("gc_store_read")
}

#' @export
gc_store_read.gc_local_store <- function(store, ...) {
  read_gc_talks(store$root, ...)
}

print.gc_local_store <- function(x, ...) {
  cat("<gc_local_store>", x$root, "\n", sep = "")
  invisible(x)
}
