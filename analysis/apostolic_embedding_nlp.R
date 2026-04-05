# Apostolic + First Presidency corpus: prescriptive language over time using
# TF–IDF-weighted *mean-pooled GloVe word embeddings* (word tokenizer aligned
# with GloVe: lowercase, non-alphanumeric → space, split on whitespace).
#
# TF–IDF-weighted average embedding: standard practice (see e.g. arXiv:1902.09875;
# surveys often cite unweighted vs IDF-weighted pooling). Here w_i ∝ tf_i * idf_i
# computed *per chunk*, with idf from all chunks as pseudo-documents.
#
# Subword models (e.g. MPNet / E5): require the *same* tokenizer as the encoder.
# Porting this script to sentence-transformers + reticulate means building tf–idf
# (or similar) on *subword* tokens and pooling last-layer vectors — doable but
# heavier; GloVe keeps token↔embedding alignment explicit.
#
# Prototypes:
#   - Prescriptive: centroid of chunk embeddings for chunks matching a *broad*
#     regex bundle (prohibitions, “must not”, “do not”, “woe unto”, etc.).
#   - Gentle/Christ-invitation: centroid of chunks matching a contrasting bundle.
# Talk score: mean chunk cosine to prescriptive prototype minus mean cosine to
# gentle prototype (both constructed from L2-normalized weighted chunk vectors).
#
# Requires: generalconference v0.3.2, conferencestats sources, dplyr, stringr,
#   purrr, Matrix, ggplot2, mgcv, rappdirs, pkgload; data.table (fast GloVe read).

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(purrr)
  library(Matrix)
  library(ggplot2)
  library(mgcv)
  library(rappdirs)
})

pkg_root <- Sys.getenv("CONFERENCESTATS_ROOT", unset = normalizePath(".."))
if (!dir.exists(file.path(pkg_root, "R"))) {
  stop("Set CONFERENCESTATS_ROOT to the package root (parent of analysis/).")
}
pkgload::load_all(pkg_root, quiet = TRUE)

if (!requireNamespace("generalconference", quietly = TRUE)) {
  stop("Install generalconference: remotes::install_github('bryanwhiting/generalconference@v0.3.2')")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", repos = "https://cloud.r-project.org", quiet = TRUE)
}

# ---- GloVe tokenizer (matches common English GloVe preprocessing) --------------------
glove_tokenize <- function(text) {
  x <- str_to_lower(as.character(text))
  x <- str_replace_all(x, "[^a-z0-9\\s]", " ")
  w <- str_split(str_squish(x), regex("\\s+"), simplify = FALSE)[[1]]
  w[nzchar(w)]
}

# ---- Apostolic / First Presidency filter (broader than presidents-only) ------------
pres_pat <- paste(
  "Russell M\\. Nelson",
  "Thomas S\\. Monson",
  "Gordon B\\. Hinckley",
  "Howard W\\. Hunter",
  "Ezra Taft Benson",
  "Harold B\\. Lee",
  "Joseph Fielding Smith",
  "David O\\. McKay",
  "George Albert Smith",
  "Heber J\\. Grant",
  "Spencer W\\. Kimball",
  sep = "|"
)

is_apostolic_or_fp_talk <- function(author1) {
  a <- str_squish(str_remove(author1, "^By\\s+"))
  if (!nzchar(a)) {
    return(FALSE)
  }
  if (str_starts(a, "Sister ")) {
    return(FALSE)
  }
  if (str_detect(a, "Acting President")) {
    return(FALSE)
  }
  if (str_detect(a, regex(
    "General President|Relief Society General|Young Women General|Young Men General|Sunday School General|Primary General|General Officer",
    ignore_case = TRUE
  ))) {
    return(FALSE)
  }
  if (str_detect(a, regex("Area Seventy", ignore_case = TRUE))) {
    return(FALSE)
  }
  # Church President
  if (str_detect(a, regex(pres_pat))) {
    return(TRUE)
  }
  # Apostle / Seventy — spoken as “Elder …” (includes Seventies; narrow later if needed)
  if (str_detect(a, regex("^Elder [A-Z]", ignore_case = FALSE))) {
    return(TRUE)
  }
  # First Presidency counselors: “President Dallin H. Oaks” etc. (not auxiliary org presidents)
  if (str_detect(a, regex("^President [A-Z]", ignore_case = FALSE)) &&
    !str_detect(a, regex(
      "General President|Relief Society|Young Women|Young Men|Sunday School|Primary|Humanitarian",
      ignore_case = TRUE
    ))) {
    return(TRUE)
  }
  FALSE
}

# ---- GloVe download / read ------------------------------------------------------------
ensure_glove_100d <- function(max_vocab_lines = NULL) {
  custom <- Sys.getenv("CONFERENCESTATS_GLOVE_TXT", "")
  if (nzchar(custom)) {
    gpath <- custom
    if (!file.exists(gpath)) {
      stop("CONFERENCESTATS_GLOVE_TXT set but file missing: ", gpath)
    }
  } else {
  cache <- rappdirs::user_cache_dir("conferencestats", "conferencestats")
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  gpath <- file.path(cache, "glove.6B.100d.txt")
  if (!file.exists(gpath)) {
    z <- file.path(cache, "glove.6B.zip")
    message("Downloading GloVe 6B zip (~862 MB once); extracting 100d file …")
    utils::download.file(
      "https://nlp.stanford.edu/data/glove.6B.zip",
      z,
      mode = "wb"
    )
    utils::unzip(z, files = "glove.6B.100d.txt", exdir = cache, overwrite = TRUE)
    unlink(z)
  }
  }
  message("Reading GloVe matrix (data.table) …")
  g <- data.table::fread(
    gpath,
    sep = " ",
    header = FALSE,
    quote = "",
    nrows = max_vocab_lines
  )
  words <- g[[1L]]
  M <- as.matrix(g[, -1L])
  rownames(M) <- words
  storage.mode(M) <- "double"
  M
}

# ---- Word windows within a paragraph -------------------------------------------------
windowize_words <- function(words, size = 200L, stride = 120L, min_len = 15L) {
  n <- length(words)
  if (n < min_len) {
    return(list())
  }
  if (n <= size) {
    return(list(words))
  }
  starts <- unique(c(seq(1L, max(1L, n - size + 1L), by = stride), n - size + 1L))
  map(starts, function(s) {
    words[seq(s, min(s + size - 1L, n))]
  })
}

# ---- Global chunk IDF (chunks as “documents”) ---------------------------------------
chunk_document_frequency <- function(chunk_tokens_list) {
  N <- length(chunk_tokens_list)
  freq <- new.env(parent = emptyenv(), hash = TRUE)
  for (toks in chunk_tokens_list) {
    for (t in unique(toks)) {
      freq[[t]] <- if (is.null(freq[[t]])) 1L else freq[[t]] + 1L
    }
  }
  terms <- ls(freq)
  idf <- numeric(length(terms))
  names(idf) <- terms
  for (t in terms) {
    idf[t] <- log((N + 1L) / (freq[[t]] + 1L)) + 1
  }
  idf
}

tfidf_pool_embedding <- function(tokens, E, idf_env_named_vector) {
  tokens <- tokens[tokens %in% rownames(E)]
  if (length(tokens) < 5L) {
    return(rep(NA_real_, ncol(E)))
  }
  tab <- table(tokens)
  types <- names(tab)
  tf <- as.numeric(tab) / sum(tab)
  idi <- idf_env_named_vector[types]
  idi[is.na(idi)] <- 0
  w <- tf * idi
  sw <- sum(w)
  if (sw <= 0) {
    return(rep(NA_real_, ncol(E)))
  }
  v <- w %*% E[types, , drop = FALSE] / sw
  v <- as.numeric(v)
  v <- v / sqrt(sum(v * v))
  v
}

centroid_l2 <- function(mat) {
  ok <- rowSums(is.na(mat)) == 0
  if (!any(ok)) {
    return(rep(NA_real_, ncol(mat)))
  }
  v <- colMeans(mat[ok, , drop = FALSE])
  v / sqrt(sum(v * v))
}

# ---- Regex bundles for exemplar mining ---------------------------------------------
presc_rx <- regex(
  paste(
    "do\\s+not",
    "don'?t",
    "can\\s+not",
    "cannot",
    "must\\s+not",
    "should\\s+not",
    "ought\\s+not",
    "thou\\s+shalt\\s+not",
    "shall\\s+not",
    "never\\s+\\w+", # keep broad; pairing with prototype averaging
    "cease\\s+(from|to)",
    "beware",
    "wo\\s*(e)?\\s*unto",
    "woe\\s+unto",
    "condemn",
    "rebellion\\s+against",
    "turn\\s+your\\s+backs",
    "must\\s+stop",
    "stop\\s+\\w+ing\\s+", # weak — prototype averaging reduces noise
    sep = "|"
  ),
  ignore_case = TRUE
)

gentle_rx <- regex(
  paste(
    "i invite",
    "we invite",
    "invitation",
    "come\\s+unto\\s+christ",
    "jesus\\s+christ",
    "our\\s+savior",
    "redeemer",
    "his\\s+love",
    "peace\\s+in\\s+christ",
    "gentle",
    "tender",
    "mercies",
    "consider\\s+how",
    "ponder",
    sep = "|"
  ),
  ignore_case = TRUE
)

# ---- Main ------------------------------------------------------------------------------
max_glove_lines <- as.integer(Sys.getenv("CONFERENCESTATS_GLOVE_NROWS", unset = "0"))
if (max_glove_lines <= 0L) {
  max_glove_lines <- NULL
}

para <- gc_as_paragraph_tibble_generalconference()
talks <- gc_normalize_talks(gc_as_tibble_generalconference())

para <- para |>
  inner_join(
    talks |> select(talk_id, year, month, conference_date, speaker, title, url, word_count),
    by = c("year", "month", "speaker", "title", "url")
  ) |>
  filter(map_lgl(.data$speaker, is_apostolic_or_fp_talk))

message(
  "Paragraphs after apostolic/FP filter: ",
  nrow(para),
  " | years ",
  min(para$year),
  "–",
  max(para$year)
)

para_w <- para |>
  mutate(
    words = map(.data$text, glove_tokenize),
    wins = map(
      .data$words,
      windowize_words,
      size = 200L,
      stride = 120L,
      min_len = 15L
    )
  )

chunk_tbl <- purrr::pmap_dfr(
  list(
    para_w$talk_id,
    para_w$year,
    para_w$month,
    para_w$conference_date,
    para_w$speaker,
    para_w$title,
    para_w$wins
  ),
  function(tid, yr, mo, cdate, spk, ttl, wins) {
    if (!length(wins)) {
      return(NULL)
    }
    dplyr::bind_rows(lapply(wins, function(ch) {
      tibble::tibble(
        talk_id = tid,
        year = yr,
        month = mo,
        conference_date = cdate,
        speaker = spk,
        title = ttl,
        chunk_tokens = list(ch)
      )
    }))
  }
)

chunk_tokens_list <- lapply(seq_len(nrow(chunk_tbl)), function(i) chunk_tbl$chunk_tokens[[i]])

message("Total word windows (chunks): ", length(chunk_tokens_list))
idf_vec <- chunk_document_frequency(chunk_tokens_list)

E <- ensure_glove_100d(max_vocab_lines = max_glove_lines)

chunk_emb <- t(vapply(chunk_tokens_list, function(tok) {
  tfidf_pool_embedding(tok, E, idf_vec)
}, numeric(ncol(E))))

ok_chunk <- rowSums(is.na(chunk_emb)) == 0L
chunk_tbl <- chunk_tbl[ok_chunk, , drop = FALSE]
chunk_emb <- chunk_emb[ok_chunk, , drop = FALSE]

chunk_tbl <- chunk_tbl |>
  mutate(
    chunk_text = map_chr(.data$chunk_tokens, ~ paste(.x, collapse = " ")),
    has_presc = str_detect(.data$chunk_text, presc_rx),
    has_gentle = str_detect(.data$chunk_text, gentle_rx)
  )

u_presc <- centroid_l2(chunk_emb[chunk_tbl$has_presc, , drop = FALSE])
u_gent <- centroid_l2(chunk_emb[chunk_tbl$has_gentle, , drop = FALSE])

if (any(is.na(u_presc)) || any(is.na(u_gent))) {
  stop("Could not build both prototypes — relax regex or increase data.")
}

axis <- u_presc - u_gent
axis <- axis / sqrt(sum(axis * axis))

score_chunks <- as.numeric(chunk_emb %*% matrix(axis, ncol = 1))

# cosine distances to each pole (unit vectors)
s_presc <- as.numeric(chunk_emb %*% u_presc)
s_gent <- as.numeric(chunk_emb %*% u_gent)
chunk_tbl <- chunk_tbl |>
  mutate(
    axis_score = score_chunks,
    cos_presc = s_presc,
    cos_gentle = s_gent,
    net_presc = .data$cos_presc - .data$cos_gentle
  )

talk_scores <- chunk_tbl |>
  group_by(
    .data$talk_id,
    .data$year,
    .data$month,
    .data$conference_date,
    .data$speaker
  ) |>
  summarise(
    n_chunks = dplyr::n(),
    mean_axis = mean(.data$axis_score),
    mean_net_presc = mean(.data$net_presc),
    q75_net = stats::quantile(.data$net_presc, 0.75),
    .groups = "drop"
  )

message("\nGAM: mean_net_presc ~ s(year)")
m <- mgcv::gam(mean_net_presc ~ s(year), data = talk_scores, method = "REML")
message(
  "edf=",
  round(sum(m$edf[-1]), 2),
  " ref.df=",
  round(sum(m$edf[-1]), 2),
  " p=",
  format.pval(summary(m)$s.table[1, 4])
)

outdir <- file.path(pkg_root, "analysis", "output")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

p <- ggplot(talk_scores, aes(x = year, y = mean_net_presc)) +
  geom_hline(yintercept = 0, linetype = 3, alpha = 0.45) +
  geom_point(aes(size = n_chunks), alpha = 0.22, color = "#1d4e89") +
  geom_smooth(
    method = "gam",
    formula = y ~ s(x, k = 10),
    se = TRUE,
    color = "#b03a48",
    fill = "#b03a4844"
  ) +
  scale_size(range = c(0.6, 3.2), guide = "none") +
  labs(
    title = "Apostolic & First Presidency talks: prescriptive vs gentle prototype (GloVe)",
    subtitle = "Per chunk: TF-IDF-weighted mean GloVe 100d; net = cos(chunk,u_presc)-cos(chunk,u_gentle); u_* from regex-mined chunks.",
    x = "Conference year",
    y = "Mean chunk score (net prescriptive alignment)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(outdir, "apostolic_glove_prototype_trend.png"), p, width = 10, height = 5.5, dpi = 150)
message("Saved: ", file.path(outdir, "apostolic_glove_prototype_trend.png"))

decade <- talk_scores |>
  mutate(decade = floor(.data$year / 10) * 10) |>
  group_by(.data$decade) |>
  summarise(
    n_talks = dplyr::n(),
    mean_net = mean(.data$mean_net_presc),
    se = stats::sd(.data$mean_net_presc) / sqrt(dplyr::n()),
    .groups = "drop"
  )
message("\nPer-decade talk-level means (net prescriptive):\n")
print(decade, n = 50)
