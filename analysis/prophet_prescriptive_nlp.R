# Prophet (President-of-the-Church) talks: prescriptive vs Christ-centered language over time.
# Methods:
#   1) TF-IDF "semantic axis" — projection onto admonition-vs-invitation seed directions (VSM, not phrase grep).
#   2) LDA document-topic mix — topics scored by overlap with seed lexicons; prevalence regressed on year.
#
# Requires: generalconference v0.3.2 (GitHub tag; main DESCRIPTION may be invalid on R >= 4.5),
#   conferencestats sources, quanteda, topicmodels, irlba, ggplot2, mgcv, gsl (for topicmodels build).

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(purrr)
  library(Matrix)
  library(quanteda)
  library(topicmodels)
  library(irlba)
  library(ggplot2)
  library(mgcv)
})

pkg_root <- Sys.getenv("CONFERENCESTATS_ROOT", unset = normalizePath(".."))
if (!dir.exists(file.path(pkg_root, "R"))) {
  stop("Set CONFERENCESTATS_ROOT to the package root (parent of analysis/).")
}
pkgload::load_all(pkg_root, quiet = TRUE)

# ---- 1. Load & filter Church President talks (narrow "prophet" sense) -----------------
if (!requireNamespace("generalconference", quietly = TRUE)) {
  stop("Install generalconference (e.g. remotes::install_github('bryanwhiting/generalconference@v0.3.2'))")
}

talks <- gc_normalize_talks(gc_as_tibble_generalconference())

# Exclude auxiliary "President" roles; exclude Acting President of the Twelve (not Church President)
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
  # Spencer W. Kimball only when not "Acting President of the Council of the Twelve"
  "Spencer W\\. Kimball",
  sep = "|"
)

is_president_talk <- function(author1) {
  a <- str_squish(str_remove(author1, "^By\\s+"))
  if (str_detect(a, "Acting President")) {
    return(FALSE)
  }
  if (str_detect(a, "General President|Young Women|Young Men|Relief Society|Sunday School")) {
    return(FALSE)
  }
  str_detect(a, regex(pres_pat))
}

prophet <- talks %>%
  filter(map_lgl(.data$speaker, is_president_talk)) %>%
  mutate(decade = floor(.data$year / 10) * 10)

message("President-of-the-Church talks: ", nrow(prophet), " (years ", min(prophet$year), "–", max(prophet$year), ")")

if (nrow(prophet) < 50L) {
  stop("Very few president talks matched — check name patterns.")
}

# ---- 2. Quanteda DFM + TF-IDF --------------------------------------------------------
corp <- corpus(
  prophet,
  text_field = "text",
  docid_field = "talk_id"
)

toks <- tokens(corp, remove_punct = TRUE, remove_numbers = FALSE, remove_separators = TRUE) %>%
  tokens_remove(stopwords("english")) %>%
  tokens_wordstem(language = "en")

dfm <- dfm(toks) %>%
  dfm_trim(
    min_docfreq = 0.008,
    max_docfreq = 0.35,
    docfreq_type = "prop"
  )

dfm_t <- dfm_tfidf(dfm)
X <- as(dfm_t, "CsparseMatrix")
feat <- colnames(dfm_t)

# ---- 3. Seed semantic axis in TF-IDF space (Kozlowski-style, sparse VSM) ------------
# Seeds are *grouped stems* after same stemmer as DFM (Snowball en)
seed_admon <- c(
  "command", "commandment", "forbid", "forbidden", "must", "warn", "sin", "wicked",
  "rebelli", "disobedi", "ceas", "refus", "avoid", "never", "repent", "judgment",
  "condemn", "woe", "destruct", "pervers", "adulteri", "whoremong"
)
seed_christ <- c(
  "jesus", "christ", "savior", "redeem", "aton", "merci", "grace", "love", "chariti",
  "invit", "peace", "comfort", "heal", "tender", "gentl", "hope", "joy", "prais",
  "thank", "humbl", "servant", "shepherd", "light", "spirit", "coven", "ordin"
)
seed_invite <- c(
  "invit", "consider", "ponder", "encourag", "strive", "descript", "learn", "understand",
  "discov", "feel", "experi", "journey", "path", "grow"
)

pick_feats <- function(stems) {
  sort(unique(intersect(feat, stems)))
}

f_adm <- pick_feats(seed_admon)
f_chr <- pick_feats(seed_christ)
f_inv <- pick_feats(seed_invite)
message("Seed feats matched — admon: ", length(f_adm), ", christ: ", length(f_chr), ", invite: ", length(f_inv))

idx <- function(terms) match(terms, feat)
axis_admon <- sparseMatrix(
  i = idx(f_adm),
  j = rep(1L, length(f_adm)),
  x = rep(1 / length(f_adm), length(f_adm)),
  dims = c(length(feat), 1L)
)
axis_christ <- sparseMatrix(
  i = idx(f_chr),
  j = rep(1L, length(f_chr)),
  x = rep(1 / length(f_chr), length(f_chr)),
  dims = c(length(feat), 1L)
)
axis_inv <- sparseMatrix(
  i = idx(f_inv),
  j = rep(1L, length(f_inv)),
  x = rep(1 / length(f_inv), length(f_inv)),
  dims = c(length(feat), 1L)
)

# Higher score ⇒ more mass on admonition / prohibition pole vs Christ-centered + invitational pole
v <- as.numeric(axis_admon - 0.5 * axis_christ - 0.5 * axis_inv)
v <- v / sqrt(sum(v * v))
axis_sparse <- Matrix::sparseMatrix(
  i = seq_along(feat),
  j = rep(1L, length(feat)),
  x = v,
  dims = c(length(feat), 1L)
)
tfidf_axis_score <- as.numeric(X %*% axis_sparse)

# ---- 4. LSA compression + smooth trajectory (unsupervised geometry) ------------------
# Truncated SVD on TF-IDF for dominant axes; project seed axis into this space for stability check.
set.seed(42)
nv <- min(40L, min(dim(X)) - 1L)
if (nv < 5L) stop("DFM too small for LSA.")
ls <- irlba::irlba(X, nv = nv, maxit = 500L, work = nv + 7L)
doc_lsa <- as.matrix(X %*% ls$v %*% diag(1 / ls$d)) # docs in latent space (roughly)

# ---- 5. LDA: topic mixtures scored vs seeds -----------------------------------------
K <- 12L
set.seed(42)
dtm_tm <- quanteda::convert(dfm, to = "topicmodels")
lda <- LDA(dtm_tm, k = K, control = list(seed = 42, verbose = 0))
g <- topicmodels::posterior(lda)$topics

# Topic-term distribution beta (topics x terms in topicmodels order)
beta_nt <- lda@beta
colnames(beta_nt) <- lda@terms
adm_i <- match(f_adm, colnames(beta_nt))
adm_i <- adm_i[!is.na(adm_i)]
chr_i <- match(f_chr, colnames(beta_nt))
chr_i <- chr_i[!is.na(chr_i)]
lda_admon_loading <- apply(beta_nt, 1L, function(row) {
  sum(row[adm_i]) / (sum(row[adm_i]) + sum(row[chr_i]) + 1e-9)
})
message(
  "LDA topics P(seed admon|seed admon+christ): ",
  paste(round(lda_admon_loading, 2), collapse = ", ")
)

lda_admon_prop <- as.numeric(g %*% matrix(lda_admon_loading, ncol = 1))

# ---- 6. Combine (z-score components) for a single index ------------------------------
z1 <- as.numeric(scale(tfidf_axis_score))
z2 <- as.numeric(scale(lda_admon_prop))
composite <- as.numeric(scale(0.55 * z1 + 0.45 * z2))

plot_df <- prophet %>%
  mutate(
    tfidf_admon_axis = tfidf_axis_score,
    lda_admonstance = lda_admon_prop,
    composite_prescriptive = composite
  )

# ---- 7. GAM on year ------------------------------------------------------------------
m <- mgcv::gam(composite_prescriptive ~ s(year), data = plot_df, method = "REML")
message("\nGAM composite ~ s(year): edf=", round(summary(m)$edf, 2), ", p=", format.pval(summary(m)$s.pv))

pd <- plot_df %>%
  mutate(
    fit = predict(m),
    # pointwise ~95% from predict; approximate with gam helper
    .pred = NA
  )
p <- ggplot(plot_df, aes(x = year, y = composite_prescriptive)) +
  geom_hline(yintercept = 0, linetype = 3, alpha = 0.5) +
  geom_point(aes(size = word_count), alpha = 0.35, color = "#2c5c7c") +
  geom_smooth(method = "gam", formula = y ~ s(x, k = 8), se = TRUE, color = "#c2554a", fill = "#c2554a33") +
  scale_size(range = c(0.5, 3), guide = "none") +
  labs(
    title = "Church president talks: composite prescriptive / admonishing index over time",
    subtitle = "TF-IDF seed axis + LDA topic stance (z-mix). Higher ⇒ more prohibition/command-weight vs Christ/invitational.",
    x = "Conference year",
    y = "Composite index (standardized)"
  ) +
  theme_minimal(base_size = 12)

outdir <- file.path(pkg_root, "analysis", "output")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
ggsave(file.path(outdir, "prophet_prescriptive_trend.png"), p, width = 9, height = 5.5, dpi = 150)
message("Saved plot: ", file.path(outdir, "prophet_prescriptive_trend.png"))

# Per-decade summary
summ <- plot_df %>%
  group_by(decade) %>%
  summarise(
    n = dplyr::n(),
    mean_comp = mean(composite_prescriptive),
    se = stats::sd(composite_prescriptive) / sqrt(n),
    .groups = "drop"
  )
message("\nPer-decade mean composite:\n")
print(summ, n = 100)
