# Methods and statistical inference — General Conference semantic explorer

This note describes **what the pipeline computes**, **what the Shiny app reports**, and **how statistically sound those summaries are** when read as exploratory evidence (not as causal or ecclesiastical claims).

---

## 1. Purpose and scope

The project measures **regularities in wording** across General Conference talks using a fixed **sentence embedding model** and **pre-chosen semantic cues** (“poles”). Outputs include:

- A **main prescriptive vs invitational score** per talk (regex-mined exemplar sentences define two poles; each chunk is scored by cosine similarity to those poles).
- **User-defined “custom poles”**: any short phrase embedded in the **same space** as chunk text, producing per-talk **mean cosine** to that phrase direction.
- **Two-phrase contrast**: per-talk scores for phrase **A** and **B**; the app emphasizes **A − B** (relative alignment in embedding space).

Everything below assumes a **frozen** model (`BAAI/bge-small-en-v1.5` in the bundled pipeline), **semantic chunking** aligned to that model’s tokenizer, and **tf–idf–style pooling** over subword representations as implemented in the Python driver (`gc_chunk_embed_pipeline.py` and `embed_query_phrase.py`).

---

## 2. What embeddings measure here

**Cosine similarity** between vectors answers: “Do two texts sit in a similar region of the model’s learned representation?” The model was **not** trained on LDS theology as a supervised task; it is a **general English** encoder. Therefore:

- Scores are **geometric**, not moral or doctrinal verdicts.
- Similarity to a **hand-typed phrase** is sensitivity to whatever lexical and shallow-semantic patterns the model associates with that wording in **this** subword space.
- Rare or offensive strings can produce unstable or misleading directions; **interpretation requires reading exemplar talks and chunks**, not only a trend line.

---

## 3. Talk-level scores

### 3.1 Main “net prescriptive” score (Explore tab)

For each semantic chunk in a talk:

- Cosine to the **prescriptive pole** (average of regex-hit sentences) minus cosine to the **gentle / invitational pole**.

The **talk score** is the **unweighted mean** across chunks (chunk count varies by talk length).

**Soundness:** The construction is transparent and reproducible given the same regex bundles and model. **Validity** of “prescriptive” vs “gentle” is **operational** — it is exactly what the regex exemplars evoke in embedding space, not a sociolinguistic gold standard.

### 3.2 Custom pole (single phrase)

For a user phrase **u** (unit-normalized), each talk’s stored **sum of chunk embedding vectors** \(S\) and chunk count \(n\) yield:

\[
\text{mean\_cos} = \frac{S^\top u}{n}
\]

(i.e. average cosine alignment of chunks with **u** under the same pooling as the pipeline).

**Soundness:** Algebra is consistent with “average of per-chunk cosines” when using summed embeddings and the same normalization conventions as the pipeline implementation.

### 3.3 Two-phrase contrast

For phrases **A** and **B** with unit vectors \(u_A, u_B\):

\[
\text{contrast} = \text{mean\_cos}(u_A) - \text{mean\_cos}(u_B)
\]

per talk. This is a **difference of mean cosines**, not a cosine of a difference vector. It answers a **comparative** question: “Which direction does this talk’s chunk cloud lean toward, **A** or **B**, in this geometry?”

### 3.4 Phrase-aligned exemplar passages (Showpiece / Contrast)

When **`chunks_scored.parquet`** and phrase vectors \(u_A, u_B\) are available, the app can show one **quoted chunk per exemplar talk** chosen **inside that talk** by **argmax** (toward A) or **argmin** (toward B) of \(\cos(z, u_A) - \cos(z, u_B)\) over chunk embeddings \(z\), using the **same model and pooling** as the pipeline (`best_contrast_chunks.py`). This is **not** the same rule as “swing” highlights in the Chunk insights tab (those use precomputed net prescriptive vs invitational scores). If phrase-aligned data are missing, the UI falls back to swing excerpts.

---

## 4. Graphics: OLS and GAM on year

### 4.1 Ordinary least squares (OLS)

Where the app draws an **OLS line** (and shaded interval), it fits a **linear regression** of the plotted **y** on **calendar year** using only talks in the **current filter** (year range, and era filters where applicable). The ribbon is a **nominal 95% confidence band** for the **conditional mean** from `predict(..., interval = "confidence")` in R, not a simultaneous band over the whole curve and not a prediction interval for new talks.

**Interpretation:** Slope summarizes **linear** association with time in the window. It can be dominated by a few years or speakers; it is **not** a causal effect of “the year” on language.

### 4.2 Generalized additive model (GAM)

On the Explore scatter, a second layer fits `mgcv::gam(y ~ s(year, k = ...))` (**REML**) on the same filtered points and predicts a smooth over a dense year grid. The shaded band uses **±1.96 × standard error of the smooth** from the model’s `predict(..., se.fit = TRUE)`.

**Interpretation:** The smooth allows **nonlinear** drift. Agreement between OLS and GAM suggests a roughly linear trend; divergence suggests curvature, outliers, or era-specific behavior.

### 4.3 Plotly and layers

OLS + GAM curves are built as explicit `geom_line` / `geom_ribbon` layers (not only `geom_smooth`) so fits remain visible after conversion to **plotly**.

---

## 5. Significance tests surfaced in the app

Different panels report different tests; they **answer different questions** and share common limitations (Section 6).

### 5.1 Pearson correlation

**Statistic:** Pearson \(r\) between **year** and talk-level **y** (mean cosine, net score, or contrast).

**Null hypothesis (informal):** No linear association between year and \(y\) **if** years and scores were drawn in a way that makes this test appropriate.

**Limitations:** Talks are **not** independent draws: same speakers, series, and thematic years cluster. **n** is large enough that trivially small effects can yield tiny *p*-values. **Correlation ≠ causation** and does not separate vocabulary change from topic mix change.

### 5.2 GAM smooth term *p*-value

From `mgcv`, the reported *p*-value for the smooth term is a **default frequentist test** that the smooth is needed relative to a flat model **under the GAM specification**. It is sensitive to **k** (basis dimension), **REML**, and residual assumptions.

Use it as **heuristic evidence of non-flat drift**, not as a standalone proof of “real-world importance.”

### 5.3 Welch two-sample *t*-test (early vs late)

The app compares distributions of \(y\) **before vs on/after a split year** (median year on some cards; user-chosen split on the contrast tab).

**Design:** Unweighted by default — each talk counts once.

**Limitations:** Choice of split is somewhat arbitrary; **multiple splits** or many outcomes inflate false positives unless explicitly adjusted. Clustering by speaker again violates the **independence** idealization.

### 5.4 Chunk-weighted OLS on a late indicator

Some cards fit `lm(y ~ I(year >= split), weights = n_chunks)`.

**Intent:** Up-weight longer talks (more chunks) when summarizing a mean shift between halves of the timeline.

**Caveat:** This is **not** a principled hierarchical model of chunks nested in talks. It is an **exploratory weighting scheme**: weights echo “how much text contributed to the talk score,” but standard errors still assume the **linear model’s** error structure at the **talk** level. Treat *p*-values as **directional**, not confirmatory.

---

## 6. Threats to validity (shared across analyses)

1. **Non-independence:** Speaker identity, multi-year assignments, and recurring themes induce **serial dependence**. Time-series or mixed models would be more appropriate for **rigorous** inference; the app stays simple for transparency.

2. **Confounding by topic:** Year correlates with **what is preached about**. A rising cosine to a phrase may reflect **subject matter** moving toward domains where that phrase is natural, not a uniform rhetorical drift.

3. **Multiple comparisons:** Trying many phrases, splits, or filters will produce “significant” patterns by chance unless **pre-registration** or **multiplicity control** is used.

4. **Embedding partiality:** Different models and pooling choices change numbers. Results are **pipeline-relative**.

5. **Selection:** The public corpus definition (years included, language, transcription) defines the population; conclusions generalize to **that** corpus artifact.

---

## 7. How to report results responsibly

- Lead with **effect size and uncertainty**: ranges of \(y\), visually apparent slope or gap, not only *p*.
- Show **raw scatter** (the app does) and mention **exemplar talks** qualitatively.
- Pair **A − B contrast** claims with what **A** and **B** literally were.
- Prefer language like **“associated with,” “leans toward in embedding space,”** rather than **“became more X.”**
- Where stakes are high (esp. sensitive terms), add **human readership** of passages and, if needed, **institutional ethics** review.

---

## 8. Relation to code artifacts

| Concept | Primary implementation |
|--------|-------------------------|
| Chunking, poles, Parquet outputs | `analysis/python/gc_chunk_embed_pipeline.py` |
| Phrase embedding for Shiny | `analysis/python/embed_query_phrase.py` |
| Phrase-aligned exemplar chunks | `analysis/python/best_contrast_chunks.py` |
| JSONL → talk Parquet (optional corpus) | `analysis/python/jsonl_to_talks_parquet.py` |
| Figures + RDS sync | `analysis/plot_gc_chunk_embed_results.R` |
| Explorer, custom pole, contrast UI | `analysis/shiny_gc_family/app.R` |

Regenerating `talk_emb_sums.rds`, `subword_idf.npy`, and related files after pipeline changes keeps **custom** and **contrast** tabs aligned with the frozen pooling and IDF used offline.

---

## 9. Summary

The pipeline is **internally coherent** and **reproducible** under fixed choices (model, chunking, poles, phrases). Classical *p*-values in the app are best read as **exploratory alarms**: they flag consistency with simple models more than they certify **independent, causal, or ecclesiastical** claims. For stronger inference, extend toward **dependent-data models**, **topic controls**, and **multiplicity-aware** design — while keeping the current interface useful for transparent, exploratory analysis.
