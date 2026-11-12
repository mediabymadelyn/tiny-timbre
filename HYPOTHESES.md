# Hypotheses

Write each prediction **before** its experiment runs, then commit it. The commit timestamp is the
proof that the prediction came first. Fill in **Result** and **Verdict** afterwards, in a separate
commit. Never edit a prediction after its experiment has started; if you change your mind, add a
dated note underneath instead.

A null or wrong result still counts, as long as the prediction was committed first.

**Shared facts** (from `labels.py` on the real data, 50-track filter):
C = 97 genres: 15 top-level, 82 fine-grained. Rare = fewer than 200 training tracks (51 genres);
common = 200 or more (46). Split: 19,922 train / 2,505 validation / 2,573 test, artist-filtered.

**Held fixed across all runs:** data split, number of epochs, optimizer, 10 s random crops,
seed policy. Only the row's own variable changes.

---

## E0: Logistic regression on features.csv vs. CNN + flat head

- **Runs:** 1 + 1
- **Metric:** macro mAP on the test split
- **Prediction** (Maddi Smith, 11/11):
 Logistic regression on features.csv will beat the CNN +
  flat head by at least 5 points of macro mAP. features.csv contains hand-designed sound
  statistics, while our CNN is trained from scratch and has to learn its own features from the
  spectrograms, which I expect to be hard with ~20K clips and 97 overlapping genres.
- **Committed in:** _(commit hash)_
- **Result:** 
- **Verdict:** supported / not supported / mixed

## E1: Flat head vs. GCN head (hierarchy edges), 3 seeds each

- **Runs:** 6
- **Metrics:** macro mAP overall, top-level, fine-grained, rare, common; micro-F1;
  hierarchy-violation rate
- **Prediction** (author, date):
- **Committed in:**
- **Result:**
- **Verdict:**

## E2: GCN with hierarchy edges vs. hierarchy + co-occurrence edges (training split only)

- **Runs:** 3
- **Metrics:** macro mAP (overall and by group), hierarchy-violation rate vs. the E1 flat head
- **Prediction** (author, date):
- **Committed in:**
- **Result:**
- **Verdict:**

## E3: GCN depth, 1 vs. 2 vs. 3 layers

- **Runs:** 3
- **Metrics:** macro mAP; over-smoothing, measured as the mean cosine similarity between
  sibling genres' detectors
- **Prediction** (author, date):
- **Committed in:**
- **Result:**
- **Verdict:**

## E4: Recommendation prototype, nearest neighbours on embeddings

- **Runs:** none new (reuses the E0/E1 checkpoints)
- **Setup:** queries are test-split tracks; candidates are train + validation tracks (no shared
  artists). Embeddings compared: GCN model, flat model, features.csv, random.
- **Metric:** genre overlap@10 (mean Jaccard between the query's genres and each recommendation's
  genres), overall, for rare-genre queries, and for fine-grained queries
- **Prediction** (author, date):
- **Committed in:**
- **Result:**
- **Verdict:**

---

## Diagnostics (not separate runs)

### Gradient norms per layer, flat vs. GCN

- **Prediction** (author, date):
- **Result:**

### Encoder feature maps (block 2) on clips from three genres

- **What we expect to see** (author, date):
- **Result:**

### Metadata audit: 30 most confident model–tag disagreements

Label each one as a model error, a debatable tag, or a likely wrong tag.

- **Expected split** (author, date):
- **Result:**
