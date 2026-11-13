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
- **Prediction** The CNN + flat head beats logistic regression on features.csv by at least 5 points of macro mAP.
- **Committed in:** 9fce93f
- **Result:**
- **Prediction** (Maddi Smith, 11/11):
 Logistic regression on features.csv will beat the CNN +
  flat head by at least 5 points of macro mAP. features.csv contains hand-designed sound
  statistics, while our CNN is trained from scratch and has to learn its own features from the
  spectrograms, which I expect to be hard with ~20K clips and 97 overlapping genres.
- **Committed in:** _(commit hash)_
- **Prediction** (Ayesha, 11/13): The CNN + flat head beats logistic regression on features.csv
  by at least 5 points of macro mAP.
- **Committed in:** 2cf04c9
- **Result:** 
- **Verdict:** supported / not supported / mixed

## E1: Flat head vs. GCN head (hierarchy edges), 3 seeds each

- **Runs:** 6
- **Metrics:** macro mAP overall, top-level, fine-grained, rare, common; micro-F1;
  hierarchy-violation rate
- **Prediction** The GCN head gains 2-4 points of macro mAP over the flat head, mostly on genres with fewer than 200 training tracks. Top-level genres barely change.
- **Committed in:** 9fce93f
- **Prediction** (Maddi, 11/12): To be honest, I'm not sure that giving the model the genre
  tree will help for all genre types, especially some of the more niche ones. For the super
  mainstream genres like Rock (the most present in the FMA dataset), I don't think the tree will
  add or change very much, so I'm guessing the GCN and flat head will be within 1 point of macro
  mAP on common genres. For the rare, fine-grained genres, mixing information between relatives
  may blur similar subgenres together and could possibly make the GCN perform worse, so I'm going
  to guess the GCN will be around 6 points of macro mAP lower than the flat head on rare genres,
  and about the same or slightly lower overall. However, I think the GCN's hierarchy violation
  rate will be lower, because the tree connects each subgenre to its parent, so when the GCN
  predicts a subgenre it should also predict the parent.
- **Committed in:**
- **Prediction** (Ayesha, 11/13): The GCN head gains 2–4 points of macro mAP over the flat head,
  mostly on rare genres (fewer than 200 training tracks); top-level genres barely change. The flat
  head will have the lower hierarchy-violation rate.
- **Committed in:** 2cf04c9
- **Result:**
- **Verdict:**

## E2: GCN with hierarchy edges vs. hierarchy + co-occurrence edges (training split only)

- **Runs:** 3
- **Metrics:** macro mAP (overall and by group), hierarchy-violation rate vs. the E1 flat head
- **Prediction** (Maddi, 11/12): I actually think the GCN with the tree plus genres often tagged
  together might score lower than the tree-only GCN from E1. Mixing information between relatives
  and subgenres, and adding co-occurrence edges, could be problematic because similar genres might
  blur together like I predicted in E1. I think it could decrease macro mAP by 2–3 points (a smaller
  change than my E1 prediction, but I'm not confident).
- **Committed in:**
- **Result:**
- **Verdict:**

## E3: GCN depth, 1 vs. 2 vs. 3 layers

- **Runs:** 3
- **Metrics:** macro mAP; over-smoothing, measured as the mean cosine similarity between
  sibling genres' detectors
- **Prediction** (Maddi, 11/12): I actually think fewer layers might be better because of my
  general thought that mixing genres causes more blurring/mixing up, so I think one layer will
  score best on macro mAP. As I continue to add layers, sibling genres will become more similar
  (higher cosine similarity), and at 3 layers I'm going to guess macro mAP will be about 4 points
  lower than with 1 layer.
- **Committed in:**
- **Result:**
- **Verdict:**

## E4: Recommendation prototype, nearest neighbours on embeddings

- **Runs:** none new (reuses the E0/E1 checkpoints)
- **Setup:** queries are test-split tracks; candidates are train + validation tracks (no shared
  artists). Embeddings compared: GCN model, flat model, features.csv, random.
- **Metric:** genre overlap@10 (mean Jaccard between the query's genres and each recommendation's
  genres), overall, for rare-genre queries, and for fine-grained queries
- **Prediction** GCN embeddings beat flat embeddings on genre overlap@10, mostly for rare-genre queries. Both beat features.csv, and all beat random.
- **Committed in:** 9fce93f
- **Prediction** (Maddi, 11/12): Based on my E0 and E1 predictions, I think the ranking by genre
  overlap@10 will be features.csv > flat > GCN > random. Since I predicted logistic regression on
  features.csv beats the CNN in E0, I expect features.csv vectors to find songs with more similar
  genres too. I expect the gap between flat and GCN to be largest for rare-genre and fine-grained
  queries, because if the GCN blurs similar subgenres together, its embeddings will recommend
  songs from the wrong subgenre more often. All three should clearly beat random.
- **Committed in:**
- **Prediction** (Ayesha, 11/13): Ranking by genre overlap@10: GCN > flat > features.csv > random.
- **Committed in:** 2cf04c9
- **Result:**
- **Verdict:**

---

## Diagnostics (not separate runs)

### Gradient norms per layer, flat vs. GCN

- **Prediction** (Maddi, 11/12): Under the GCN, rare genres' detectors will get larger gradients
  than under the flat head, because they also receive updates from their relatives' examples. But
  I think those gradients will also be less specific to each genre, since they mix in signals from
  other genres, which fits my guess that the GCN blurs similar subgenres together.
- **Result:**

### Encoder feature maps (block 2) on clips from three genres

- **What we expect to see** (Maddi, 11/12): I expect block 2 to pick up simple sound patterns:
  vertical stripes for drum hits in Hip-Hop and Electronic, horizontal lines for held notes in
  Classical, and messier patterns spread across many frequencies in Rock (distorted guitars).
- **Result:**

### Metadata audit: 30 most confident model–tag disagreements

Label each one as a model error, a debatable tag, or a likely wrong tag.

- **Expected split** (17) Miranda, Nov 11
- **Result:** 
- **Expected split** (Maddi, 11/12): About 15 model errors, 10 debatable tags, and 5 tags that look
  wrong. Since I expect our from-scratch CNN to be fairly weak (E0), I think most disagreements will
  be the model's mistakes, but FMA's genre tags are chosen by the artists themselves, so some will
  be debatable or simply wrong.
- **Result:**
