# Hypotheses

Write each prediction **before** its experiment runs, then commit it. The commit timestamp is the
proof that the prediction came first. Fill in **Result** and **Verdict** afterwards, in a separate
commit. Never edit a prediction after its experiment has started; if you change your mind, add a
dated note underneath instead.

A null or wrong result still counts, as long as the prediction was committed first.

_Layout reorganized on Nov 22, 2026 (each experiment now has Predictions → Result → Verdicts).
The prediction texts are unchanged; compare with commit `83ee7e3` to check._

**Shared facts** (from `labels.py` on the real data, 50-track filter):
C = 97 genres: 15 top-level, 82 fine-grained. Rare = fewer than 200 training tracks (51 genres);
common = 200 or more (46). Split: 19,922 train / 2,505 validation / 2,573 test, artist-filtered.

**Held fixed across all runs:** data split, number of epochs, optimizer, 10 s random crops,
seed policy. Only the row's own variable changes.

**How results were computed:** test split, best-on-validation epoch of each run, scored with
`baseline.score` through `analyze.results_table` (the CNN and logistic regression are scored the
same way). Mean ± std is over seeds 0, 1, 2. All CNN runs: 20 epochs on a Colab T4 GPU.

**Verdict options:** supported / not supported / mixed, plus one sentence on why.

---

## E0: Logistic regression on features.csv vs. CNN + flat head

- **Runs:** 1 + 1 (`e0_baseline`, `e0_cnn_flat`)
- **Metric:** macro mAP on the test split

### Predictions

- **Miranda** (Nov 11, 2026; committed in `9fce93f`): The CNN + flat head beats logistic regression on features.csv by at least 5 points of macro mAP.
- **Maddi** (11/11; committed in `598dbce`):
 Logistic regression on features.csv will beat the CNN +
  flat head by at least 5 points of macro mAP. features.csv contains hand-designed sound
  statistics, while our CNN is trained from scratch and has to learn its own features from the
  spectrograms, which I expect to be hard with ~20K clips and 97 overlapping genres.
- **Ayesha** (11/13; committed in `8522a9d`): The CNN + flat head beats logistic regression on features.csv
  by at least 5 points of macro mAP.

### Result

The CNN actually won by 4.3 points: CNN + flat (0.207) vs (0.164).

| Test split | macro mAP | top-level | fine-grained | rare | common | micro-F1 | hierarchy violations ↓ |
|---|---|---|---|---|---|---|---|
| Logistic regression on features.csv | 0.164 | 0.436 | 0.113 | 0.074 | 0.260 | 0.396 | 0.022 |
| CNN + flat head (`e0_cnn_flat`) | **0.207** | **0.475** | **0.157** | **0.112** | **0.309** | **0.434** | **0.007** |

Logistic regression: Ayesha, 11/15; `configs/e0_baseline.yaml`, seed 0, C = 0.003 picked on
validation, 15 failed clips dropped; 19,909 train / 2,572 test. The CNN is better on every metric.

### Verdicts

<!-- Miranda / Ayesha: right direction (CNN wins), but by 4.3 points, not 5+. Maddi: predicted the opposite. -->
- **Miranda:** Mixed. The CNN beat logistic regression, as I predicted, but by 4.3 points, not 5+.
- **Maddi:** I originally predicted that the logistic regression would win by 5 points, but the cnn actually won by 4.3 points and was better on pretty much every metric. Learning features from the spectrograms worked a lot better than I expected with 20k music snippets.
- **Ayesha:**

---

## E1: Flat head vs. GCN head (hierarchy edges), 3 seeds each

- **Runs:** 6 (`e0_cnn_flat`, `e1_flat_s1`, `e1_flat_s2`; `e1_gcn_s0`, `e1_gcn_s1`, `e1_gcn_s2`)
- **Metrics:** macro mAP overall, top-level, fine-grained, rare, common; micro-F1;
  hierarchy-violation rate

### Predictions

- **Miranda** (Nov 11, 2026; committed in `9fce93f`): The GCN head gains 2-4 points of macro mAP over the flat head, mostly on genres with fewer than 200 training tracks. Top-level genres barely change.
- **Maddi** (11/12; committed in `aa41799`): To be honest, I'm not sure that giving the model the genre
  tree will help for all genre types, especially some of the more niche ones. For the super
  mainstream genres like Rock (the most present in the FMA dataset), I don't think the tree will
  add or change very much, so I'm guessing the GCN and flat head will be within 1 point of macro
  mAP on common genres. For the rare, fine-grained genres, mixing information between relatives
  may blur similar subgenres together and could possibly make the GCN perform worse, so I'm going
  to guess the GCN will be around 6 points of macro mAP lower than the flat head on rare genres,
  and about the same or slightly lower overall. However, I think the GCN's hierarchy violation
  rate will be lower, because the tree connects each subgenre to its parent, so when the GCN
  predicts a subgenre it should also predict the parent.
- **Ayesha** (11/13; committed in `8522a9d`): The GCN head gains 2–4 points of macro mAP over the flat head,
  mostly on rare genres (fewer than 200 training tracks); top-level genres barely change. The flat
  head will have the lower hierarchy-violation rate.

### Result

| Test split, mean ± std over 3 seeds | Flat head | GCN head | GCN − flat |
|---|---|---|---|
| macro mAP, all genres | **0.207 ± 0.002** | 0.187 ± 0.004 | −2.0 points |
| rare | **0.112 ± 0.005** | 0.087 ± 0.003 | −2.5 |
| common | **0.307 ± 0.001** | 0.294 ± 0.006 | −1.3 |
| top-level | 0.477 ± 0.002 | 0.481 ± 0.014 | +0.4 (tie) |
| fine-grained | **0.156 ± 0.003** | 0.132 ± 0.004 | −2.4 |
| micro-F1 | 0.439 ± 0.006 | 0.443 ± 0.030 | +0.4 (tie) |
| hierarchy violations ↓ | 0.006 ± 0.002 | 0.007 ± 0.009 | tie (both < 1%) |

Per seed, macro mAP: flat 0.207 / 0.204 / 0.209; GCN 0.185 / 0.192 / 0.185. Every flat seed beats
every GCN seed.

Not uniform across genres (AP per genre, 3-seed mean): the GCN beats flat on 25 of 97 genres.
Biggest GCN gains: Classical +9.2 points (0.711 vs 0.619), Blues +6.3, Spoken +3.3 (Blues and
Spoken have only 8 and 12 test songs). Biggest losses: International −3.0, Country −2.9, Pop −2.8.

Why (see E3): the GCN makes sibling genres' detectors very similar (cosine 0.81–0.82 vs 0.37–0.39
for flat), so it struggles to tell subgenres apart.

### Verdicts

<!-- Miranda / Ayesha: GCN lost 2 points instead of gaining 2-4, and lost most on rare genres; top-level did barely change.
     Maddi: right direction on rare (−2.5, not −6) and overall (slightly lower); common differs by 1.3 (not within 1); violations equal, not lower.
     Ayesha's violation part: equal, not lower for flat. -->
- **Miranda:** I predicted a 2-4 point GCN gain on rare genres; it lost about 2 overall and most on rare genres. Top-level genres barely changed, as I predicted.
- **Maddi:** I was right that the GCN would do worse on rare genres (-2.5 points though, not 6) and slightly worse overall (-2.0). E3;s over smoothing result backs up my original hypothesis about bluring genres, but I was still wrong about the common genres, which differd by 1.3 points, not within 1, and hierarchy violations were the same for both the CNN and GCN. 
- **Ayesha:**

---

## E2: GCN with hierarchy edges vs. hierarchy + co-occurrence edges (training split only)

- **Runs:** 3 planned; 1 run (`e2_gcn_cooc_s0`, seed 0, limited by free GPU time), compared with
  `e1_gcn_s0`
- **Metrics:** macro mAP (overall and by group), hierarchy-violation rate vs. the E1 flat head

### Predictions

- **Maddi** (11/12; committed in `aa41799`): I actually think the GCN with the tree plus genres often tagged
  together might score lower than the tree-only GCN from E1. Mixing information between relatives
  and subgenres, and adding co-occurrence edges, could be problematic because similar genres might
  blur together like I predicted in E1. I think it could decrease macro mAP by 2–3 points (a smaller
  change than my E1 prediction, but I'm not confident).

### Result

| Test split, seed 0 | Tree only (`e1_gcn_s0`) | Tree + co-occurrence (`e2_gcn_cooc_s0`) | E1 flat head (3-seed mean) |
|---|---|---|---|
| macro mAP, all genres | 0.185 | 0.185 | **0.207** |
| rare | 0.084 | 0.081 | **0.112** |
| common | 0.293 | 0.296 | **0.307** |
| top-level | 0.466 | 0.469 | 0.477 |
| fine-grained | 0.133 | 0.132 | **0.156** |
| micro-F1 | 0.477 | 0.474 | 0.439 |
| hierarchy violations ↓ | 0.004 | 0.009 | 0.006 |
| sibling-detector cosine (E3) | 0.821 | 0.799 | 0.37–0.39 |

Adding co-occurrence edges changed nothing (all differences ≤ 0.3 points, smaller than the
seed-to-seed spread in E1). Both GCNs stay about 2 points below the flat head.

### Verdicts

<!-- Maddi: predicted −2 to −3 points; actual change 0. -->
- **Maddi:** I predicted co-occurance edges woiuld lower macro mAP by 2-3 points but there was no hange. The tree-only GCN already made sibling detectors similar (cosine 0.82), so the extra edges had little left to blur. 

---

## E3: GCN depth, 1 vs. 2 vs. 3 layers

- **Runs:** 3 (`e3_gcn_1layer_s0`, `e3_gcn_3layer_s0`; 2 layers is `e1_gcn_s0`), seed 0 each
- **Metrics:** macro mAP; over-smoothing, measured as the mean cosine similarity between
  sibling genres' detectors

### Predictions

- **Maddi** (11/12; committed in `aa41799`): I actually think fewer layers might be better because of my
  general thought that mixing genres causes more blurring/mixing up, so I think one layer will
  score best on macro mAP. As I continue to add layers, sibling genres will become more similar
  (higher cosine similarity), and at 3 layers I'm going to guess macro mAP will be about 4 points
  lower than with 1 layer.

### Result

| Test split, seed 0 | macro mAP | rare | fine-grained | sibling-detector cosine ↑ = more smoothed |
|---|---|---|---|---|
| Flat head (reference, 3 seeds) | 0.207 | 0.112 | 0.156 | 0.37–0.39 |
| GCN, 1 layer | **0.196** | **0.098** | **0.144** | 0.666 |
| GCN, 2 layers (`e1_gcn_s0`) | 0.185 | 0.084 | 0.133 | 0.821 |
| GCN, 3 layers | 0.188 | 0.082 | 0.134 | 0.808 |

Cosine similarity is averaged over all 317 sibling pairs (genres with the same parent), using each
genre's detector from `analyze.detectors` (`analyze.mean_pair_cosine`). The 2-layer GCN's other two
seeds give 0.818 and 0.811.

One layer scores best of the GCNs. Smoothing jumps from 1 to 2 layers and then levels off, and so
does the loss in macro mAP (3 layers is 0.8 points below 1 layer). Even 1 layer is 1.1 points
below the flat head.

### Verdicts

<!-- Maddi: 1 layer best, supported; siblings more similar with depth, supported (rises then plateaus); size: −0.8, not −4. -->
- **Maddi:**   One layer did score best, and the genre siblings DID become more similar with depth (costine 0.67 -> 0.82 -> 0.81), but the drop at three layers was not as signifigant as I arbitrarily guessed because the smoothing leveled off after the second layer. 

---

## E4: Recommendation prototype, nearest neighbours on embeddings

- **Runs:** none new (reuses the E0/E1 checkpoints)
- **Setup:** queries are test-split tracks; candidates are train + validation tracks (no shared
  artists). Embeddings compared: GCN model, flat model, features.csv, random.
- **Metric:** genre overlap@10 (mean Jaccard between the query's genres and each recommendation's
  genres), overall, for rare-genre queries, and for fine-grained queries

### Predictions

- **Miranda** (Nov 11, 2026; committed in `9fce93f`): GCN embeddings beat flat embeddings on genre overlap@10, mostly for rare-genre queries. Both beat features.csv, and all beat random.
- **Maddi** (11/12; committed in `aa41799`): Based on my E0 and E1 predictions, I think the ranking by genre
  overlap@10 will be features.csv > flat > GCN > random. Since I predicted logistic regression on
  features.csv beats the CNN in E0, I expect features.csv vectors to find songs with more similar
  genres too. I expect the gap between flat and GCN to be largest for rare-genre and fine-grained
  queries, because if the GCN blurs similar subgenres together, its embeddings will recommend
  songs from the wrong subgenre more often. All three should clearly beat random.
- **Ayesha** (11/13; committed in `8522a9d`): Ranking by genre overlap@10: GCN > flat > features.csv > random.

### Result

| Genre overlap@10 | all queries | rare-genre queries | fine-grained queries |
|---|---|---|---|
| Flat model (`e0_cnn_flat`) | **0.307** | **0.180** | **0.287** |
| GCN model (`e1_gcn_s0`) | 0.301 | 0.170 | 0.280 |
| features.csv (z-scored) | 0.230 | 0.132 | 0.220 |
| Random | 0.067 | 0.039 | 0.072 |

2,572 test queries, 22,413 train + validation candidates, cosine similarity; computed with
`recommend.py` (through `demo.py`'s `Recommender.overall`). Ranking: flat > GCN > features.csv >
random, in every group; flat vs. GCN differs by about 1 point. _To confirm: Miranda._

### Verdicts

<!-- Miranda: GCN did not beat flat (slightly lower); both beat features.csv and random, supported.
     Maddi: flat > GCN > random right; features.csv was third, not first.
     Ayesha: flat and GCN swapped; rest right. -->
- **Miranda:** GCN did not beat flat (about 1 point lower), but both beat features.csv and random, as I predicted.
- **Maddi:** so the flat > GCN > random was rifht, and the fdlat vs GCN gap was the biggest for rare genres (1.0 pt vs 0.6 overall) as I had thought. but features csv came in third. the CNNS learned features beat the hand made ones. 
- **Ayesha:**

---

## Diagnostics (not separate runs)

### Gradient norms per layer, flat vs. GCN

- **Prediction** (Maddi, 11/12; committed in `aa41799`): Under the GCN, rare genres' detectors will get larger gradients
  than under the flat head, because they also receive updates from their relatives' examples. But
  I think those gradients will also be less specific to each genre, since they mix in signals from
  other genres, which fits my guess that the GCN blurs similar subgenres together.
- **Result:** W&B logs one gradient norm per layer, not per genre, so we measured per genre
  directly: for each trained model's head, the change one gradient step would make to each genre's
  detector (flat: the gradient row; GCN: the directional derivative of the detectors along the
  gradient), on the 199 local clips with the trained encoder fixed.

  | 3 seeds each | detector update size, rare ÷ common | cosine between sibling genres' updates |
  |---|---|---|
  | Flat head | 0.33–0.43 | 0.64–0.78 |
  | GCN head | **0.73–0.77** | **0.94–0.95** |

  Under the GCN, rare genres' detectors get relatively about twice as much update (compared with
  common genres) as under the flat head, and sibling genres receive almost the same update. Raw
  sizes are not compared across heads (different parameterizations). Limits: 199 clips; plain
  gradient direction, while training used Adam.
- **Verdict (Maddi):** rare genres dectors got larger updates under the GCN bc the shared weights pass their relatives' signal to them, and those updates were less genre specific: siblings got basically the dsame update (blurring that I was reffering to before)

### Encoder feature maps (block 2) on clips from three genres

- **What we expect to see** (Maddi, 11/12; committed in `aa41799`): I expect block 2 to pick up simple sound patterns:
  vertical stripes for drum hits in Hip-Hop and Electronic, horizontal lines for held notes in
  Classical, and messier patterns spread across many frequencies in Rock (distorted guitars).
- **Result:** Block 2 of the flat model (`e0_cnn_flat`), one clip per genre, the same 6 channels
  (the most active ones) for each; figure: `report/figures/feature_maps_block2.png`.
  - Hip-Hop ("Food", AWOL): dense vertical stripes, i.e. regular beats (strongest in channels 35,
    20, 52).
  - Rock ("Peel Back The Mountain Sky", Abominog): horizontal bands spread across many
    frequencies, i.e. sustained, distorted sound (channels 38, 0, 59).
  - Classical ("Preludes, Book 2", Paul Pitman; solo piano): separate note shapes with silent
    gaps, not long held lines.
  - The same channel responds very differently by genre: channel 35 is strong on the Hip-Hop
    beats and nearly blank on Rock.
- **Verdict (Maddi):**  vertical stripes for hiip hop drums and messy frequency patterns for rock match what i expected. Classical music didnt show horizontal held-note lines but this clip is solo piano , so notes fade instead of sustaining, so it shows seperate notes instead. One clip per genre is a small sample and I didnt check electronic. 

### Metadata audit: 30 most confident model–tag disagreements

Label each one as a model error, a debatable tag, or a likely wrong tag.

- **Expected split** (Miranda, Nov 15, 2026; committed in `c7af702`): about 10 model errors, 12 debatable tags, and 8 likely wrong tags.
- **Expected split** (Maddi, 11/12; committed in `aa41799`): About 15 model errors, 10 debatable tags, and 5 tags that look
  wrong. Since I expect our from-scratch CNN to be fairly weak (E0), I think most disagreements will
  be the model's mistakes, but FMA's genre tags are chosen by the artists themselves, so some will
  be debatable or simply wrong.
- **Result:** _pending (`analyze.audit_table`, then labelled by all three of us)_
  - model errors: · debatable tags: · likely wrong tags:
- **Verdicts:**
  - **Miranda:**
  - **Maddi:**
