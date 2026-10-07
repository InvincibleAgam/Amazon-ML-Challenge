# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Dynamic Team  
**Team Members:** Aryan Bansal, Abhirath Adamane, Agam Manash Roy, Siddavatam Divya (IIIT Bangalore)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our pipeline is blocking plus a chain of LightGBM models: country-aware normalisation, a sparse
TF-IDF search inside each state, a small model that keeps at most two candidates per record, and
three matching stages. Most of the gain came from studying how the distractors in the data were
generated (a house number shifted by a fixed offset, one word of the name swapped) and from letting
the confident copies of a business confirm its uncertain ones. Four small fine-tuned transformers
(MIT licence, at most 560 M parameters) add a text-based score for the pairs the pipeline is unsure
about. Out-of-fold macro F0.5 on the full training set is 0.9915; our best public leaderboard
score is 0.99058, with 5.6 candidates per Source 1 entity.

---

## 2. Methodology

### 2.1 Problem Analysis

The training set has 2.21 M Source 1, 5.03 M Source 2 and 5.29 M Source 3 records with 7.64 M
labelled pairs. The test set has 1.73 M Source 1 entities and 9.97 M Source 2/3 records, and
includes France (15 % of the entities), which does not appear in training.

What we learned from the data:

- **Structure.** Each Source 2/3 record matches at most one Source 1 entity, so matching is an
  assignment problem. About a quarter of the records match nothing, 5.6 % of Source 1 entities
  have no match at all (they score either 1 or 0), and an entity has 3.5 matches on average.
- **Noise.** Legal-form variants (Pvt Ltd / Private Limited / S.A.R.L.), reordered words, typos,
  accented and digit look-alikes (Onc0logy, 5olutions), appended IDs and phone numbers, domain names
  instead of names, DBA names, honorifics, invented trade names, and names in ten Indic scripts
  (15 % of Source 2 names). Addresses are abbreviated, reordered, partial or missing (3.4 %). In
  France, names are sometimes reduced to initials (CAA for Coaching Air Amis) and dotted legal
  forms lose their final dot (S.A.S).
- **House numbers matter most.** There are no postal codes, so the house number carries most of
  the address signal. A truncated number (118 vs 1189) is usually a typo (80-91 % true matches),
  a same-length change (6106 vs 6101) usually is not (13 %).
- **Namesakes.** The median Source 1 name is shared with at least one other entity in the same
  country. A record without an address whose name several entities share cannot be resolved;
  this is our largest remaining error.
- **The test set is harder.** We estimate about 40 % distractor records in test against 26 % in
  train, so we tuned the decision threshold on a simulated test-like mix as well.

The most useful finding was that **the distractors are built from real entities**. 74 % of the
training distractors carry the house number of a Source 1 entity plus one of a few fixed offsets
(+1, +2, +3, +4, +5, +7, +9, +11, +13, +21); a true copy almost never has one of these offsets
(0.1-0.8 % for +3 and above). Another 16 % take the Source 1 name and swap one word for another
common name word. We call these records siblings. A string similarity cannot tell a sibling from a
typo, so we added features that describe the difference itself (stage 3).

Two more observations drove the later stages:

- **A business keeps its house number within a source.** If one Source 2 copy writes "305-307"
  instead of "305", the other Source 2 copies of that business usually do too (84 %), while a
  distractor carries that number only 4 % of the time. So confident copies can confirm uncertain
  ones (stage 4).
- **France behaves differently.** In US and Indian names the swapped word is the first word; in
  French names it is the second. Our early models had learned the swap by its position and
  accepted many French siblings. We noticed this without labels: France had fewer records per
  entity than the other countries but more predicted matches.

### 2.2 Solution Strategy

**Approach Type:** Hybrid: blocking plus a multi-stage LightGBM classifier, with fine-tuned
transformer scores as additional features  
**Core Innovation:** Features that describe how a record differs from its Source 1 candidate in
the terms the data generator uses (number offset, swapped word, changed letter), and a collective
step in which the confident matches of an entity confirm or reject its uncertain records.

```
raw TSV files
  -> normalisation (names, addresses, states, Indic transliteration)
  -> blocking: sparse TF-IDF top-N per country and state            (~27 candidates per record)
  -> stage 1: LightGBM pruning, top 2 per record                     -> candidate_pairs.tsv
  -> stage 2: LightGBM matcher (string, house-number and collective features)
  -> stage 3: LightGBM correction (how the two records differ)
  -> stage 4: LightGBM on uncertain records (confident copies of the same business,
              scores of four fine-tuned cross-encoders)
  -> best candidate per record, threshold 0.75, margin 0.1, word-swap veto
  -> France rule, sibling-group rule                                 -> matching_results.tsv
```

**Validation.** We split the training records into two folds by the id of their true Source 1
entity, so all records of one business are in the same fold. Every model is cross-fitted, which
gives an out-of-fold score for every training pair. We compute the macro F0.5 exactly as defined
in the problem statement (our own implementation, `evaluate.py`) over all 2.21 M training
entities and choose models and thresholds on that number.

---

## 3. Candidate Generation (Blocking)

Records are normalised first: lower-case ASCII, legal forms mapped to a canonical form and kept in
a separate field, digit look-alikes reversed, IDs, phone numbers and domains stripped, DBA names
split off, Indic-script words transliterated with a dictionary we learned from matched training
pairs (1,347 words), address abbreviations expanded (US, India and France) and the state detected.

Each Source 2/3 record is then compared with the Source 1 records of its country and state by a
sparse TF-IDF top-N search (`sparse_dot_topn`). The cost grows linearly with the number of
records, not quadratically.

- **Blocking keys used:** name words; consonant skeletons of name words (robust to typos and
  transliteration); 4-letter prefixes; a prefix of the joined name (for domain names); the legal
  form (low weight); initials, so that CAA finds Coaching Air Amis; address words; and up to three
  house numbers. The score is 0.6 × name similarity + 0.4 × address similarity, where the address
  part is normalised so that a partial address ("A 80, East Delhi") still matches the full one.
  We keep the top 20 by this score, the top 10 by name and the top 10 by address. A record is
  searched in its own state, in an alternative state (a city named after a state, as in
  "Indiana, PA") and in neighbouring states learned from the training pairs (Telangana and
  Andhra Pradesh).
- **Candidate pairs generated:** 9,725,262 test pairs, 5.6 per Source 1 entity, after a small
  LightGBM model (300 trees, 26 cheap features) keeps at most two candidates per record with a
  probability of at least 0.01. All later models run only on this set, so this is exactly what
  `candidate_pairs.tsv` contains. The full cross product would be 1.7 × 10¹³ pairs.
- **How you ensured true matches were not lost:** we measured recall on the full training set
  after each step:

| | after retrieval | after pruning (submitted) |
| --- | --- | --- |
| recall of true pairs (train) | 99.17 % | 98.35 % |
| candidates per Source 1 entity (train / test) | 124 / 154 | 4.6 / 5.6 |

With an address present, retrieval recall is 99.96 % (US) and 99.57 % (India). Almost all misses
are records without an address and with a very common name, which a precision-weighted metric
would not reward anyway. Keeping eight candidates instead of two would have raised the recall
ceiling by 0.5 points but changed only 453 of 7.29 M correct final matches, so we kept the
smaller set.

---

## 4. Matching Model

**Features used:**

- **Name features:** rapidfuzz ratio, token-sort, token-set and partial ratio, Jaro-Winkler,
  normalised Levenshtein, word Jaccard and containment, first-word and exact-name equality,
  consonant-skeleton similarity, character 3-4-gram TF-IDF cosine, legal-form agreement, initials
  match, alias and domain similarity, and the number of Source 1 entities with the same name.
- **Address features:** word ratios, Jaccard and containment, character n-gram cosine of the
  street words, state agreement, how two house numbers relate (equal, prefix, suffix, same length
  but different, other), number overlap, and the number of Source 1 entities at the same address.
- **Other:**
  - retrieval scores and ranks, source and script flags;
  - collective features from stage 1: how many records claim the same Source 1 entity, per
    source (the generator caps an entity at 5 Source 2 and 6 Source 3 copies);
  - difference features (stage 3): signed difference of the house numbers and digits changed;
    Source 1 words left over; a word swapped in (a common name word that is not a typo or
    shortening of the missing word); generic words added (services, groupe); 13 synonym pairs
    learned from the labels; and the letter edit between two near-identical words, where
    respellings such as c→k or s→z point to siblings and scanner-like errors such as g→6 or
    i→l point to true copies;
  - collective features (stage 4): for records that stage 3 is unsure about, how many confident
    matches of the same entity share the record's unusual house number, address or name, whether
    confident copies from the same source agree with the record's number, and spelling cues from
    the raw names (punctuation and the spelling of the legal form survive in true copies);
  - text features (stage 4): the probabilities of the four cross-encoders, each model's margin
    over the record's other candidate, and how many other records it gives to the same entity
    with confidence.

**Model type:** LightGBM (MIT licence), trained from scratch, in four stages: stage 1 with 300
trees, stage 2 as an average of two models (2,000 and 3,000 trees), stage 3 with 600 trees and
stage 4 with 1,200 trees. The text scores come from four public checkpoints that we fine-tuned as
cross-encoders on "name | address" of both records, only with our training labels, on the 0.9 M
uncertain training pairs (same two folds, about 1.5 hours on GPUs):

| checkpoint | licence | parameters | held-out log loss |
| --- | --- | --- | --- |
| microsoft/MiniLM-L12-H384-uncased | MIT | 33 M | 0.264 |
| FacebookAI/xlm-roberta-base | MIT | 278 M | 0.256 |
| FacebookAI/xlm-roberta-large | MIT | 560 M | 0.241 |
| microsoft/deberta-v3-base | MIT | 184 M | 0.246 |

(On the same pairs, the stage-3 probability has a log loss of 0.298.)

**Threshold selection method:** each record keeps only its best candidate, which is accepted if
its probability is at least 0.75 and exceeds the runner-up by 0.1. We chose the threshold and
the margin by a grid search on the out-of-fold macro F0.5, both on the training class mix and on
a simulated test mix with 40 % distractors, and took the best value on the test-like mix. A
per-entity expected-F0.5 decoder was slightly better on the training mix but worse on the
test-like one, so we kept the simple threshold. Three rules follow:

- **Word-swap veto.** A match is dropped when the record's name swaps in a common name word for
  a missing Source 1 word. On training data this removed 10 % of the false merges at 99 %
  precision; its main purpose is to protect France.
- **France rule.** France has no labels, and the text models, trained on US and Indian data,
  accepted French siblings. On the leaderboard, text models in every country scored 0.98924,
  against 0.98967 with France left on stage 4 without them. So in France we keep a match only if
  stage 4 without text models accepts it and the text version also gives it at least 0.75
  (0.99026). This rule was chosen with leaderboard feedback, not with out-of-fold scores.
- **Sibling-group rule (US and India).** In the test data siblings come in groups of two or
  three at the same house number, which never happens in training. We drop a match with
  probability at most 0.98 whose unusual number is shared with records of the same entity that
  were confidently rejected (374 of 5.85 M matches; 7 of 8 were siblings when we checked them by
  hand). This rule is also test-specific; it added 0.00003 on the leaderboard.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9915 out-of-fold on the full training set (0.9914 on the simulated
  test mix); best public leaderboard score 0.99058.

Out-of-fold results of the final model:

| class mix | macro F0.5 | US | India | singletons | pair precision | pair recall |
| --- | --- | --- | --- | --- | --- | --- |
| training | 0.9915 | 0.9915 | 0.9916 | 0.9982 | 0.9993 | 0.9750 |
| test-like (40 % distractors) | 0.9914 | 0.9914 | 0.9915 | 0.9982 | 0.9990 | 0.9750 |

How the leaderboard score developed:

| version | main change | public leaderboard |
| --- | --- | --- |
| v2 | first complete pipeline with house-number features | 0.9748 |
| v7 | normalisation fixes for France, model ensemble, character n-grams | 0.9761 |
| v10 | word-swap veto | 0.9809 |
| v12 | stage 3 (difference features, letter edits) | 0.9861 |
| v13 | stage 4 (confident copies confirm uncertain ones) | 0.9881 |
| v15b | first text model (US and India) | 0.9897 |
| v16c | France rule, large text model | 0.9903 |
| v19d | four text models, wider stage-4 input | 0.99055 |
| v19s | sibling-group rule (final) | 0.99058 |

For the US and India, out-of-fold gains predicted the leaderboard gains closely (for the first
text model: +0.00153 predicted, +0.00152 measured), so we relied on out-of-fold scores and used
the leaderboard mainly to check France. Two early diagnostic uploads (one with all French
predictions left empty) showed France at about 0.94 while the US and India were at 0.982; the
word-swap veto closed most of that gap.

- **Common false positives (wrong merges):** mostly siblings: the same address with the house
  number shifted (6106 vs 6101 Doc Sandlin Rd), one name word swapped (Pryor Gulf Gravity vs
  Morris Gulf Gravity), a different Roman numeral (Council II vs III) or a different legal form at
  the same address. The word-swap veto, stages 3 and 4 and the text models reduced the
  out-of-fold false merges from 27,693 to 5,501. What remains is mostly name-only records given to the wrong namesake, and
  number changes that look the same whether they are typos or siblings.
- **Common false negatives (missed matches):** records without an address whose name is shared
  by several Source 1 entities. Here the model is well calibrated (97 % of such pairs are true
  with one namesake and 16 % with five or more; the model predicts 96 % and 18 %), so these are
  ambiguous rather than model errors. Other misses are invented trade names with only a fragment
  of the address, and heavily garbled names with partial addresses. Recall is 99.4 % for records
  with an address and 56 % for records without one; about a third of the misses were never
  retrieved.

---

## 6. Conclusion

Most of our score came from understanding the data rather than from bigger models: how the
generator builds distractors (fixed number offsets, swapped words, respelled letters), that a
business keeps its house number within a source, and that the test set contains more distractors
than the training set. The models themselves are LightGBM with careful cross-fitting and a
threshold chosen on the exact metric; the fine-tuned transformers added the last 0.002. Our main
lessons: validate on the exact metric with folds split by entity, check an unlabelled country
with statistics that need no labels, and remember that a model can learn the right rule for the
wrong reason (the position of a word) and fail where that reason does not hold.

---

## Appendix

### A. Code Artefacts

The runnable pipeline is in `code/business_entity_resolution/`, with a `README.md` and a
`requirements.txt` with pinned versions. The entry point is

```bash
cd code/business_entity_resolution
bash src/run_all.sh
```

It runs every step from the raw TSV files to both output files: about 3 h 40 min on a 10-core
laptop, plus about 1.5 h on a GPU for the text models. `bash src/run_all.sh <step>` resumes from a
step; the README also explains how to reproduce everything after the GPU step from the saved text
scores. The main files in `src/`:

| files | purpose |
| --- | --- |
| `normalize.py`, `translit.py`, `learn_translit.py`, `preprocess.py` | normalisation, Indic transliteration |
| `blocking.py`, `run_blocking.py` | candidate generation |
| `stage1.py` | pruning; writes the candidate set |
| `features.py`, `models.py`, `stage2.py`, `tune_prior.py` | main matcher and threshold selection |
| `swap.py`, `stage3.py` | difference features, word-swap veto, correction model |
| `stage4.py` | collective correction |
| `export_pairs.py`, `text_model.py`, `score_pairs.py` | text models (GPU) |
| `france_rule.py`, `sibling_rule.py` | the two test-specific rules |
| `make_submission.py`, `evaluate.py` | output files, macro F0.5 |

`artifacts/` holds the trained models, decision rules and text scores of the submitted run.

### B. Additional Results

House-number difference (record minus Source 1) for the best candidates of US training records
whose first number differs but has the same number of digits (5 % sample). This table is how we
found the generator's offsets:

| difference | records | true matches |
| --- | --- | --- |
| +3, +4, +5, +7, +9, +11, +13, +21 | 5,300-5,650 each | 0.1-0.8 % |
| +1, +2 | 6,100-6,300 each | 8.4 % |
| other positive values up to 25 | 9-95 each | 10-97 % |
| -1, -2 | 545 each | 95-96 % |
| other negative values down to -25 | 8-56 each | 30-98 % |

Recall of true training pairs when keeping the top k candidates after pruning: k = 2: 98.35 %,
k = 4: 98.69 %, k = 8: 98.87 %, k = 10: 98.90 %.

Decision rules on the stage-2 probability (out-of-fold):

| rule | training mix | test-like mix |
| --- | --- | --- |
| threshold 0.75, margin 0.1 | 0.98355 | 0.98245 |
| threshold 0.70, margin 0.1 | 0.98371 | 0.98243 |
| expected F0.5 per entity | 0.98389 | 0.98211 |

Tried and not used, because the out-of-fold score did not improve: a larger stage-4 model, a
second pass of stage 4 on its own output, averaging stage-4 versions, self-training on French
pseudo-labels, and recovering unretrieved records through shared addresses.

---

**Fair play.** We used no external databases, APIs, geocoders or internet data. The four
transformer checkpoints are public model weights (MIT licence) and were fine-tuned only on the
challenge's training pairs. All other learned resources come from the training labels
(transliteration dictionary, neighbouring states, synonym pairs, the LightGBM models) or from the
provided records without labels (name vocabularies, TF-IDF weights). Hand-written tables hold only
general knowledge: state and region names, French departments, street-type and legal-form
abbreviations.
