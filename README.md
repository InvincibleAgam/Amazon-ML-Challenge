# Business Entity Resolution: Amazon ML Challenge 2026

Our solution to the Amazon ML Challenge 2026, by **Dynamic Team** (IIIT Bangalore): Aryan Bansal, Abhirath Adamane,
Agam Manash Roy and Siddavatam Divya.

The task: business records arrive from three sources with no shared IDs. Source 1 is a clean reference list.
For every Source 1 business, find all the Source 2 and Source 3 records that are the same business. The score is
macro F0.5, computed per Source 1 business, so a wrong merge costs more than a missed match.

## Results

| | |
| --- | --- |
| Final standing | Top 10 of 32,000+ teams (Grand Finale) |
| Public leaderboard, macro F0.5 | 0.99058 |
| Out-of-fold macro F0.5 on all 2.21 M training businesses | 0.9915 |
| Pair precision / pair recall (out-of-fold) | 99.93% / 97.50% |
| Candidate pairs per Source 1 business | 5.6 (9.73 M pairs), keeping 98.35% of true pairs |
| Run time | about 4 h on a 10-core laptop, plus 1.5 h on a GPU for the text models |

We were selected as one of the Top 10 teams out of 32,000+, and presented this solution at the Grand Finale to a
panel of senior Amazon scientists on 7 October 2026.

## The idea

Our first pipeline stalled at 0.975. Reading its wrong merges showed why: most distractor records are built from a
real business with one thing changed, either the house number shifted by one of a few fixed amounts or one word of
the name replaced. To a similarity score such a record looks the same as a true copy with a typo. So we stopped
scoring how similar two records are and started describing how they differ. That, and letting the confident copies
of a business confirm its uncertain ones, gave most of our score.

## Pipeline

| Step | What runs | Pairs left (test) |
| --- | --- | --- |
| Normalise | Rules and tables: legal forms, look-alike digits, address abbreviations, Indic transliteration | |
| Blocking | Sparse TF-IDF top-k search inside each state | 266 M |
| Stage 1 | LightGBM pruning model, keeps at most 2 candidates per record | 9.73 M |
| Stage 2 | LightGBM matcher on 66 similarity features | |
| Stage 3 | LightGBM on how two records differ: number difference, swapped word, changed letter | |
| Stage 4 | LightGBM on the records still uncertain: support from confident copies, four cross-encoder scores | 2.39 M uncertain |
| Decision | Best candidate per record, threshold 0.75, margin 0.1, three rules | 5.85 M matches |

Every model is cross-fitted on two folds split by the true business, and the decision rule is chosen on the exact
metric under a simulated test-like class mix.

## What is in this repository

| Path | Content |
| --- | --- |
| `code/business_entity_resolution/` | The full pipeline, with its own README (setup, steps, run times) and pinned requirements |
| `code/business_entity_resolution/artifacts/` | Small learned resources of our submitted run: transliteration dictionary, decision rules, tuning results |
| `docs/Methodology.md` | The methodology write-up we submitted |
| `docs/Architecture_and_Design_Choices.pdf` | Three pages: every step, its settings, and why we chose it |
| `docs/Presentation.pdf` | The slides of our jury presentation |
| `kaggle/` | Notebooks we used to fine-tune two of the text models on Kaggle GPUs |

## Running it

Put the challenge data in `dataset/train/` and `dataset/test/` at the root of this repository, then:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
bash src/run_all.sh
```

Details, the list of steps and how to resume from a step are in `code/business_entity_resolution/README.md`.

## What is not here

- **The challenge data and our output files.** They belong to the organisers and are large.
- **Trained models and text-model scores.** The pipeline regenerates them; together they are several hundred MB.

We used only the data provided by the challenge. The four text models start from public MIT-licensed checkpoints
and were fine-tuned on the training labels.
