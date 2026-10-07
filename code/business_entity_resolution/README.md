# Business Entity Resolution (Dynamic Team)

This folder contains our full pipeline. It produces `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the challenge data. It uses only the provided data; the text
models start from public MIT-licensed checkpoints and are fine-tuned on the training labels.
Our methodology is described in `docs/Methodology.md` at the root of the repository.

## Setup

- Python 3.12 (we used 3.12.3 on macOS arm64; Linux works as well), about 16 GB of RAM and about
  25 GB of free disk space for intermediate files.
- `pip install -r requirements.txt`
- LightGBM needs OpenMP: `brew install libomp` on macOS, `apt install libgomp1` on Debian/Ubuntu.
- The `text` step needs a CUDA GPU with at least 16 GB of memory (we used NVIDIA RTX 6000 Ada
  cards). All other steps run on a normal CPU.

## Data

Put the challenge data in `dataset/train/` and `dataset/test/`, three levels above `src/`, which is
the root of this repository. Intermediate files go to `work/` in the same place; `ER_WORK` changes that location and
`ER_JOBS` the number of parallel workers (default: all cores).

## Running

From this folder:

```bash
bash src/run_all.sh                  # everything, about 3 h 40 min on a 10-core laptop + 1.5 h on a GPU
bash src/run_all.sh stage3           # resume from a step
bash src/run_all.sh translit base    # run from one step up to and including another
```

The steps, in order:

| step | what it does | time |
| --- | --- | --- |
| `translit` | learns the Indic-script transliteration dictionary from matched training pairs | 10 s |
| `preprocess` | normalises all 24.2 M records | 3 min |
| `blocking` | TF-IDF candidate search per country and state (about 27 candidates per record) | 25 min |
| `stage1` | pruning model; keeps at most 2 candidates per record | 55 min |
| `stage2` | main matcher (two LightGBM models) and threshold selection | 95 min |
| `stage3` | correction model on difference features (numbers, words, letters) | 25 min |
| `stage4` | collective correction of uncertain records | 12 min |
| `base` | decision of stage 4 without text models (first vote of the France rule) | 2 min |
| `text` | fine-tunes the four text models and scores the uncertain pairs (GPU) | 1.5 h |
| `stage4france` | stage 4 with text models, used as the second vote in France | 10 min |
| `stage4text` | stage 4 with text models on a wider set of uncertain records (US and India) | 15 min |
| `submit` | writes both output files | 2 min |
| `france` | France rule: a French match needs both votes | 1 min |
| `siblings` | sibling-group rule for the US and India | 1 min |
| `validate` | runs `utils/validate_submission.py` if it is present | 1 min |

The CPU steps use fixed seeds and give the same results on every run. Training the text models on
a GPU can differ slightly from run to run, so `artifacts/text_scores/` keeps the exact scores behind
the submitted output (see below).

### Without a GPU

The scores of our trained text models (`artifacts/text_scores/`) are part of our submission package
but are too large for this repository. With those files in place, to reproduce the outputs without
a GPU, run everything before the `text` step, copy the saved scores into the work
folder and continue:

```bash
bash src/run_all.sh translit base
mkdir -p ../../work/text && cp artifacts/text_scores/*.parquet ../../work/text/
bash src/run_all.sh stage4france
```

## Outputs

- `output/candidate_pairs.tsv`: the pruned candidate set from stage 1, which is exactly the set
  of pairs the later models score (9.73 M test pairs, 5.6 per Source 1 entity).
- `output/matching_results.tsv`: the final matches.

## Code

| file | purpose |
| --- | --- |
| `run_all.sh` | runs the whole pipeline |
| `common.py` | paths, TSV reading and writing |
| `normalize.py`, `translit.py`, `learn_translit.py`, `preprocess.py` | normalisation of names, addresses and states; Indic transliteration |
| `blocking.py`, `run_blocking.py` | candidate generation |
| `features.py`, `models.py` | pair features, LightGBM helpers |
| `stage1.py` | pruning model |
| `stage2.py`, `tune_prior.py` | main matcher, threshold selection (also for a simulated test-like class mix) |
| `swap.py`, `stage3.py` | how two names differ, word-swap veto, correction model |
| `stage4.py` | collective correction |
| `export_pairs.py`, `text_model.py`, `score_pairs.py` | text models |
| `france_rule.py`, `sibling_rule.py` | the two test-specific rules |
| `make_submission.py` | writes the output files |
| `evaluate.py` | macro F0.5 as defined in the problem statement |
| `eval_blocking.py`, `eval_funnel.py`, `error_analysis.py` | diagnostics |
| `stage2_adapt.py`, `exp_trees.py`, `final_numbers.py` | experiments and result tables (not needed for the outputs) |

## Text models

| tag | checkpoint | licence | parameters | epochs | learning rate |
| --- | --- | --- | --- | --- | --- |
| A | `microsoft/MiniLM-L12-H384-uncased` | MIT | 33 M | 3 | 5e-5 |
| C | `xlm-roberta-base` | MIT | 278 M | 2 | 3e-5 |
| X | `xlm-roberta-large` | MIT | 560 M | 2 | 1.5e-5 |
| D | `microsoft/deberta-v3-base` | MIT | 184 M | 2 | 3e-5 |

Each is fine-tuned as a cross-encoder on "name | address" of the record and of the Source 1
entity, with the same two folds as the other stages, so all training scores are out-of-fold.

## Learned resources

In this repository `artifacts/` holds the small learned resources of our submitted run: the
transliteration dictionary, the neighbouring states, the synonym pairs, the frequent added and
dropped words, and the chosen decision rules with all the rules we tried. The trained models,
the character n-gram vectorisers and the text-model scores are regenerated by the pipeline.

## Validation

Training records are split into two folds by the id of their true Source 1 entity, so all
records of one business are in the same fold, and each fold's models score the other fold. We
choose thresholds on the exact macro F0.5 over all 2.2 M training entities (`evaluate.py`).
`python src/eval_funnel.py` prints the precision and recall of each stage.
