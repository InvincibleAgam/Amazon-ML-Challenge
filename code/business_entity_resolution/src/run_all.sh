#!/usr/bin/env bash
# End-to-end reproduction: data -> normalisation -> blocking -> stages 1-4 -> text models -> outputs.
#
#   bash src/run_all.sh                 # run every step
#   bash src/run_all.sh stage1          # run from a given step onward
#   bash src/run_all.sh translit base   # run from one step up to and including another
#
# Steps (in order): translit preprocess blocking stage1 stage2 stage3 stage4 base text stage4france
#                   stage4text submit france siblings validate
# The step `text` fine-tunes four transformers and needs a CUDA GPU (about 1.5 h on two RTX 6000);
# every other step runs on a laptop.  TEXT_GPUS="0 1" chooses the cards (one fold per card).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
FROM="${1:-translit}"
TO="${2:-validate}"
ROOT="$(cd ../../.. && pwd)"
WORK="${ER_WORK:-$ROOT/work}"
TEXT="$WORK/text"
read -r -a GPUS <<< "${TEXT_GPUS:-0 1}"
ZONE="0.001,0.999"                      # wider open zone of stage 4 with text models (US / India decisions)
T_FRANCE=0.75                           # France: stage 4 without text models AND stage 4 with text models on the
                                        # normal zone (probability >= T_FRANCE) must accept (france_rule.py)
# text models: tag, checkpoint (MIT licence), epochs, learning rate
MODELS=("A microsoft/MiniLM-L12-H384-uncased 3 5e-5"
        "C xlm-roberta-base 2 3e-5"
        "X xlm-roberta-large 2 1.5e-5"
        "D microsoft/deberta-v3-base 2 3e-5")
STEPS=(translit preprocess blocking stage1 stage2 stage3 stage4 base text stage4france stage4text submit france siblings validate)
started=0
for step in "${STEPS[@]}"; do
  [[ "$step" == "$FROM" ]] && started=1
  [[ $started -eq 0 ]] && continue
  echo "=== $step  $(date '+%H:%M:%S')"
  case "$step" in
    translit)   "$PY" learn_translit.py ;;
    preprocess) "$PY" preprocess.py train test ;;
    blocking)   "$PY" run_blocking.py train test ;;
    stage1)     "$PY" stage1.py train
                "$PY" -c "import stage1; stage1.predict_and_prune('train')"
                "$PY" -c "import stage1; stage1.predict_and_prune('test')" ;;
    stage2)     "$PY" stage2.py features train
                "$PY" stage2.py features test
                "$PY" stage2.py train
                "$PY" stage2.py predict train
                "$PY" stage2.py tune
                "$PY" tune_prior.py 0.40
                "$PY" stage2.py predict test ;;
    stage3)     "$PY" swap.py learn
                "$PY" stage3.py features train
                "$PY" stage3.py features test
                "$PY" stage3.py train
                "$PY" stage3.py tune
                "$PY" stage3.py predict test ;;
    stage4)     "$PY" stage4.py words
                "$PY" stage4.py features train
                "$PY" stage4.py features test
                "$PY" stage4.py train
                "$PY" stage4.py tune
                "$PY" stage4.py predict test ;;
    base)       # the decision of stage 4 without text models, kept for the France rule
                ER_OUT="$WORK/output_notext" "$PY" make_submission.py ;;
    text)       "$PY" export_pairs.py
                "$PY" export_pairs.py extra
                for m in "${MODELS[@]}"; do
                  read -r tag model epochs lr <<< "$m"
                  mkdir -p "$TEXT/out_$tag"
                  for f in 0 1; do     # one fold per card, both at once
                    CUDA_VISIBLE_DEVICES="${GPUS[$f]:-${GPUS[0]}}" "$PY" text_model.py --fold $f --gpu 0 --data "$TEXT" \
                        --out "$TEXT/out_$tag" --model "$model" --epochs "$epochs" --lr "$lr" > "$TEXT/out_$tag/fold$f.log" 2>&1 &
                  done
                  wait
                  for f in 0 1; do
                    CUDA_VISIBLE_DEVICES="${GPUS[$f]:-${GPUS[0]}}" "$PY" score_pairs.py --tag "$tag" --fold $f --gpu 0 \
                        --data "$TEXT" --out "$TEXT/out_$tag" > "$TEXT/out_$tag/score$f.log" 2>&1 &
                  done
                  wait
                  "$PY" score_pairs.py --merge --tag "$tag" --out "$TEXT/out_$tag" --into "$TEXT"
                done ;;
    stage4france) # stage 4 with text models on the normal zone: its probability is the second vote in France
                ER_TEXT=A,X,C ER_TEXT_DIR="$TEXT" "$PY" stage4.py features train
                ER_TEXT=A,X,C ER_TEXT_DIR="$TEXT" "$PY" stage4.py features test
                ER_TEXT=A,X,C ER_TEXT_DIR="$TEXT" "$PY" stage4.py train
                ER_TEXT=A,X,C ER_TEXT_DIR="$TEXT" "$PY" stage4.py predict test
                cp "$WORK/test_scores4.parquet" "$WORK/test_scores4_france.parquet" ;;
    stage4text) export ER_TEXT=A,X,C,D ER_ZONE="$ZONE" ER_TEXT_DIR="$TEXT"
                "$PY" stage4.py features train
                "$PY" stage4.py features test
                "$PY" stage4.py train
                "$PY" stage4.py tune
                "$PY" stage4.py predict test ;;
    submit)     "$PY" make_submission.py ;;
    france)     "$PY" france_rule.py "$ROOT/output/matching_results.tsv" "$WORK/output_notext/matching_results.tsv" \
                    "$WORK/test_scores4_france.parquet" "$T_FRANCE" "$ROOT/output" ;;
    siblings)   # US / India: drop less certain matches that belong to a group of rejected siblings (sibling_rule.py)
                "$PY" sibling_rule.py "$ROOT/output/matching_results.tsv" "$WORK/test_scores4.parquet" "$ROOT/output" ;;
    validate)   # the challenge's validator (student_resource/utils/, not part of this package)
                if [[ -f "$ROOT/utils/validate_submission.py" ]]; then
                  "$PY" "$ROOT/utils/validate_submission.py" \
                      --matching "$ROOT/output/matching_results.tsv" \
                      --candidate "$ROOT/output/candidate_pairs.tsv" \
                      --test-dir "$ROOT/dataset/test"
                else
                  echo "utils/validate_submission.py not found next to code/: skipped"
                fi ;;
  esac
  [[ "$step" == "$TO" ]] && break
done
echo "=== done  $(date '+%H:%M:%S')"
