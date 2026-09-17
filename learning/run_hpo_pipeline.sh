#!/usr/bin/env bash
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
D="file://$PWD/data_real"
PY=.venv/bin/python
TABM_BASE="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 4096"
HPO="--hpo-clogit 15 --hpo-lgbm 8 --hpo-tabm 2"
TODAY=2026-09-17

run_family () {
  local FAMILY=$1 CONF=$2 ART=$3
  local CONFENV=""
  [ "$CONF" != "conf" ] && CONFENV="NAR_CONF_DIR=$CONF"

  echo "PIPELINE_STAGE learn_${FAMILY}_start $(date -Iseconds)"
  env $CONFENV $PY -m nar.cli --data-root "$D" --artifacts "$ART" learn \
    --models clogit,lgbm,tabm,bayes $TABM_BASE $HPO --null-runs 8 --seed 0
  rc=$?
  echo "PIPELINE_STAGE learn_${FAMILY}_done rc=$rc $(date -Iseconds)"
  [ $rc -ne 0 ] && { echo "PIPELINE_FAILED learn_${FAMILY}"; return 1; }

  $PY scripts/extract_hpo_params.py "$ART/hpo.json" "$ART/hpo_params.json"
  rc=$?
  echo "PIPELINE_STAGE extract_hpo_${FAMILY}_done rc=$rc $(date -Iseconds)"
  [ $rc -ne 0 ] && { echo "PIPELINE_FAILED extract_hpo_${FAMILY}"; return 1; }

  echo "PIPELINE_STAGE gate_${FAMILY}_start $(date -Iseconds)"
  env $CONFENV $PY -m nar.cli --data-root "$D" --artifacts "$ART" gate-report
  echo "PIPELINE_STAGE gate_${FAMILY}_done rc=$? $(date -Iseconds)"

  echo "PIPELINE_STAGE fit_final_eval_${FAMILY}_start $(date -Iseconds)"
  env $CONFENV $PY -m nar.cli --data-root "$D" --artifacts "$ART" fit-final \
    --out "$ART/final_hpo_eval" $TABM_BASE --null-runs 8 --seed 0 \
    --hpo-params-json "$ART/hpo_params.json"
  rc=$?
  echo "PIPELINE_STAGE fit_final_eval_${FAMILY}_done rc=$rc $(date -Iseconds)"
  [ $rc -ne 0 ] && { echo "PIPELINE_FAILED fit_final_eval_${FAMILY}"; return 1; }

  echo "PIPELINE_STAGE evaluate_final_${FAMILY}_start $(date -Iseconds)"
  env $CONFENV $PY -m nar.cli --data-root "$D" --artifacts "$ART" evaluate-final \
    --final-dir "$ART/final_hpo_eval" --unlock-oos \
    --reason "2026-09-17 HPO run ($FAMILY): evaluating the properly embargoed (through 2023-08-04) HPO-tuned model"
  echo "PIPELINE_STAGE evaluate_final_${FAMILY}_done rc=$? $(date -Iseconds)"

  echo "PIPELINE_STAGE fit_final_prod_${FAMILY}_start $(date -Iseconds)"
  env $CONFENV $PY -m nar.cli --data-root "$D" --artifacts "$ART" fit-final \
    --out "$ART/final_hpo_prod" --through $TODAY $TABM_BASE --null-runs 8 --seed 0 \
    --hpo-params-json "$ART/hpo_params.json"
  rc=$?
  echo "PIPELINE_STAGE fit_final_prod_${FAMILY}_done rc=$rc $(date -Iseconds)"
  [ $rc -ne 0 ] && { echo "PIPELINE_FAILED fit_final_prod_${FAMILY}"; return 1; }

  return 0
}

run_family flat conf ./artifacts/hpo_flat
[ $? -ne 0 ] && exit 1

run_family banei conf_banei ./artifacts/hpo_banei
[ $? -ne 0 ] && exit 1

echo "PIPELINE_ALL_DONE $(date -Iseconds)"
