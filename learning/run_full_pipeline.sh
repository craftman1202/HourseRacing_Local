#!/usr/bin/env bash
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
D="file://$PWD/data_real"
PY=.venv/bin/python
TABM="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 4096"

echo "PIPELINE_STAGE learn_flat_start $(date -Iseconds)"
$PY -m nar.cli --data-root "$D" --artifacts ./artifacts learn \
  --models clogit,lgbm,tabm,bayes $TABM --seed 0
rc=$?
echo "PIPELINE_STAGE learn_flat_done rc=$rc $(date -Iseconds)"
[ $rc -ne 0 ] && { echo "PIPELINE_FAILED learn_flat"; exit 1; }

echo "PIPELINE_STAGE learn_banei_start $(date -Iseconds)"
NAR_CONF_DIR=conf_banei $PY -m nar.cli --data-root "$D" --artifacts ./artifacts/banei learn \
  --models clogit,lgbm,tabm,bayes $TABM --seed 0
rc=$?
echo "PIPELINE_STAGE learn_banei_done rc=$rc $(date -Iseconds)"
[ $rc -ne 0 ] && { echo "PIPELINE_FAILED learn_banei"; exit 1; }

echo "PIPELINE_STAGE gate_flat_start $(date -Iseconds)"
$PY -m nar.cli --data-root "$D" --artifacts ./artifacts gate-report
rc=$?
echo "PIPELINE_STAGE gate_flat_done rc=$rc $(date -Iseconds)"

echo "PIPELINE_STAGE gate_banei_start $(date -Iseconds)"
NAR_CONF_DIR=conf_banei $PY -m nar.cli --data-root "$D" --artifacts ./artifacts/banei gate-report
rc=$?
echo "PIPELINE_STAGE gate_banei_done rc=$rc $(date -Iseconds)"

echo "PIPELINE_STAGE fit_final_flat_start $(date -Iseconds)"
$PY -m nar.cli --data-root "$D" --artifacts ./artifacts fit-final \
  --out ./artifacts/final_new --through 2026-09-17 $TABM --seed 0
rc=$?
echo "PIPELINE_STAGE fit_final_flat_done rc=$rc $(date -Iseconds)"
[ $rc -ne 0 ] && { echo "PIPELINE_FAILED fit_final_flat"; exit 1; }

echo "PIPELINE_STAGE fit_final_banei_start $(date -Iseconds)"
NAR_CONF_DIR=conf_banei $PY -m nar.cli --data-root "$D" --artifacts ./artifacts/banei fit-final \
  --out ./artifacts/banei/final_new --through 2026-09-17 $TABM --seed 0
rc=$?
echo "PIPELINE_STAGE fit_final_banei_done rc=$rc $(date -Iseconds)"
[ $rc -ne 0 ] && { echo "PIPELINE_FAILED fit_final_banei"; exit 1; }

echo "PIPELINE_STAGE evaluate_final_flat_start $(date -Iseconds)"
$PY -m nar.cli --data-root "$D" --artifacts ./artifacts evaluate-final \
  --final-dir ./artifacts/final_new --unlock-oos \
  --reason "2026-09-17 full re-run: fresh data + fit-final-matched TabM width in walk-forward + scaling-skill review (no scaling change adopted)"
rc=$?
echo "PIPELINE_STAGE evaluate_final_flat_done rc=$rc $(date -Iseconds)"

echo "PIPELINE_STAGE evaluate_final_banei_start $(date -Iseconds)"
NAR_CONF_DIR=conf_banei $PY -m nar.cli --data-root "$D" --artifacts ./artifacts/banei evaluate-final \
  --final-dir ./artifacts/banei/final_new --unlock-oos \
  --reason "2026-09-17 full re-run: fresh data + fit-final-matched TabM width in walk-forward + scaling-skill review (no scaling change adopted)"
rc=$?
echo "PIPELINE_STAGE evaluate_final_banei_done rc=$rc $(date -Iseconds)"

echo "PIPELINE_ALL_DONE $(date -Iseconds)"
