#!/usr/bin/env bash
# GPU を CatBoost と TabM で分け合う版（2026-10-14）。run_all_models.sh の GPU 鎖の置き換え。
# CatBoost の 1 試行が約 20 分で GPU 使用率が 30% 台だったため、TabM を並走させる。
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
PY=".venv/bin/python -u scripts/all_models.py"
L=artifacts/all_models/logs
step() {
  echo "STAGE $2/$1 hpo_start $(date -Iseconds)"
  $PY hpo $1 $2 --trials $3 || { echo "FAILED $2/$1 hpo $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 fit_start $(date -Iseconds)"
  $PY fit $1 $2 || { echo "FAILED $2/$1 fit $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 done $(date -Iseconds)"
}
( export CB_GPU_RAM_PART=0.45; step pop catboost 300; step fund catboost 300 ) >> $L/gpu.log 2>&1 &
sleep 90
( step pop tabm 300; step fund tabm 300 ) > $L/gpu_tabm.log 2>&1 &
wait
echo "GPU_CHAINS_DONE $(date -Iseconds)"
