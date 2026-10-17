#!/usr/bin/env bash
# 2026-10-16 の組み替え（run_all_models.sh の残りを置き換える）。
#   - LightGBM・clogit・Bayes は HPO・walk-forward とも完了。CPU が空いた
#   - CatBoost fund は CPU で（HPO・fit とも CB_TASK=CPU、8 スレッド）
#   - CatBoost pop は GPU のまま。走行中の HPO（PID を引数で受ける）が終わるのを待って続きと fit
#   - TabM は hidden=512 を打ち切り、上限 70（trial_limits.json）
# 再起動後は引数なしで起動すればよい（待つ PID が無ければすぐ次へ進む）:
#   cd learning && setsid nohup bash run_phase2.sh > artifacts/all_models/logs_phase2.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
PY=".venv/bin/python -u scripts/all_models.py"
L=artifacts/all_models/logs
WAIT_PID=${1:-}

step() {  # variant model trials
  echo "STAGE $2/$1 hpo_start $(date -Iseconds)"
  $PY hpo $1 $2 --trials $3 || { echo "FAILED $2/$1 hpo $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 fit_start $(date -Iseconds)"
  $PY fit $1 $2 || { echo "FAILED $2/$1 fit $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 done $(date -Iseconds)"
}

( export CB_GPU_RAM_PART=0.45
  if [ -n "$WAIT_PID" ]; then while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done; fi
  step pop catboost 300 ) >> $L/gpu.log 2>&1 &
( export CB_TASK=CPU CB_THREADS=8; step fund catboost 300 ) >> $L/catboost_fund_cpu.log 2>&1 &
sleep 90
( step pop tabm 300; step fund tabm 300 ) >> $L/gpu_tabm.log 2>&1 &

( while true; do
    av=$(free -g | awk '/Mem:/{print $7}')
    echo "$(date -Iseconds) $(free -g | awk '/Mem:/{print "used="$3"G avail="$7"G"}') $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null) $(ps -eo rss,args --sort=-rss | awk '/all_models.py/ && !/awk/{printf "%.1fG:%s/%s ", $1/1048576, $(NF-3), $(NF-2)}' | head -c 300)"
    if [ "$av" -lt 5 ]; then
      pid=$(ps -eo pid,rss,args --sort=-rss | awk '/all_models.py/ && !/awk/{print $1; exit}')
      echo "$(date -Iseconds) WATCHDOG avail=${av}G kill $pid"; kill $pid
    fi
    sleep 30; done ) >> $L/mem.log 2>&1 &
MEM=$!
wait %1 %2 %3
kill $MEM
echo "PHASE2_DONE $(date -Iseconds)"
