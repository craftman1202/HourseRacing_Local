#!/usr/bin/env bash
# 全モデル × 2 系統の HPO → walk-forward（scripts/all_models.py、2026-10-13〜）。
# どの段も再実行で続きから（HPO は optuna の sqlite、fit は fold ごとのファイル）。
# 停電・再起動のあとは、このスクリプトをもう一度起動するだけでよい:
#   cd learning && setsid nohup bash run_all_models.sh > artifacts/all_models/logs_main.log 2>&1 < /dev/null &
#
# 鎖（並列）:
#   gpu_cb   CatBoost pop → fund（GPU、VRAM 45% まで）
#   gpu_tabm TabM pop → fund（GPU を CatBoost と分け合う）
#   lgbm_pop / lgbm_fund  LightGBM（CPU 5 スレッドずつ）
#   cpu_misc clogit（済み）→ Bayes fund → Bayes pop
# 試行数の上限は artifacts/all_models/trial_limits.json が優先（2026-10-15 削減: lgbm120/catboost120/tabm150/bayes50）
# 経緯: 2026-10-14 に 4 本同時起動で gold を同時に読み WSL ごと落ちた → キャッシュ・時間差起動・見張り。
#       同日 CatBoost の GPU 使用率が 30% 台 → TabM を並走。
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
PY=".venv/bin/python -u scripts/all_models.py"
L=artifacts/all_models/logs
mkdir -p $L
if pgrep -f "scripts/all_models.py (hpo|fit)" > /dev/null; then
  echo "all_models.py が既に動いています。二重起動しません"; exit 1
fi

step() {  # variant model trials
  echo "STAGE $2/$1 hpo_start $(date -Iseconds)"
  $PY hpo $1 $2 --trials $3 || { echo "FAILED $2/$1 hpo $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 fit_start $(date -Iseconds)"
  $PY fit $1 $2 || { echo "FAILED $2/$1 fit $(date -Iseconds)"; return 1; }
  echo "STAGE $2/$1 done $(date -Iseconds)"
}

[ -f artifacts/all_models/data_cache.parquet ] || $PY bench pop --models clogit > $L/cache.log 2>&1 \
  || { echo "FAILED cache"; exit 1; }

( export CB_GPU_RAM_PART=0.45; step pop catboost 300; step fund catboost 300 ) >> $L/gpu.log 2>&1 &
sleep 90
( step pop tabm 300; step fund tabm 300 ) >> $L/gpu_tabm.log 2>&1 &
sleep 90
( export LGBM_THREADS=5; step pop lgbm 500 ) >> $L/lgbm_pop.log 2>&1 &
sleep 90
( export LGBM_THREADS=5; step fund lgbm 500 ) >> $L/lgbm_fund.log 2>&1 &
sleep 90
( export XLA_FLAGS="--xla_force_host_platform_device_count=1"
  step pop clogit 100; step fund clogit 100; step fund bayes 50; step pop bayes 50 ) >> $L/cpu_misc.log 2>&1 &

# メモリの推移と見張り: 空きが 5GB を切ったら最大のプロセスを止める（WSL を落とさないため。
# 止まった鎖の残りは再起動で続きから）
( while true; do
    av=$(free -g | awk '/Mem:/{print $7}')
    echo "$(date -Iseconds) $(free -g | awk '/Mem:/{print "used="$3"G avail="$7"G"}') $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null) $(ps -eo rss,args --sort=-rss | awk '/all_models.py/ && !/awk/{printf "%.1fG:%s/%s ", $1/1048576, $(NF-3), $(NF-2)}' | head -c 300)"
    if [ "$av" -lt 5 ]; then
      pid=$(ps -eo pid,rss,args --sort=-rss | awk '/all_models.py/ && !/awk/{print $1; exit}')
      echo "$(date -Iseconds) WATCHDOG avail=${av}G kill $pid"; kill $pid
    fi
    sleep 60; done ) >> $L/mem.log 2>&1 &
MEM=$!
wait %1 %2 %3 %4 %5
kill $MEM
echo "ALL_DONE $(date -Iseconds)"
