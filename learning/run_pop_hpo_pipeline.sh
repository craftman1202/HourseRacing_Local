#!/usr/bin/env bash
# 人気順派生系統（conf_pop / conf_banei_pop）を nested HPO（大: clogit100/lgbm100/tabm20）と
# 特徴量選択（null importance 15 回）付きで回し直し、評価用モデルで OOS を1回評価する。
# 手順は run_hpo_pipeline.sh と同じ。違いは試行数・null-runs・出力先。
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
D="file://$PWD/data_real"
PY=.venv/bin/python
TABM_BASE="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 4096"
HPO="--hpo-clogit 40 --hpo-lgbm 30 --hpo-tabm 6"   # 2026-10-03: 大(100/100/20)は fold1 だけで 8h 超のため中止し中に変更
REASON="2026-10-02 人気順派生系統（%s）・HPO中＋特徴量選択の配布モデル最終評価（ユーザー承認済み）"

stage() { echo "PIPELINE_STAGE $1 $(date -Iseconds)"; }
run() {  # name conf artifacts
  local name=$1 conf=$2 art=$3
  if [ ! -f "$art/hpo.json" ]; then
    stage "${name}_learn_start"
    NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" learn \
      --models clogit,lgbm,tabm,bayes $TABM_BASE $HPO --null-runs 15 --seed 0 \
      --tabm-hpo-space small --checkpoint-folds \
      || { echo "PIPELINE_FAILED ${name}_learn"; return 1; }
  else
    stage "${name}_learn_skipped"
  fi
  $PY scripts/extract_hpo_params.py "$art/hpo.json" "$art/hpo_params.json" \
    || { echo "PIPELINE_FAILED ${name}_extract"; return 1; }
  stage "${name}_fit_final_start"
  NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" fit-final \
    --out "$art/final_eval" $TABM_BASE --null-runs 15 --seed 0 \
    --hpo-params-json "$art/hpo_params.json" || { echo "PIPELINE_FAILED ${name}_fit_final"; return 1; }
  stage "${name}_evaluate_final_start"
  NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" evaluate-final \
    --final-dir "$art/final_eval" --unlock-oos --reason "$(printf "$REASON" "$name")"
  echo "PIPELINE_STAGE ${name}_evaluate_final_done rc=$? $(date -Iseconds)"
}

# メモリの推移を残す（2026-10-03 の OOM の再発確認用）
( while true; do echo "$(date -Iseconds) $(free -g | awk '/Mem:/{print "used="$3"G avail="$7"G"}') $(ps -eo rss,cmd --sort=-rss | awk '/nar.cli/{printf "rss=%.1fG", $1/1048576; exit}')"; sleep 600; done ) > artifacts/pop_hpo_mem.log 2>&1 &
MEMLOG=$!
trap 'kill $MEMLOG 2>/dev/null' EXIT

run flat_pop_hpo conf_pop ./artifacts/pop_hpo
run banei_pop_hpo conf_banei_pop ./artifacts/banei_pop_hpo
echo "PIPELINE_ALL_DONE $(date -Iseconds)"
