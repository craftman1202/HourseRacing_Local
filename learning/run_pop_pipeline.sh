#!/usr/bin/env bash
# 人気順を足した派生系統（conf_pop / conf_banei_pop）の完全な再学習と OOS 評価。
# run_full_pipeline.sh と同じ手順・同じ TabM 設定。違いは NAR_CONF_DIR と出力先だけ。
# 評価用 fit-final の --tabm-batch 256 は、比較対象の現行評価モデル
# （final_eval_new / final_banei_eval_new）が当時のバグで 256 で学習されているのに揃えるため。
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src
D="file://$PWD/data_real"
PY=.venv/bin/python
TABM_LEARN="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 4096"
TABM_FINAL="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 256"
REASON="2026-10-02 人気順を特徴量に足した派生系統（%s）の配布モデル最終評価（ユーザー承認済み・1回）"

stage() { echo "PIPELINE_STAGE $1 $(date -Iseconds)"; }
run() {  # name conf artifacts final_dir gold_subdir
  local name=$1 conf=$2 art=$3 final=$4
  # 再開用: gold が既にあれば作り直さない（gold は決定的なので同じ content_hash になる）
  local gold="data_real/gold/$5/features.parquet"
  if [ -f "$gold" ] && [ -f "$(dirname "$gold")/content_hash.txt" ]; then
    stage "${name}_features_skipped"
  else
    stage "${name}_features_start"
    NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" features || { echo "PIPELINE_FAILED ${name}_features"; return 1; }
  fi
  stage "${name}_learn_start"
  NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" learn \
    --models clogit,lgbm,tabm,bayes $TABM_LEARN --seed 0 || { echo "PIPELINE_FAILED ${name}_learn"; return 1; }
  stage "${name}_fit_final_start"
  NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" fit-final \
    --out "$final" $TABM_FINAL --seed 0 || { echo "PIPELINE_FAILED ${name}_fit_final"; return 1; }
  stage "${name}_evaluate_final_start"
  NAR_CONF_DIR=$conf $PY -m nar.cli --data-root "$D" --artifacts "$art" evaluate-final \
    --final-dir "$final" --unlock-oos --reason "$(printf "$REASON" "$name")"
  echo "PIPELINE_STAGE ${name}_evaluate_final_done rc=$? $(date -Iseconds)"
}

run flat_pop conf_pop ./artifacts/pop ./artifacts/final_pop_eval features_noodds_pop
run banei_pop conf_banei_pop ./artifacts/banei_pop ./artifacts/final_banei_pop_eval features_noodds_banei_pop
echo "PIPELINE_ALL_DONE $(date -Iseconds)"
