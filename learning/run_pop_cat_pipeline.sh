#!/usr/bin/env bash
# 人気 + カテゴリ変数の派生系統（conf_pop_cat、平地）の完全な再学習と OOS 評価（2026-10-07）。
# 手順は run_pop_pipeline.sh と同じ（HPO なし・特徴量選択 null 15 回・fold チェックポイント）。
# 評価用 fit-final の --tabm-batch 256 は比較対象（人気系統の評価用モデル）に揃えるため。
set -uo pipefail
cd /home/machine_learner/HourseRacing_Local/learning
export PYTHONPATH=src NAR_CONF_DIR=conf_pop_cat
D="file://$PWD/data_real"
PY=.venv/bin/python
ART=./artifacts/pop_cat
TABM_LEARN="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 4096"
TABM_FINAL="--tabm-epochs 3 --tabm-k 4 --tabm-hidden 128 --tabm-layers 2 --tabm-batch 256"
stage() { echo "PIPELINE_STAGE $1 $(date -Iseconds)"; }

if [ ! -f data_real/gold/features_noodds_pop_cat/content_hash.txt ]; then
  stage features_start
  $PY -m nar.cli --data-root "$D" features || { echo "PIPELINE_FAILED features"; exit 1; }
fi
stage learn_start
$PY -m nar.cli --data-root "$D" --artifacts $ART learn --models clogit,lgbm,tabm,bayes \
  $TABM_LEARN --null-runs 15 --seed 0 --checkpoint-folds || { echo "PIPELINE_FAILED learn"; exit 1; }
stage fit_final_start
$PY -m nar.cli --data-root "$D" --artifacts $ART fit-final --out $ART/final_eval \
  $TABM_FINAL --null-runs 15 --seed 0 || { echo "PIPELINE_FAILED fit_final"; exit 1; }
stage evaluate_final_start
$PY -m nar.cli --data-root "$D" --artifacts $ART evaluate-final --final-dir $ART/final_eval \
  --unlock-oos --reason "2026-10-07 人気+カテゴリ変数の派生系統（flat_pop_cat）の配布モデル最終評価（ユーザー依頼の再学習）"
echo "PIPELINE_STAGE evaluate_final_done rc=$? $(date -Iseconds)"
echo "PIPELINE_ALL_DONE $(date -Iseconds)"
