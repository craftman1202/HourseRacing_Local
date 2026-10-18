#!/usr/bin/env bash
# 楽天競馬の自動操作を実行する入口（許可ルールを1本で書けるようにするためのラッパー）。
#   scripts/rakuten.sh scripts/rakuten_probe.py <出馬表URL> [...]   投票画面の記録（購入しない）
#   scripts/rakuten.sh -m narops.orders.runner [--once]            発注ランナー
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
exec ../learning/.venv/bin/python -u "$@"
