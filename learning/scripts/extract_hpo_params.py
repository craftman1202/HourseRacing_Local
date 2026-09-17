#!/usr/bin/env python3
"""`nar learn` が書き出す artifacts/hpo.json から、fit-final に渡す
hpo_params.json を作る。

nested HPO は fold ごとに別々の best_params を持つ（外側5-foldそれぞれの
内側3-foldで探索し直す設計、CV-09）。本番モデル1つを作るには集約が要る。
ここでは最新（最大 fold 番号）の外側 fold の best_params を採用する——
walk-forward は expanding window なので、最終 fold の学習期間が最も長く
現在の regime に最も近い。

使い方:
    python scripts/extract_hpo_params.py artifacts/hpo.json artifacts/hpo_params.json
"""
import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"使い方: {sys.argv[0]} <hpo.json> <出力先>", file=sys.stderr)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    folds = json.loads(open(src, encoding="utf-8").read())
    if not folds:
        print(f"ERROR: {src} が空です。HPO が1件も実行されていません。", file=sys.stderr)
        return 1
    latest = max(folds, key=lambda f: f["fold"])
    out = {}
    for model in ("clogit", "lgbm", "tabm"):
        if model in latest:
            out[model] = latest[model]["best_params"]
    if not out:
        print(f"ERROR: fold {latest['fold']} に既知モデルの HPO 結果がありません。",
              file=sys.stderr)
        return 1
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"fold {latest['fold']}（最新）の best_params を採用: {list(out)}")
    print(f"→ {dst}")
    for model, params in out.items():
        print(f"  {model}: {params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
