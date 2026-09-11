"""追加特徴量が Null Importance のしきい値を超えるかだけを安価に確かめる。

`nar learn` の特徴量選択（設計書 §9.3）は fold ごとに Null Importance 15 回 +
RFE 33 回の LightGBM 再学習を行う。実測で fold あたり約3時間かかり、5 fold では
15 時間規模になる。ここで答えたい問いは「2026-09 に足した3列がノイズより有意か」
だけなので、第1段（Null Importance）を**直近期間のレース部分集合**に対して回す。

部分集合にするのは統計的なスクリーニングとして妥当な範囲での妥協で、
本番の特徴量選択の代わりにはならない。順位の目安を出すためだけに使う。

実行:
    cd learning && PYTHONPATH=src .venv/bin/python scripts/new_feature_screen.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nar.config import feature_config
from nar.features.builder import asof_features
from nar.features.selection import null_importance
from nar.train.pipeline import prepare, trainable

GOLD = Path("data_real/gold/features_noodds/features.parquet")
OUT = Path("artifacts/new_feature_screen.json")
NEW = ("h_pace_bal_last3", "jt_starts_prior", "jt_winrate_wilson")
N_RACES = 60_000
N_RUNS = 10


def main() -> int:
    fcfg = feature_config()
    feat = trainable(pd.read_parquet(GOLD))
    cols = [c for c in asof_features(fcfg) if c in feat.columns]

    # 直近側から N_RACES レースを取る。ペースバランスは 2010 年以降にしか
    # 実質値が入らないので、1998 年から一様に取ると評価が薄まる。
    races = feat[["race_id", "race_date"]].drop_duplicates().sort_values("race_date")
    keep = set(races["race_id"].tail(N_RACES))
    d = feat[feat["race_id"].isin(keep)]
    span = (d["race_date"].min(), d["race_date"].max())
    print(f"対象 {len(d):,} 行 / {d['race_id'].nunique():,} レース "
          f"（{span[0]} 〜 {span[1]}）, 候補 {len(cols)} 列, null {N_RUNS} 回")

    ni = null_importance(prepare(d.copy(), cols), cols, n_runs=N_RUNS, seed=0)
    ni = ni.sort_values("actual_gain", ascending=False).reset_index(drop=True)
    ni["rank"] = ni.index + 1

    print(f"\n生存 {int(ni['survives'].sum())} / {len(ni)} 列\n")
    print("=== 追加した3列 ===")
    show = ni[ni["feature"].isin(NEW)]
    print(show[["rank", "feature", "actual_gain", "null_p95", "survives"]]
          .to_string(index=False))
    print("\n=== 上位10列 ===")
    print(ni.head(10)[["rank", "feature", "actual_gain", "null_p95", "survives"]]
          .to_string(index=False))

    OUT.write_text(json.dumps({
        "setting": {"n_races": int(d["race_id"].nunique()), "n_rows": int(len(d)),
                    "period": [str(span[0]), str(span[1])], "n_null_runs": N_RUNS,
                    "note": "本番の特徴量選択（fold 内 Null Importance + RFE）の"
                            "代替ではない。追加列の順位の目安を出すためのスクリーニング"},
        "n_survivors": int(ni["survives"].sum()), "n_candidates": int(len(ni)),
        "new_features": show.to_dict(orient="records"),
        "all": ni.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
