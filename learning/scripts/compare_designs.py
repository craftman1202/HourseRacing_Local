"""旧設計（2026-08-27）と新設計の OOF 予測を対応のある形で比較する。

設計書 §9.5 のとおり、単一の点推定でモデルを選ばない。開催日をブロックとする
ブロックブートストラップで、**同一レース上の**レース内 NLL の差に信頼区間を付ける。

対応のある比較にするのが要点。旧新で評価対象レースが1件でもずれていると、
差が「モデルの差」なのか「対象集合の差」なのか分からなくなる。共通レースだけに
揃えてから差を取る。

実行:
    cd learning && PYTHONPATH=src .venv/bin/python scripts/compare_designs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nar.eval.metrics import EPS, block_bootstrap_ci

OLD = Path("artifacts/baseline_2026-08-27/oof_predictions.parquet")
NEW = Path("artifacts/oof_predictions.parquet")
OUT = Path("artifacts/design_comparison.json")
N_ITER = 10_000


def per_race_nll(df: pd.DataFrame, col: str) -> pd.Series:
    """レースごとの −log p(勝ち馬)。レース単位の指標なので行ではなくレースで集計する。"""
    w = df[df["is_win"] == 1]
    return pd.Series(-np.log(np.clip(w[col].to_numpy(), EPS, 1.0)),
                     index=w["race_id"].to_numpy())


def main() -> int:
    if not NEW.exists():
        print(f"{NEW} がありません。先に `nar learn` を完走させてください。")
        return 1
    old = pd.read_parquet(OLD)
    new = pd.read_parquet(NEW)
    models = [m for m in ("lgbm", "tabm", "clogit", "bayes", "ensemble_stacked")
              if m in old.columns and m in new.columns]
    print(f"旧 {len(old):,} 行 / 新 {len(new):,} 行 / 共通モデル {models}")

    # 共通の (race_id, horse_no) に揃える。特徴量が増えて学習対象行が変わりうるので、
    # ここを揃えないと差の意味が変わる。
    key = ["race_id", "horse_no"]
    common = old[key].merge(new[key], on=key, how="inner")
    o = old.merge(common, on=key, how="inner").sort_values(key).reset_index(drop=True)
    n = new.merge(common, on=key, how="inner").sort_values(key).reset_index(drop=True)
    assert (o["is_win"].to_numpy() == n["is_win"].to_numpy()).all(), "ラベルが食い違います"
    races = o["race_id"].nunique()
    print(f"共通 {len(o):,} 行 / {races:,} レース\n")

    day = (o[o["is_win"] == 1].set_index("race_id")["race_date"]
           .astype(str).str.slice(0, 10))

    rows = []
    for m in models:
        a = per_race_nll(o, m)
        b = per_race_nll(n, m)
        idx = a.index.intersection(b.index)
        diff = (b.loc[idx] - a.loc[idx])          # 新 − 旧。負なら新設計が良い
        blocks = day.reindex(idx)
        point, lo, hi = block_bootstrap_ci(diff, blocks, n_iter=N_ITER, seed=0)
        rows.append({
            "model": m, "旧NLL": float(a.loc[idx].mean()), "新NLL": float(b.loc[idx].mean()),
            "差(新-旧)": point, "CI下限": lo, "CI上限": hi,
            "改善": bool(hi < 0), "悪化": bool(lo > 0),
        })
        print(f"{m:18s} 旧 {a.loc[idx].mean():.5f} → 新 {b.loc[idx].mean():.5f} / "
              f"差 {point:+.5f} [95%CI {lo:+.5f}, {hi:+.5f}]"
              f"{'  ← 改善（CIが0を跨がない）' if hi < 0 else ('  ← 悪化' if lo > 0 else '  ← 有意差なし')}")

    df = pd.DataFrame(rows)
    OUT.write_text(json.dumps({
        "n_races": int(races), "n_rows": int(len(o)), "n_iter": N_ITER,
        "note": "新 − 旧。負なら新設計が良い。開催日ブロックのブロックブートストラップ"
                "（設計書 §9.5）。比較モデル数だけ同時に見ているので、"
                "個別のCIは多重性の補正を受けていない点に注意",
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}")
    print(f"\n注: {len(models)} モデルを同時に見ているため、個別の 95%CI は"
          f"多重比較の補正を受けていない。主指標は lgbm と ensemble_stacked。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
