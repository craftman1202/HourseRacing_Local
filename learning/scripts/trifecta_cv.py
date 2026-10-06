"""3連単・3連複の全ての買い方について、walk-forward 5-fold と OOS の成績（2026-10-06）。

買い方は trifecta_strategies.py と同じ 52 構造 × 軸の単勝確率の下限 5 段階（= 260 通り）。
オッズを使う「期待値で絞る」買い方は 3連のオッズが 2026-02〜しか無いので対象外。

期間: fold1〜5 = 2019〜2023 の各年（walk-forward の valid）、OOS = 2024-02-01〜（全体を1期間）。
確率の出所（すべて保存済み、OOS を開封しない）:
  dev_reg  本番の乖離回帰（fold: deviation_models.py walkforward / OOS: 評価用モデル）
  popfeat  人気を特徴量に入れた単勝モデル（fold: artifacts/pop の OOF / OOS: 評価用モデル）
  base     人気を使わない単勝モデル（fold: artifacts の OOF / OOS: 評価用モデル）
  poponly  人気 × 頭数の勝率表だけ（fold ごとに学習区間で集計）
fold の popfeat / base は walk-forward の温度較正を valid 上で測っており、アンサンブル重みも
OOF 全体で推定しているので、fold の数字は僅かに楽観的（順位はほぼ変わらない）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifecta_strategies as T  # noqa: E402

ROOT = T.ROOT
OUT = ROOT / "artifacts/trifecta_cv"
DEV = ROOT / "artifacts/dev_flat_pop"


def _stacked(oof_path: Path, weights_csv: Path) -> pd.DataFrame:
    from nar.models.ensemble import ConstrainedStacker

    o = pd.read_parquet(oof_path)
    w = pd.read_csv(weights_csv)
    w = {str(r["model"]): float(r["weight"]) for _, r in w.iterrows()
         if r["weight"] > 1e-9 and r["model"] in o.columns}
    st = ConstrainedStacker(list(w))
    st.weights = np.array(list(w.values()))
    st.weights = st.weights / st.weights.sum()
    o["ensemble"] = st.predict_proba(o[list(w)], o["race_id"].to_numpy())
    return o[["race_id", "horse_no", "race_date", "fold", "ensemble"]]


def sources() -> dict:
    """名前 → (fold の予測, OOS の予測) を返す関数。指定したものだけ読む。"""
    def oos(path):
        d = pd.read_parquet(ROOT / path, columns=["race_id", "horse_no", "race_date", "ensemble"])
        return d.assign(fold=0)

    return {
        "dev_reg": lambda: (pd.read_parquet(DEV / "oof_win_reg.parquet"),
                            oos("artifacts/dev_flat_pop/oos_win_reg.parquet")),
        "poponly": lambda: (pd.read_parquet(DEV / "oof_win_pop.parquet"),
                            oos("artifacts/dev_flat_pop/oos_win_pop.parquet")),
        "popfeat": lambda: (_stacked(ROOT / "artifacts/pop/oof_predictions.parquet",
                                     ROOT / "artifacts/pop/ensemble_weights.csv"),
                            oos("artifacts/pop/oos_predictions.parquet")),
        "base": lambda: (_stacked(ROOT / "artifacts/oof_predictions.parquet",
                                  ROOT / "artifacts/ensemble_weights.csv"),
                         oos("artifacts/oos_predictions.parquet")),
    }


def run_model(name: str, folds: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    win = pd.concat([folds[["race_id", "horse_no", "race_date", "fold", "ensemble"]], oos],
                    ignore_index=True).rename(columns={"ensemble": "p"})
    win = win[win["race_id"].str[:2].astype(int) > 4]          # 平地のみ
    period = win.drop_duplicates("race_id").set_index("race_id")["fold"]
    dates = win.drop_duplicates("race_id").set_index("race_id")["race_date"]
    tri, trio = T.tickets(win)
    tri, trio = T.attach_payout(tri, "3連単"), T.attach_payout(trio, "3連複")
    rows = []
    for t in (tri, trio):
        t["fold"] = t["race_id"].map(period).to_numpy()
        t["race_date"] = pd.to_datetime(t["race_id"].map(dates))
    for sname, bet, p in T.strategies():
        t = tri if bet == "tri" else trio
        base = T.mask(t, bet, p)
        for x in T.X_GRID:
            sel = t[base & (t["p1"].to_numpy() >= x)]
            for f, g in sel.groupby("fold"):
                row = {"model": name, "strategy": sname, "bet": bet, "x": x,
                       "period": "oos" if f == 0 else f"fold{int(f)}", **T.summarize(g)}
                if f == 0:
                    row["ci_lo"], row["ci_hi"], _ = T.boot_ci(g)
                rows.append(row)
    return pd.DataFrame(rows)


def wide(long: pd.DataFrame) -> pd.DataFrame:
    key = ["model", "strategy", "bet", "x"]
    f = long[long["period"].str.startswith("fold")]
    roi = f.pivot_table(index=key, columns="period", values="roi")
    agg = f.groupby(key).agg(fold_mean_roi=("roi", "mean"), fold_min_roi=("roi", "min"),
                             fold_max_roi=("roi", "max"),
                             folds_roi_gt1=("roi", lambda s: int((s > 1).sum())),
                             fold_mean_hit=("race_hit", "mean"),
                             fold_races=("n_races", "sum"),
                             fold_stake=("stake", "sum"), fold_ret=("ret", "sum"))
    agg["fold_pooled_roi"] = agg["fold_ret"] / agg["fold_stake"]
    o = long[long["period"] == "oos"].set_index(key)[
        ["roi", "race_hit", "n_races", "tickets_per_race", "ci_lo", "ci_hi"]].rename(
        columns={"roi": "oos_roi", "race_hit": "oos_hit", "n_races": "oos_races"})
    out = roi.join(agg.drop(columns=["fold_stake", "fold_ret"])).join(o).reset_index()
    return out.sort_values(["model", "fold_mean_roi"], ascending=[True, False])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = sources()
    which = sys.argv[1:] or list(src)
    for name in which:
        long = run_model(name, *src[name]())
        long.to_csv(OUT / f"long_{name}.csv", index=False)
        print(f"{name}: {len(long)} 行")
    longs = [pd.read_csv(p) for p in sorted(OUT.glob("long_*.csv"))]
    w = wide(pd.concat(longs, ignore_index=True))
    w.to_csv(OUT / "all_strategies_5fold_oos.csv", index=False)
    pd.set_option("display.width", 250)
    for m, g in w.groupby("model"):
        print(f"\n#### {m}: 5-fold 平均 ROI 上位10")
        print(g.head(10)[["strategy", "x", "fold_mean_roi", "fold_min_roi", "folds_roi_gt1",
                          "fold_mean_hit", "oos_roi", "oos_hit", "oos_races"]].round(3).to_string(index=False))
        print(f"  fold 平均と OOS の順位相関: {g['fold_mean_roi'].corr(g['oos_roi'], method='spearman'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
