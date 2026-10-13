"""roi_search.py の集計・選択・報告（2026-10-13）。

候補 = (λ, 買い方, 軸の確信度 x, セグメント)。全候補について fold1〜5 と OOS の ROI を出し、
  - 各 fold と OOS に 30 レース以上ある候補だけを残す
  - 券種ごとに「5-fold 平均 ROI 最大」「5-fold 最小 ROI 最大（頑健）」を選び、OOS を1回だけ見る
  - 施策を段階的に足したときの効果（買い方だけ → +確信度 → +確率の混合 → +セグメント）
  - 選択バイアスの大きさ（fold 平均 > 1 の候補のうち OOS も > 1 の割合、順位相関）
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402

OUT = R.OUT
SEGS = ("頭数", "競馬場", "クラス", "馬場", "距離", "一致")
MIN_RACES = 30
N_BOOT = 2000


def candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    attrs = pd.read_parquet(OUT / "race_attrs.parquet")
    rl = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(OUT / "race_level_lam*.parquet")))],
                   ignore_index=True)
    rl = rl.join(attrs, on="race_id")
    rl["stake"] = rl["n"] * 100.0
    rl["hit"] = (rl["ret"] > 0).astype(int)
    rl["bet"] = rl["structure"].map({k: v[0] for k, v in R.structures().items()})
    parts = []
    for x in R.X_GRID:
        d = rl[rl["p1"] >= x]
        for seg in (None, *SEGS):
            keys = ["lam", "structure", "bet", "fold"] + ([seg] if seg else [])
            g = d.groupby(keys, observed=True).agg(stake=("stake", "sum"), ret=("ret", "sum"),
                                                  races=("race_id", "size"), hits=("hit", "sum")).reset_index()
            g["x"] = x
            g["seg"] = seg or "全体"
            g["seg_value"] = g[seg].astype(str) if seg else "全体"
            parts.append(g.drop(columns=[seg]) if seg else g)
    long = pd.concat(parts, ignore_index=True)
    long["roi"] = long["ret"] / long["stake"]
    long["hit_rate"] = long["hits"] / long["races"]
    return long, rl


def wide(long: pd.DataFrame) -> pd.DataFrame:
    key = ["lam", "structure", "bet", "x", "seg", "seg_value"]
    f = long[long["fold"] > 0]
    roi = f.pivot_table(index=key, columns="fold", values="roi")
    roi.columns = [f"fold{int(c)}" for c in roi.columns]
    races = f.pivot_table(index=key, columns="fold", values="races")
    agg = f.groupby(key).agg(fold_mean_roi=("roi", "mean"), fold_min_roi=("roi", "min"),
                             folds_gt1=("roi", lambda s: int((s > 1).sum())),
                             fold_hit=("hit_rate", "mean"), n_folds=("roi", "size"))
    agg["fold_min_races"] = races.min(axis=1)
    o = long[long["fold"] == 0].set_index(key)[["roi", "hit_rate", "races", "stake"]].rename(
        columns={"roi": "oos_roi", "hit_rate": "oos_hit", "races": "oos_races", "stake": "oos_stake"})
    w = roi.join(agg).join(o).reset_index()
    ok = (w["n_folds"] == 5) & (w["fold_min_races"] >= MIN_RACES) & (w["oos_races"] >= MIN_RACES)
    return w[ok].reset_index(drop=True)


def boot_ci(rl: pd.DataFrame, c: pd.Series) -> tuple[float, float]:
    m = (rl["lam"] == c["lam"]) & (rl["structure"] == c["structure"]) & (rl["p1"] >= c["x"]) & (rl["fold"] == 0)
    if c["seg"] != "全体":
        m &= rl[c["seg"]].astype(str) == c["seg_value"]
    d = rl[m].groupby("race_date").agg(ret=("ret", "sum"), stake=("stake", "sum"))
    r, s = d["ret"].to_numpy(), d["stake"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    bs = r[idx].sum(1) / s[idx].sum(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main() -> int:
    long, rl = candidates()
    w = wide(long)
    w.to_parquet(OUT / "candidates.parquet", index=False)
    w.to_csv(OUT / "candidates.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 30)
    cols = ["structure", "lam", "x", "seg", "seg_value", "fold_mean_roi", "fold_min_roi", "folds_gt1",
            "fold_hit", "oos_roi", "oos_hit", "oos_races"]

    rows, steps = [], []
    for bet, g in w.groupby("bet"):
        jp = R.BET_JP[bet]
        stages = {
            "① 買い方だけ（本番モデル・絞りなし）": g[(g.lam == 0) & (g.x == 0) & (g.seg == "全体")],
            "② + 軸の確信度": g[(g.lam == 0) & (g.seg == "全体")],
            "③ + 確率の混合（λ）": g[g.seg == "全体"],
            "④ + レース選び（セグメント）": g,
        }
        for st, s in stages.items():
            b = s.loc[s["fold_mean_roi"].idxmax()]
            steps.append({"券種": jp, "段階": st, "候補数": len(s), "選ばれた候補": f"{b.structure} / λ={b.lam} / x≥{b.x} / {b.seg}={b.seg_value}",
                          "5fold平均": b.fold_mean_roi, "5fold最小": b.fold_min_roi, "OOS": b.oos_roi})
        for crit in ("fold_mean_roi", "fold_min_roi"):
            b = g.loc[g[crit].idxmax()]
            lo, hi = boot_ci(rl, b)
            rows.append({"券種": jp, "選び方": "5fold平均最大" if crit == "fold_mean_roi" else "5fold最小最大",
                         **{k: b[k] for k in cols}, "oos_ci_lo": lo, "oos_ci_hi": hi})
    sel = pd.DataFrame(rows)
    st = pd.DataFrame(steps)
    sel.to_csv(OUT / "selected.csv", index=False)
    st.to_csv(OUT / "steps.csv", index=False)

    bias = []
    for bet, g in w.groupby("bet"):
        hi = g[g["fold_mean_roi"] > 1]
        bias.append({"券種": R.BET_JP[bet], "候補数": len(g), "fold平均>1": len(hi),
                     "うちOOS>1": int((hi["oos_roi"] > 1).sum()),
                     "OOS>1（全候補）": int((g["oos_roi"] > 1).sum()),
                     "順位相関": g["fold_mean_roi"].corr(g["oos_roi"], method="spearman"),
                     "fold平均上位20のOOS平均": g.nlargest(20, "fold_mean_roi")["oos_roi"].mean()})
    bias = pd.DataFrame(bias)
    bias.to_csv(OUT / "selection_bias.csv", index=False)

    print("=== 施策を段階的に足したときの最良（5fold平均で選択） ===")
    print(st.round(3).to_string(index=False))
    print("\n=== 券種ごとの選択結果 ===")
    print(sel.round(3).to_string(index=False))
    print("\n=== 選択バイアス ===")
    print(bias.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
