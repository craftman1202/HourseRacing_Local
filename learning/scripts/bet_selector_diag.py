"""bet_selector の結果の頑健性チェック（2026-10-13）。

  1. 選ばれたレースの fold ごと・OOS の ROI（各年が独立したアウトオブサンプル）
  2. 上位の高額払戻を k 件除いた ROI（少数の大穴だけで出ていないか）
  3. 選ばれたレースの中身（頭数・軸の確率・点数）
  4. 帰無: 的中/払戻をレース間でシャッフルした目的変数で同じ手順を回したときの OOS ROI
     （手順そのものが偶然で >1 を作れるかどうか。20 回）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bet_selector as B  # noqa: E402
import roi_search as R  # noqa: E402

OUT = B.OUT


def per_period(d: pd.DataFrame, tau: float) -> dict:
    s = d[d["pred"] >= tau]
    out = {}
    for f, g in s.groupby("fold"):
        out["OOS" if f == 0 else f"fold{int(f)}"] = (round(g["ret"].sum() / g["stake"].sum(), 3), len(g))
    return out


def trimmed(s: pd.DataFrame, k: int) -> float:
    r = s["ret"].sort_values(ascending=False).iloc[k:].sum()
    return float(r / s["stake"].sum())


def main() -> int:
    pr = pd.read_parquet(OUT / "predictions.parquet")
    cmp_ = pd.read_csv(OUT / "comparison.csv")
    gold = pd.read_parquet(R.ROOT / "data_real/gold/features_noodds_pop_cat/features.parquet",
                           columns=["race_id", "field_size"]).drop_duplicates("race_id").set_index("race_id")
    rows = []
    for c in cmp_.itertuples():
        d = pr[(pr["structure"] == c.買い方) & (pr["lam"] == c.λ)].join(gold, on="race_id")
        d = d[d["fold"].between(2, 5) | (d["fold"] == 0)]
        tau = c._4                                  # メタ τ
        s = d[(d["pred"] >= tau) & (d["fold"] == 0)]
        rows.append({
            "券種": c.券種, "買い方": c.買い方, "λ": c.λ, "τ": tau,
            "期間別 ROI(件数)": per_period(d, tau),
            "OOS": round(c._8, 3),
            "OOS 上位1件除く": round(trimmed(s, 1), 3) if len(s) > 1 else np.nan,
            "OOS 上位5件除く": round(trimmed(s, 5), 3) if len(s) > 5 else np.nan,
            "選択レース 頭数中央値": float(s["field_size"].median()) if len(s) else np.nan,
            "選択レース 軸確率中央値": float(s["p1"].median()) if len(s) else np.nan,
            "全OOS 頭数中央値": float(d.loc[d.fold == 0, "field_size"].median()),
        })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "diag.csv", index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120)
    print(res.to_string(index=False))
    return 0


def null_test(bet: str, structure: str, lam: float, n_rep: int = 20) -> None:
    """払戻をレース間でシャッフル（fold 内）して、同じ手順の OOS ROI がどこまで出るか。"""
    win = R.load_probs()
    attrs = pd.read_parquet(R.OUT / "race_attrs.parquet")
    gold = pd.read_parquet(R.ROOT / "data_real/gold/features_noodds_pop_cat/features.parquet",
                           columns=["race_id", "field_size", "distance"]).drop_duplicates("race_id").set_index("race_id")
    f = pd.read_parquet(OUT / f"race_features_lam{lam}.parquet")
    rl = pd.read_parquet(R.OUT / f"race_level_lam{lam}.parquet")
    rl = rl[rl["structure"] == structure].drop(columns=["p1"]).set_index("race_id")
    d = rl.join(f, how="inner").join(attrs[["fold", "race_date", "一致"]]).join(gold)
    d["p_hit"] = d[f"phit::{structure}"]
    d["track"] = d.index.str[:2].astype(int)
    d["month"] = pd.to_datetime(d["race_date"]).dt.month
    d["agree"] = (d["一致"] == "一致").astype(int)
    d["stake"] = d["n"] * 100.0
    d = d.reset_index()
    rng = np.random.default_rng(0)
    out = []
    for rep in range(n_rep):
        e = d.copy()
        # 払戻率をレース間でシャッフル（fold 内）。特徴量と結果の関係だけを壊す
        e["rr"] = e.groupby("fold")["ret"].transform(lambda s: s.to_numpy()[rng.permutation(len(s))]) / e["stake"]
        e["ret"] = e["rr"] * e["stake"]
        e["hit"] = (e["ret"] > 0).astype(int)
        pred = pd.Series(np.nan, index=e.index)
        for k in (2, 3, 4, 5, 0):
            tr = e[(e["fold"] > 0) & (e["fold"] < k)] if k else e[e["fold"] > 0]
            te = e[e["fold"] == k]
            pred[te.index] = B.fit_predict(tr, te)
        e["pred"] = pred
        cv, oos = e[e["fold"].between(2, 5)], e[e["fold"] == 0]
        best = None
        for tau in np.round(np.arange(0.6, 2.01, 0.05), 2):
            roi, n = B.roi_of(cv[cv["pred"] >= tau])
            if n >= B.MIN_RACES * 4 and (best is None or roi > best[1]):
                best = (tau, roi, n)
        r, n = B.roi_of(oos[oos["pred"] >= best[0]])
        out.append({"rep": rep, "tau": best[0], "cv_roi": best[1], "oos_roi": r, "oos_races": n})
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(OUT / f"null_{bet}_{lam}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "null":
        null_test(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 20)
    else:
        raise SystemExit(main())
