"""券種ごとに「CI 中央値 ≥ 1 かつ CI 下限が最大」の戦略を選ぶ（2026-10-20）。

候補（的中確率の足切りを含む）
  EV 系   予測払戻 × 確率 の EV ≥ t、的中確率（組の p_ens）≥ p_min、全部買う(all) / レースで EV 最大の1点(top1)
          t ∈ T_GRID, p_min ∈ P_GRID。対象: 単勝・複勝・馬連・馬単・ワイド・3連複・3連単（payout_ev の組ごとの表）
  無オッズ系  買い方（軸流し・BOX など）× 人気との混合 λ × 軸の確率 ≥ x（strategy_select のレース単位の表）。枠連・枠単もここだけ
評価
  選択期間 fold 2〜5（2020〜2023）をまとめ、開催日単位のブロックブートストラップ（B 回）で ROI の分布を作る。
  条件: ブートストラップ中央値 ≥ 1 かつ ベット 200 レース以上。条件を満たすものの中で CI 下限（2.5%）が最大。満たすものが無ければ「採用なし」。
  OOS（2024-02〜）は選択に使わず、選んだ戦略の確認だけに使う（OOS は 2026-10-17 に一度開封済みなので、真の未見データではない）。
出力: artifacts/ci_select/{candidates_<券種>.csv, selected.csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402
import strategy_select as S  # noqa: E402

ROOT = R.ROOT
OUT = ROOT / "artifacts/ci_select"
PAYOUT = ROOT / "artifacts/payout_ev"
OOSD = ROOT / "artifacts/strategy_ens/oos"
EVBETS = ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta")
T_GRID = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
P_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
B = 3000
MIN_RACES = 200
FOLDS = (2, 3, 4, 5)


def day_map() -> pd.Series:
    a = pd.read_parquet(ROOT / "artifacts/all_models/pop/oof_all.parquet", columns=["race_id", "race_date"])
    b = pd.read_parquet(OOSD / "oos_ensemble.parquet", columns=["race_id", "race_date"])
    m = pd.concat([a, b]).drop_duplicates("race_id").set_index("race_id")["race_date"]
    return pd.to_datetime(m)


def boot(day_ret: np.ndarray, day_stake: np.ndarray, rng) -> tuple[float, float, float, float]:
    """日単位ブートストラップ。ROI の (点推定, 中央値, 2.5%, 97.5%)。"""
    n = len(day_ret)
    idx = rng.integers(0, n, size=(B, n))
    bs = day_ret[idx].sum(1) / day_stake[idx].sum(1)
    return (float(day_ret.sum() / day_stake.sum()), float(np.median(bs)),
            float(np.quantile(bs, .025)), float(np.quantile(bs, .975)))


def summarize(df: pd.DataFrame, days: pd.Series, stake_per_ticket: float, rng) -> dict | None:
    """df: 選ばれたチケット（race_id, ret[払戻の倍率 or 円], n=点数）。"""
    if df.empty:
        return None
    g = df.assign(day=df["race_id"].map(days)).groupby("day").agg(ret=("ret", "sum"), n=("n", "sum"))
    races = df["race_id"].nunique()
    if races < MIN_RACES or len(g) < 20:
        return None
    pt, med, lo, hi = boot(g["ret"].to_numpy(float), g["n"].to_numpy(float) * stake_per_ticket, rng)
    hit = float(df.groupby("race_id")["ret"].sum().gt(0).mean())
    return {"roi": pt, "boot_median": med, "ci_lo": lo, "ci_hi": hi, "races": races,
            "tickets": int(df["n"].sum()), "days": len(g), "hit_rate": hit}


def ev_family(bt: str, rng, days: pd.Series) -> pd.DataFrame:
    d = pd.read_parquet(PAYOUT / f"combos_{bt}.parquet", columns=["race_id", "p_ens", "ev", "ret", "fold"])
    d = d[(d["ev"] >= min(T_GRID)) & d["fold"].isin(FOLDS)]
    rows = []
    for t in T_GRID:
        for p in P_GRID:
            s = d[(d["ev"] >= t) & (d["p_ens"] >= p)]
            if s.empty:
                continue
            for mode in ("top1", "all"):
                sel = s.sort_values("ev", ascending=False).drop_duplicates("race_id") if mode == "top1" else s
                sel = sel.assign(n=1)
                r = summarize(sel[["race_id", "ret", "n"]], days, 1.0, rng)
                if r:
                    per_fold = sel.groupby("fold").apply(lambda x: x["ret"].sum() / len(x), include_groups=False)
                    rows.append({"family": "EV", "t": t, "p_min": p, "mode": mode, **r,
                                 "min_fold_roi": float(per_fold.min()), "folds": len(per_fold)})
    return pd.DataFrame(rows)


def free_family(bt: str, rng, days: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    parts = []
    for lam in R.LAMBDAS:
        rl = pd.read_parquet(S.OUT / f"race_level_lam{lam}.parquet")
        ex = S.OUT / f"race_level_extra_lam{lam}.parquet"
        if ex.exists():
            rl = pd.concat([rl, pd.read_parquet(ex)], ignore_index=True)
        rl = rl[rl["structure"].map({k: v[0] for k, v in R.structures().items()}) == bt]
        parts.append(rl.assign(lam=lam))
    rl = pd.concat(parts, ignore_index=True).join(meta[["fold"]], on="race_id")
    rl = rl[rl["fold"].isin(FOLDS)]
    for (name, lam), g in rl.groupby(["structure", "lam"]):
        for x in R.X_GRID:
            s = g[g["p1"] >= x]
            r = summarize(s[["race_id", "ret", "n"]], days, 100.0, rng)
            if r:
                pf = s.groupby("fold").apply(lambda z: z["ret"].sum() / (100.0 * z["n"].sum()), include_groups=False)
                rows.append({"family": "無オッズ", "structure": name, "lam": lam, "x": x, **r,
                             "min_fold_roi": float(pf.min()), "folds": len(pf)})
    return pd.DataFrame(rows)


def pick(c: pd.DataFrame) -> pd.Series | None:
    ok = c[(c["boot_median"] >= 1.0)]
    return None if ok.empty else ok.sort_values("ci_lo", ascending=False).iloc[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    days = day_map()
    win = S.load_win()
    meta = win.drop_duplicates("race_id").set_index("race_id")[["fold", "race_date"]]
    sel_rows = []
    for bt in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta", "bracket_q", "bracket_e"):
        parts = []
        if bt in EVBETS:
            parts.append(ev_family(bt, rng, days))
        parts.append(free_family(bt, rng, days, meta))
        c = pd.concat([p for p in parts if len(p)], ignore_index=True)
        c.to_csv(OUT / f"candidates_{bt}.csv", index=False)
        best = pick(c)
        n_ok = int((c["boot_median"] >= 1.0).sum())
        top_by_lo = c.sort_values("ci_lo", ascending=False).iloc[0]
        rec = {"bet": R.BET_JP[bt], "候補数": len(c), "中央値≥1の数": n_ok,
               "全候補の最大CI下限": float(top_by_lo["ci_lo"])}
        if best is not None:
            rec.update({k: best.get(k) for k in ("family", "structure", "lam", "x", "t", "p_min", "mode", "roi",
                                                 "boot_median", "ci_lo", "ci_hi", "races", "tickets", "hit_rate",
                                                 "min_fold_roi")})
        sel_rows.append(rec)
        print(R.BET_JP[bt], "候補", len(c), "中央値≥1", n_ok, "→", "採用なし" if best is None else
              f"{best['family']} ROI {best['roi']:.3f} CI[{best['ci_lo']:.3f},{best['ci_hi']:.3f}] "
              f"{best.get('structure', '')} λ={best.get('lam')} x={best.get('x')} t={best.get('t')} "
              f"p>={best.get('p_min')} {best.get('mode', '')} 賭{best['races']}R", flush=True)
    pd.DataFrame(sel_rows).to_csv(OUT / "selected.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
