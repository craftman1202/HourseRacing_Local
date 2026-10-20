"""券種ごとに「CI 中央値 ≥ 1 かつ CI 下限が最大」の戦略を選ぶ（2026-10-20、グリッドを細かくした版）。

候補（的中確率の足切りを含む）
  EV 系   予測払戻 × 確率 の EV ≥ t、的中確率（組の p_ens）≥ p_min、全部買う(all) / レースで EV 最大の1点(top1)
          t = 1.0〜10.0 を 0.1 刻み、p_min = 0〜60% を 1% 刻み。対象: 単勝・複勝・馬連・馬単・ワイド・3連複・3連単
  無オッズ系  買い方（軸流し・BOX など）× 人気との混合 λ × 軸の確率 ≥ x（x = 0〜80% を 1% 刻み）。枠連・枠単もここだけ
評価
  選択期間 fold 2〜5（2020〜2023）をまとめ、開催日単位のブロックブートストラップで ROI の分布を作る。
  1) 全候補を B_SCREEN 回で評価 → 2) 条件（中央値 ≥ 1、ベット 200 レース以上）を満たすものの CI 下限の上位 TOP_REFINE 個を
  B_REFINE 回で再評価 → 3) その中で CI 下限が最大のものを採用。満たすものが無ければ「採用なし」。
  OOS（2024-02〜）は選択に使わない（strategy_ci_oos.py が確認だけする。OOS は 2026-10-17 に一度開封済み）。
注意: 候補が多いほど「CI 下限が最大」の選択は検証期間に過適合する（最良を選ぶ勝者の呪い）。OOS の確認で必ず見る。
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
T_GRID = np.round(np.arange(1.0, 10.0001, 0.1), 1)
P_GRID = np.round(np.arange(0.0, 0.6001, 0.01), 2)
X_GRID = np.round(np.arange(0.0, 0.8001, 0.01), 2)
B_SCREEN, B_REFINE, TOP_REFINE = 1000, 10000, 40
MIN_RACES = 200
MIN_DAYS = 20
FOLDS = (2, 3, 4, 5)


def day_map() -> pd.Series:
    a = pd.read_parquet(ROOT / "artifacts/all_models/pop/oof_all.parquet", columns=["race_id", "race_date"])
    b = pd.read_parquet(OOSD / "oos_ensemble.parquet", columns=["race_id", "race_date"])
    m = pd.concat([a, b]).drop_duplicates("race_id").set_index("race_id")["race_date"]
    return pd.to_datetime(m)


def boot(dr: np.ndarray, dn: np.ndarray, rng, b: int) -> tuple[float, float, float, float]:
    """日単位ブートストラップ。ROI の (点推定, 中央値, 2.5%, 97.5%)。dr = 日ごとの払戻、dn = 日ごとの賭け額。"""
    n = len(dr)
    idx = rng.integers(0, n, size=(b, n))
    bs = dr[idx].sum(1) / dn[idx].sum(1)
    return float(dr.sum() / dn.sum()), float(np.median(bs)), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


class Keeper:
    """中央値 ≥ 1 の候補のうち CI 下限の上位だけを、再評価用の配列つきで持つ。"""

    def __init__(self, k: int) -> None:
        self.k, self.items = k, []

    def add(self, lo: float, params: dict, dr: np.ndarray, dn: np.ndarray) -> None:
        if len(self.items) < self.k or lo > self.items[-1][0]:
            self.items.append((lo, params, dr, dn))
            self.items.sort(key=lambda z: -z[0])
            del self.items[self.k:]


def evaluate(dc, rc, ret, stake, fold, D, rng, extra: dict, keep: Keeper, rows: list) -> bool:
    """選ばれた行の配列から指標を出して rows に積む。ベットが足りなければ False（これより絞ると増えないので打ち切れる）。"""
    ur = np.unique(rc)
    if len(ur) < MIN_RACES:
        return False
    dr = np.bincount(dc, weights=ret, minlength=D)
    dn = np.bincount(dc, weights=stake, minlength=D)
    m = dn > 0
    if m.sum() < MIN_DAYS:
        return False
    dr, dn = dr[m], dn[m]
    pt, med, lo, hi = boot(dr, dn, rng, B_SCREEN)
    hit = float((np.bincount(rc, weights=ret) > 0).sum() / len(ur))
    pf = [float(ret[fold == f].sum() / stake[fold == f].sum()) for f in FOLDS if (fold == f).any()]
    rows.append({**extra, "roi": pt, "boot_median": med, "ci_lo": lo, "ci_hi": hi, "races": len(ur),
                 "tickets": float(stake.sum()), "days": int(m.sum()), "hit_rate": hit,
                 "min_fold_roi": min(pf), "folds": len(pf)})
    if med >= 1.0:
        keep.add(lo, extra, dr, dn)
    return True


def ev_family(bt: str, rng, days: pd.Series, keep: Keeper) -> list:
    d = pd.read_parquet(PAYOUT / f"combos_{bt}.parquet", columns=["race_id", "p_ens", "ev", "ret", "fold"])
    d = d[(d["ev"] >= T_GRID[0]) & d["fold"].isin(FOLDS)].sort_values("ev", ascending=False)
    rc, uni = pd.factorize(d["race_id"])
    dcodes, _ = pd.factorize(days.reindex(uni))
    dc_all = dcodes[rc]
    D = int(dcodes.max()) + 1
    ev, p, ret, fold = d["ev"].to_numpy(), d["p_ens"].to_numpy(), d["ret"].to_numpy(float), d["fold"].to_numpy()
    rows: list = []
    for pm in P_GRID:
        mk = p >= pm
        if mk.sum() < MIN_RACES:
            break
        e, r_, rt, fo, dcm = ev[mk], rc[mk], ret[mk], fold[mk], dc_all[mk]
        neg = -e
        for t in T_GRID:
            k = int(np.searchsorted(neg, -t, side="right"))          # ev >= t の先頭 k 行
            if k < MIN_RACES:
                break
            ur, first = np.unique(r_[:k], return_index=True)
            if len(ur) < MIN_RACES:
                break
            ok = evaluate(dcm[:k], r_[:k], rt[:k], np.ones(k), fo[:k], D, rng,
                          {"family": "EV", "t": float(t), "p_min": float(pm), "mode": "all"}, keep, rows)
            if first.size >= MIN_RACES:
                evaluate(dcm[first], r_[first], rt[first], np.ones(len(first)), fo[first], D, rng,
                         {"family": "EV", "t": float(t), "p_min": float(pm), "mode": "top1"}, keep, rows)
            if not ok:
                break
    return rows


def free_family(bt: str, rng, days: pd.Series, meta: pd.DataFrame, keep: Keeper) -> list:
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
    rows: list = []
    for (name, lam), g in rl.groupby(["structure", "lam"]):
        g = g.sort_values("p1", ascending=False)
        rc, uni = pd.factorize(g["race_id"])
        dcodes, _ = pd.factorize(days.reindex(uni))
        dc = dcodes[rc]
        D = int(dcodes.max()) + 1
        p1, ret, n, fold = g["p1"].to_numpy(), g["ret"].to_numpy(float), g["n"].to_numpy(float) * 100.0, g["fold"].to_numpy()
        neg = -p1
        for x in X_GRID:
            k = int(np.searchsorted(neg, -x, side="right"))
            if k < MIN_RACES:
                break
            if not evaluate(dc[:k], rc[:k], ret[:k], n[:k], fold[:k], D, rng,
                            {"family": "無オッズ", "structure": name, "lam": float(lam), "x": float(x)}, keep, rows):
                break
    return rows


def refine(keep: Keeper, rng) -> pd.Series | None:
    """上位候補を B_REFINE 回で再評価し、中央値 ≥ 1 のうち CI 下限が最大のものを返す。"""
    best = None
    for _, params, dr, dn in keep.items:
        pt, med, lo, hi = boot(dr, dn, rng, B_REFINE)
        if med >= 1.0 and (best is None or lo > best["ci_lo"]):
            best = pd.Series({**params, "roi": pt, "boot_median": med, "ci_lo": lo, "ci_hi": hi})
    return best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    days = day_map()
    win = S.load_win()
    meta = win.drop_duplicates("race_id").set_index("race_id")[["fold", "race_date"]]
    sel_rows = []
    for bt in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta", "bracket_q", "bracket_e"):
        keep = Keeper(TOP_REFINE)
        rows = ev_family(bt, rng, days, keep) if bt in EVBETS else []
        rows += free_family(bt, rng, days, meta, keep)
        c = pd.DataFrame(rows)
        c.to_csv(OUT / f"candidates_{bt}.csv", index=False)
        n_ok = int((c["boot_median"] >= 1.0).sum()) if len(c) else 0
        best = refine(keep, rng)
        rec = {"bet": R.BET_JP[bt], "候補数": len(c), "中央値≥1の数": n_ok,
               "全候補の最大CI下限": float(c["ci_lo"].max()) if len(c) else float("nan")}
        if best is not None:
            mask = pd.Series(True, index=c.index)
            for key in ("family", "t", "p_min", "mode", "structure", "lam", "x"):
                if key in c.columns:
                    mask &= c[key].isna() if pd.isna(best.get(key)) else (c[key] == best[key])
            full = c[mask]
            extra = full.iloc[0] if len(full) else {}
            rec.update({k: best.get(k) for k in ("family", "structure", "lam", "x", "t", "p_min", "mode", "roi", "boot_median",
                                                 "ci_lo", "ci_hi")})
            for k in ("races", "tickets", "hit_rate", "min_fold_roi"):
                rec[k] = extra[k] if len(full) else float("nan")
        sel_rows.append(rec)
        print(R.BET_JP[bt], "候補", len(c), "中央値≥1", n_ok, "→",
              "採用なし" if best is None else
              f"{best['family']} ROI {best['roi']:.3f} CI[{best['ci_lo']:.3f},{best['ci_hi']:.3f}] "
              f"{best.get('structure', '')} λ={best.get('lam')} x={best.get('x')} t={best.get('t')} "
              f"p>={best.get('p_min')} {best.get('mode', '')} 賭{rec.get('races')}R", flush=True)
    pd.DataFrame(sel_rows).to_csv(OUT / "selected.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
