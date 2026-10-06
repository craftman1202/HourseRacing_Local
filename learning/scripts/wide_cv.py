"""ワイドの全ての買い方について、walk-forward 5-fold と OOS の成績（2026-10-06）。

trifecta_cv.py と同じ枠組み（同じ4つの確率源・同じ期間・同じ軸の単勝確率の下限）。
ワイドの組の確率 = Harville で「2頭とも3着以内」になる確率（3頭の順序付き組の確率を、
その中に含まれる 3 つの2頭組へ足し込む）。上位 9 頭までで打ち切るので総和は約 3×0.94。
払戻は silver/payout.parquet の「ワイド」（1レース通常3組、同着で増える）。1点 100 円。

  ワイド 1点           確率1位-2位 の1点
  ワイド 軸1頭流し     1位を軸、相手 2〜(N+1)位
  ワイド 軸2頭流し     1位と2位それぞれを軸に 3〜(N+2)位へ（1-2 も含む）
  ワイド BOX           上位 N 頭の全組
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifecta_cv as C  # noqa: E402
import trifecta_strategies as T  # noqa: E402

ROOT = T.ROOT
OUT = ROOT / "artifacts/wide_cv"
TOP = 9


def strategies() -> list[tuple[str, dict]]:
    s = [("ワイド 1点（1位-2位）", {"kind": "pair12"})]
    for n in range(1, 9):
        s.append((f"ワイド 軸1頭流し 相手{n}頭", {"kind": "axis1", "n": n}))
    for n in range(1, 8):
        s.append((f"ワイド 軸2頭流し 相手{n}頭", {"kind": "axis2", "n": n}))
    for n in range(3, 8):
        s.append((f"ワイド BOX{n}頭", {"kind": "box", "n": n}))
    return s


def mask(t: pd.DataFrame, p: dict) -> np.ndarray:
    a, b = t["ra"].to_numpy(), t["rb"].to_numpy()       # ra < rb
    k = p["kind"]
    if k == "pair12":
        return (a == 1) & (b == 2)
    if k == "axis1":
        return (a == 1) & (b <= p["n"] + 1)
    if k == "axis2":
        return (a <= 2) & (b <= p["n"] + 2)
    if k == "box":
        return b <= p["n"]
    raise ValueError(k)


def wide_tickets(win: pd.DataFrame) -> pd.DataFrame:
    w = win.sort_values(["race_id", "p"], ascending=[True, False]).copy()
    w["rank"] = w.groupby("race_id").cumcount() + 1
    w = w[w["rank"] <= TOP]
    perms = np.array(list(itertools.permutations(range(TOP), 3)))
    pairs = np.array(list(itertools.combinations(range(TOP), 2)))
    rows = []
    for rid, g in w.groupby("race_id"):
        m = len(g)
        if m < 3:
            continue
        horse, p = g["horse_no"].to_numpy(), g["p"].to_numpy()
        pr = perms[(perms < m).all(1)]
        pa, pb, pc = p[pr[:, 0]], p[pr[:, 1]], p[pr[:, 2]]
        prob = pa * pb / np.clip(1 - pa, 1e-12, None) * pc / np.clip(1 - pa - pb, 1e-12, None)
        acc = np.zeros((m, m))
        for (i, j, k), v in zip(pr, prob):
            for x, y in ((i, j), (i, k), (j, k)):
                acc[min(x, y), max(x, y)] += v
        pc_ = pairs[(pairs < m).all(1)]
        hs = np.sort(horse[pc_], axis=1)
        rows.append(pd.DataFrame({"race_id": rid, "h1": hs[:, 0], "h2": hs[:, 1],
                                  "ra": pc_[:, 0] + 1, "rb": pc_[:, 1] + 1,
                                  "P": acc[pc_[:, 0], pc_[:, 1]], "p1": p[0]}))
    return pd.concat(rows, ignore_index=True)


def attach_payout(t: pd.DataFrame) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "payout_yen"])
    pay = pay[pay["bet_type"] == "ワイド"].dropna(subset=["comb_2"])
    sold = set(pay["race_id"])
    hs = np.sort(np.c_[pay["comb_1"].astype(int), pay["comb_2"].astype(int)], axis=1)
    pay = pay.assign(h1=hs[:, 0], h2=hs[:, 1]).groupby(["race_id", "h1", "h2"])[
        "payout_yen"].sum().reset_index()
    t = t[t["race_id"].isin(sold)].merge(pay, on=["race_id", "h1", "h2"], how="left")
    t["ret"] = t["payout_yen"].fillna(0.0)
    return t.drop(columns="payout_yen")


def run_model(name, folds, oos) -> pd.DataFrame:
    win = pd.concat([folds[["race_id", "horse_no", "race_date", "fold", "ensemble"]], oos],
                    ignore_index=True).rename(columns={"ensemble": "p"})
    win = win[win["race_id"].str[:2].astype(int) > 4]
    meta = win.drop_duplicates("race_id").set_index("race_id")
    t = attach_payout(wide_tickets(win))
    t["fold"] = t["race_id"].map(meta["fold"]).to_numpy()
    t["race_date"] = pd.to_datetime(t["race_id"].map(meta["race_date"]))
    rows = []
    for sname, p in strategies():
        base = mask(t, p)
        for x in T.X_GRID:
            sel = t[base & (t["p1"].to_numpy() >= x)]
            for f, g in sel.groupby("fold"):
                row = {"model": name, "strategy": sname, "bet": "wide", "x": x,
                       "period": "oos" if f == 0 else f"fold{int(f)}", **T.summarize(g)}
                if f == 0:
                    row["ci_lo"], row["ci_hi"], _ = T.boot_ci(g)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src = C.sources()
    for name in sys.argv[1:] or list(src):
        long = run_model(name, *src[name]())
        long.to_csv(OUT / f"long_{name}.csv", index=False)
        print(f"{name}: {len(long)} 行")
    w = C.wide(pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("long_*.csv"))],
                         ignore_index=True))
    w.to_csv(OUT / "all_wide_5fold_oos.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
