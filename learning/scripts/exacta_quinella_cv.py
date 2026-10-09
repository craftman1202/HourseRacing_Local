"""馬単・馬連（馬複）の全ての買い方について、walk-forward 5-fold と OOS の成績（2026-10-09）。

trifecta_cv.py / wide_cv.py と同じ枠組み（同じ4つの確率源・期間・軸の単勝確率の下限）。
組の確率は Harville: 馬単 P(i→j) = p_i·p_j/(1−p_i)、馬連 = 両順の和。上位 9 頭まで。
払戻は silver/payout.parquet の「馬単」「馬連」。1点 100 円。

  馬単  1点（1位→2位）、裏表（1位⇄2位）、1着固定流し 相手N、2着固定流し（相手N→1位）、
        1着軸マルチ 相手N（1位を含む全順序）、BOX N、フォーメーション 1着上位a × 2着上位b
  馬連  1点（1位-2位）、軸1頭流し 相手N、BOX N、フォーメーション 上位a × 上位b
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
OUT = ROOT / "artifacts/exacta_quinella_cv"
TOP = 9


def strategies() -> list[tuple[str, str, dict]]:
    s = [("馬単 1点（1位→2位）", "ex", {"kind": "one"}),
         ("馬単 裏表（1位⇄2位）", "ex", {"kind": "both12"}),
         ("馬連 1点（1位-2位）", "qn", {"kind": "one"})]
    for n in range(2, 9):
        s.append((f"馬単 1着固定流し 相手{n}頭", "ex", {"kind": "axis1st", "n": n}))
        s.append((f"馬単 2着固定流し 相手{n}頭", "ex", {"kind": "axis2nd", "n": n}))
        s.append((f"馬単 軸1頭マルチ 相手{n}頭", "ex", {"kind": "multi", "n": n}))
        s.append((f"馬連 軸1頭流し 相手{n}頭", "qn", {"kind": "axis", "n": n}))
    for n in range(3, 7):
        s.append((f"馬単 BOX{n}頭", "ex", {"kind": "box", "n": n}))
        s.append((f"馬連 BOX{n}頭", "qn", {"kind": "box", "n": n}))
    for a in (2, 3):
        for b in range(a + 1, 7):
            s.append((f"馬単 フォーメーション 1着上位{a}/2着上位{b}", "ex", {"kind": "form", "a": a, "b": b}))
            s.append((f"馬連 フォーメーション 上位{a}×上位{b}", "qn", {"kind": "form", "a": a, "b": b}))
    return s


def mask(t: pd.DataFrame, bet: str, p: dict) -> np.ndarray:
    a, b = t["ra"].to_numpy(), t["rb"].to_numpy()
    k, n = p["kind"], p.get("n")
    if bet == "ex":                       # a = 1着の順位、b = 2着の順位
        if k == "one":
            return (a == 1) & (b == 2)
        if k == "both12":
            return ((a == 1) & (b == 2)) | ((a == 2) & (b == 1))
        if k == "axis1st":
            return (a == 1) & (b <= n + 1)
        if k == "axis2nd":
            return (b == 1) & (a <= n + 1)
        if k == "multi":
            return ((a == 1) | (b == 1)) & (a <= n + 1) & (b <= n + 1)
        if k == "box":
            return (a <= n) & (b <= n)
        if k == "form":
            return (a <= p["a"]) & (b <= p["b"])
    else:                                 # qn: a < b（順位で並べた組）
        if k == "one":
            return (a == 1) & (b == 2)
        if k == "axis":
            return (a == 1) & (b <= n + 1)
        if k == "box":
            return b <= n
        if k == "form":
            return (a <= p["a"]) & (b <= p["b"])
    raise ValueError(k)


def pair_tickets(win: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = win.sort_values(["race_id", "p"], ascending=[True, False]).copy()
    w["rank"] = w.groupby("race_id").cumcount() + 1
    w = w[w["rank"] <= TOP]
    perms = np.array(list(itertools.permutations(range(TOP), 2)))
    ex_rows, qn_rows = [], []
    for rid, g in w.groupby("race_id"):
        m = len(g)
        if m < 2:
            continue
        horse, p = g["horse_no"].to_numpy(), g["p"].to_numpy()
        pr = perms[(perms < m).all(1)]
        pa, pb = p[pr[:, 0]], p[pr[:, 1]]
        prob = pa * pb / np.clip(1 - pa, 1e-12, None)
        ex_rows.append(pd.DataFrame({"race_id": rid, "h1": horse[pr[:, 0]], "h2": horse[pr[:, 1]],
                                     "ra": pr[:, 0] + 1, "rb": pr[:, 1] + 1, "P": prob, "p1": p[0]}))
        up = pr[:, 0] < pr[:, 1]
        rev = {(j, i): v for (i, j), v in zip(map(tuple, pr), prob)}
        pq = np.array([v + rev[(i, j)] for (i, j), v in zip(map(tuple, pr[up]), prob[up])])
        hs = np.sort(np.c_[horse[pr[up, 0]], horse[pr[up, 1]]], axis=1)
        qn_rows.append(pd.DataFrame({"race_id": rid, "h1": hs[:, 0], "h2": hs[:, 1],
                                     "ra": pr[up, 0] + 1, "rb": pr[up, 1] + 1, "P": pq, "p1": p[0]}))
    return pd.concat(ex_rows, ignore_index=True), pd.concat(qn_rows, ignore_index=True)


def attach_payout(t: pd.DataFrame, bet_jp: str) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "payout_yen"])
    pay = pay[pay["bet_type"] == bet_jp].dropna(subset=["comb_2"])
    sold = set(pay["race_id"])
    h = np.c_[pay["comb_1"].astype(int), pay["comb_2"].astype(int)]
    if bet_jp == "馬連":
        h = np.sort(h, axis=1)
    pay = pay.assign(h1=h[:, 0], h2=h[:, 1]).groupby(["race_id", "h1", "h2"])[
        "payout_yen"].sum().reset_index()
    t = t[t["race_id"].isin(sold)].merge(pay, on=["race_id", "h1", "h2"], how="left")
    t["ret"] = t["payout_yen"].fillna(0.0)
    return t.drop(columns="payout_yen")


def run_model(name, folds, oos) -> pd.DataFrame:
    win = pd.concat([folds[["race_id", "horse_no", "race_date", "fold", "ensemble"]], oos],
                    ignore_index=True).rename(columns={"ensemble": "p"})
    win = win[win["race_id"].str[:2].astype(int) > 4]
    meta = win.drop_duplicates("race_id").set_index("race_id")
    ex, qn = pair_tickets(win)
    tb = {"ex": attach_payout(ex, "馬単"), "qn": attach_payout(qn, "馬連")}
    for t in tb.values():
        t["fold"] = t["race_id"].map(meta["fold"]).to_numpy()
        t["race_date"] = pd.to_datetime(t["race_id"].map(meta["race_date"]))
    rows = []
    for sname, bet, p in strategies():
        t = tb[bet]
        base = mask(t, bet, p)
        for x in T.X_GRID:
            sel = t[base & (t["p1"].to_numpy() >= x)]
            for f, g in sel.groupby("fold"):
                row = {"model": name, "strategy": sname, "bet": bet, "x": x,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
