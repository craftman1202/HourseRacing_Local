"""3連単・3連複の「軸固定 + 上位馬への流し / BOX / フォーメーション」の ROI（2026-10-05）。

単勝確率（保存済みの OOS 予測）から Harville 式で 3 連の組の確率を出し、確率順位で
軸と相手を決める。払戻は data_real/silver/payout.parquet（月次の払戻 CSV 由来）。

  構造だけの戦略（オッズ不要）: OOS 全期間 2024-02〜。dev 〜2025-07-31 で選び、test で1回測る
  期待値で絞る戦略（オッズ要）: 3連のオッズは bronze/odds に 2026-02〜しか無い（確定オッズ）。
                               dev 〜2026-05-31 / test 2026-06〜

確率の出所（MODEL 環境変数）:
  dev_reg  = 本番の乖離回帰モデル（artifacts/dev_flat_pop/oos_win_reg.parquet、既定）
  pop      = 人気だけの勝率（artifacts/dev_flat_pop/oos_win_pop.parquet）— 「人気順に買う」対照
  base     = 人気を使わない単勝モデル（artifacts/oos_predictions.parquet）
1点 100 円の均等買い。人気（確定時点）を使うモデルは楽観側。
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.transform import silver as S  # noqa: E402

MODEL = os.environ.get("MODEL", "dev_reg")
WIN = {"dev_reg": "artifacts/dev_flat_pop/oos_win_reg.parquet",
       "pop": "artifacts/dev_flat_pop/oos_win_pop.parquet",
       "base": "artifacts/oos_predictions.parquet"}[MODEL]
OUT = ROOT / f"artifacts/trifecta_{MODEL}"
TOP = 9                                   # 相手に取りうるのは確率上位 9 頭まで
STRUCT_DEV_END = pd.Timestamp("2025-07-31")
ODDS_DEV_END = pd.Timestamp("2026-05-31")
X_GRID = (0.0, 0.2, 0.3, 0.4, 0.5)        # 軸（確率1位）の単勝確率の下限
T_GRID = (0.0, 1.0, 1.2, 1.5, 2.0, 3.0)   # 1点ごとの EV 下限（0 = 絞らない）
MIN_DEV_RACES = {"struct": 500, "odds": 100}
N_BOOT = 2000


# ------------------------------------------------------------------ 戦略（順位の条件）
def strategies() -> list[tuple[str, str, dict]]:
    """(名前, 券種, パラメータ)。順位は確率順位（1 = 最も勝ちそう）。"""
    s = []
    for n in range(4, 10):
        s.append((f"3連単 1着固定流し 相手{n - 1}頭", "tri", {"kind": "t1", "n": n}))
        s.append((f"3連単 軸1頭マルチ 相手{n - 1}頭", "tri", {"kind": "multi", "n": n}))
        s.append((f"3連複 軸1頭流し 相手{n - 1}頭", "trio", {"kind": "r1", "n": n}))
        s.append((f"3連複 軸2頭流し 相手{n - 2}頭", "trio", {"kind": "r2", "n": n}))
    for n in range(4, 10):
        s.append((f"3連単 1・2着固定 3着{n - 2}頭", "tri", {"kind": "t12", "n": n}))
    for n in range(3, 7):
        s.append((f"3連単 BOX{n}頭", "tri", {"kind": "box", "n": n}))
        s.append((f"3連複 BOX{n}頭", "trio", {"kind": "box", "n": n}))
    for a in (2, 3, 4):
        for c in range(max(a + 1, 4), 9):
            s.append((f"3連単 フォーメーション 1着1位/2着上位{a}/3着上位{c}", "tri",
                      {"kind": "form", "a": a, "c": c}))
    return s


def mask(t: pd.DataFrame, bet: str, p: dict) -> np.ndarray:
    a, b, c = t["ra"].to_numpy(), t["rb"].to_numpy(), t["rc"].to_numpy()
    k, n = p["kind"], p.get("n")
    if bet == "tri":
        if k == "t1":
            return (a == 1) & (b <= n) & (c <= n)
        if k == "t12":
            return (a == 1) & (b == 2) & (c <= n)
        if k == "multi":
            has1 = (a == 1) | (b == 1) | (c == 1)
            return has1 & (a <= n) & (b <= n) & (c <= n)
        if k == "box":
            return (a <= n) & (b <= n) & (c <= n)
        if k == "form":
            return (a == 1) & (b <= p["a"]) & (c <= p["c"])
    else:   # trio: ra < rb < rc（順位で並べた組）
        if k == "r1":
            return (a == 1) & (c <= n)
        if k == "r2":
            return (a == 1) & (b == 2) & (c <= n)
        if k == "box":
            return c <= n
    raise ValueError(k)


# ------------------------------------------------------------------ 組の確率
def tickets(win: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """上位 TOP 頭の全ての順序付き3頭組（3連単）と順序なし組（3連複）。Harville 確率つき。"""
    w = win.sort_values(["race_id", "p"], ascending=[True, False]).copy()
    w["rank"] = w.groupby("race_id").cumcount() + 1
    w = w[w["rank"] <= TOP]
    races = w.groupby("race_id")
    tri_rows, trio_rows = [], []
    perms = np.array(list(itertools.permutations(range(TOP), 3)))
    combs = np.array(list(itertools.combinations(range(TOP), 3)))
    for rid, g in races:
        m = len(g)
        if m < 3:
            continue
        horse = g["horse_no"].to_numpy()
        p = g["p"].to_numpy()
        pr = perms[(perms < m).all(1)]
        pa, pb, pc = p[pr[:, 0]], p[pr[:, 1]], p[pr[:, 2]]
        prob = pa * pb / np.clip(1 - pa, 1e-12, None) * pc / np.clip(1 - pa - pb, 1e-12, None)
        tri_rows.append(pd.DataFrame({
            "race_id": rid, "h1": horse[pr[:, 0]], "h2": horse[pr[:, 1]], "h3": horse[pr[:, 2]],
            "ra": pr[:, 0] + 1, "rb": pr[:, 1] + 1, "rc": pr[:, 2] + 1, "P": prob,
            "p1": p[0]}))
        # 3連複の確率は 6 通りの順列の和
        lookup = {tuple(x): v for x, v in zip(map(tuple, pr), prob)}
        cb = combs[(combs < m).all(1)]
        ptrio = np.array([sum(lookup[q] for q in itertools.permutations(x)) for x in map(tuple, cb)])
        hs = np.sort(horse[cb], axis=1)
        trio_rows.append(pd.DataFrame({
            "race_id": rid, "h1": hs[:, 0], "h2": hs[:, 1], "h3": hs[:, 2],
            "ra": cb[:, 0] + 1, "rb": cb[:, 1] + 1, "rc": cb[:, 2] + 1, "P": ptrio,
            "p1": p[0]}))
    return pd.concat(tri_rows, ignore_index=True), pd.concat(trio_rows, ignore_index=True)


def attach_payout(t: pd.DataFrame, bet_jp: str) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "comb_3", "payout_yen"])
    pay = pay[pay["bet_type"] == bet_jp].dropna(subset=["comb_2", "comb_3"])
    sold = set(pay["race_id"])
    pay = pay.assign(h1=pay["comb_1"].astype(int), h2=pay["comb_2"].astype(int),
                     h3=pay["comb_3"].astype(int))
    if bet_jp == "3連複":
        hs = np.sort(pay[["h1", "h2", "h3"]].to_numpy(), axis=1)
        pay[["h1", "h2", "h3"]] = hs
    pay = pay.groupby(["race_id", "h1", "h2", "h3"])["payout_yen"].sum().reset_index()
    t = t[t["race_id"].isin(sold)]            # 発売の無いレース（払戻行が無い）は評価対象外
    t = t.merge(pay, on=["race_id", "h1", "h2", "h3"], how="left")
    t["ret"] = t["payout_yen"].fillna(0.0)
    return t.drop(columns="payout_yen")


def attach_odds(t: pd.DataFrame, bet_jp: str) -> pd.DataFrame:
    """3連のオッズ（確定値、2026-02〜）。番号は 3連単なら着順、3連複は昇順。"""
    code = {"3連単": "３連単", "3連複": "３連複"}[bet_jp]
    fs = sorted(glob.glob(str(ROOT / "data_real/bronze/odds/ym=*/*.parquet")))
    parts = []
    master = S.TrackMaster()
    for f in fs:
        b = pd.read_parquet(f, columns=["競馬場", "競走年月日", "レース番号", "賭式", "番号1", "番号2",
                                        "番号3", "オッズ"])
        b = b[b["賭式"] == code]
        if b.empty:
            continue
        baba = master.codes(b["競馬場"])
        ymd = S._ymd(b["競走年月日"])
        rn = S._digits(b["レース番号"]).astype("Int64")
        o = pd.DataFrame({
            "race_id": [S.make_race_id(x, d, r) for x, d, r in zip(baba, ymd, rn)],
            "h1": S._digits(b["番号1"]).to_numpy(), "h2": S._digits(b["番号2"]).to_numpy(),
            "h3": S._digits(b["番号3"]).to_numpy(), "odds": S._digits(b["オッズ"]).to_numpy()})
        parts.append(o.dropna())
    o = pd.concat(parts, ignore_index=True)
    o = o[o["race_id"].isin(set(t["race_id"]))]
    for c in ("h1", "h2", "h3"):
        o[c] = o[c].astype(int)
    if bet_jp == "3連複":
        o[["h1", "h2", "h3"]] = np.sort(o[["h1", "h2", "h3"]].to_numpy(), axis=1)
    o = o[o["odds"] > 0].drop_duplicates(["race_id", "h1", "h2", "h3"])
    return t.merge(o, on=["race_id", "h1", "h2", "h3"], how="inner")


# ------------------------------------------------------------------ 集計
def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"n_races": 0, "n_tickets": 0, "stake": 0.0, "ret": 0.0, "roi": np.nan,
                "race_hit": np.nan, "tickets_per_race": np.nan}
    r = b.groupby("race_id").agg(ret=("ret", "sum"), n=("ret", "size"))
    stake = 100.0 * len(b)
    return {"n_races": len(r), "n_tickets": len(b), "stake": stake, "ret": float(b["ret"].sum()),
            "roi": float(b["ret"].sum() / stake), "race_hit": float((r["ret"] > 0).mean()),
            "tickets_per_race": float(r["n"].mean())}


def boot_ci(b: pd.DataFrame, seed=0):
    d = b.groupby("race_date").agg(ret=("ret", "sum"), n=("ret", "size"))
    r, n = d["ret"].to_numpy(), d["n"].to_numpy() * 100.0
    idx = np.random.default_rng(seed).integers(0, len(d), size=(N_BOOT, len(d)))
    bs = r[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float((np.sum(bs <= 1) + 1) / (N_BOOT + 1))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(q, 1)
    return out


def evaluate(t_by_bet: dict, dev_end, use_odds: bool, min_dev: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid, chosen = [], []
    for name, bet, p in strategies():
        t = t_by_bet[bet]
        base = mask(t, bet, p)
        for x in X_GRID:
            for thr in (T_GRID if use_odds else (0.0,)):
                m = base & (t["p1"].to_numpy() >= x)
                if thr > 0:
                    m = m & (t["P"].to_numpy() * t["odds"].to_numpy() >= thr)
                sel = t[m]
                for per, g in sel.groupby(sel["race_date"] > dev_end):
                    grid.append({"strategy": name, "bet": bet, "x": x, "ev_min": thr,
                                 "period": "test" if per else "dev", **summarize(g)})
    grid = pd.DataFrame(grid)
    dev = grid[(grid["period"] == "dev") & (grid["n_races"] >= min_dev)]
    # 戦略（構造）ごとに dev 最良の (x, ev_min) を選ぶ
    best = dev.loc[dev.groupby("strategy")["roi"].idxmax()]
    rows = []
    for _, r in best.iterrows():
        bet = r["bet"]
        t = t_by_bet[bet]
        p = dict(next(pp for nn, bb, pp in strategies() if nn == r["strategy"]))
        m = mask(t, bet, p) & (t["p1"].to_numpy() >= r["x"])
        if r["ev_min"] > 0:
            m = m & (t["P"].to_numpy() * t["odds"].to_numpy() >= r["ev_min"])
        g = t[m & (t["race_date"] > dev_end).to_numpy()]
        s = summarize(g)
        lo, hi, pv = boot_ci(g) if len(g) else (np.nan, np.nan, np.nan)
        rows.append({"strategy": r["strategy"], "x": r["x"], "ev_min": r["ev_min"],
                     "dev_races": int(r["n_races"]), "dev_roi": r["roi"], "dev_hit": r["race_hit"],
                     "test_races": s["n_races"], "tickets_per_race": s["tickets_per_race"],
                     "test_roi": s["roi"], "test_hit": s["race_hit"], "ci_lo": lo, "ci_hi": hi,
                     "p_roi_gt1": pv})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res["p_roi_gt1"].fillna(1).to_numpy())
    return grid, res.sort_values("dev_roi", ascending=False)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    col = "ensemble"
    win = pd.read_parquet(ROOT / WIN, columns=["race_id", "horse_no", "race_date", col])
    win = win.rename(columns={col: "p"})
    win = win[win["race_id"].str[:2].astype(int) > 4]     # 平地のみ（ばんえいは別系統）
    dates = win.drop_duplicates("race_id").set_index("race_id")["race_date"]
    tri, trio = tickets(win)
    tri = attach_payout(tri, "3連単")
    trio = attach_payout(trio, "3連複")
    for t in (tri, trio):
        t["race_date"] = pd.to_datetime(t["race_id"].map(dates))

    pd.set_option("display.width", 250)
    # 1) 構造だけ（OOS 全期間）
    grid, res = evaluate({"tri": tri, "trio": trio}, STRUCT_DEV_END, False, MIN_DEV_RACES["struct"])
    grid.to_csv(OUT / "struct_grid.csv", index=False)
    res.to_csv(OUT / "struct_selected.csv", index=False)
    print(f"=== 構造だけ（{MODEL}、dev 〜{STRUCT_DEV_END.date()} / test それ以降） ===")
    print(res.round(3).to_string(index=False))

    # 2) 期待値で絞る（オッズのある 2026-02〜）
    tri_o = attach_odds(tri[tri["race_date"] >= "2026-02-01"], "3連単")
    trio_o = attach_odds(trio[trio["race_date"] >= "2026-02-01"], "3連複")
    grid2, res2 = evaluate({"tri": tri_o, "trio": trio_o}, ODDS_DEV_END, True, MIN_DEV_RACES["odds"])
    grid2.to_csv(OUT / "odds_grid.csv", index=False)
    res2.to_csv(OUT / "odds_selected.csv", index=False)
    print(f"\n=== 期待値で絞る（{MODEL}、確定オッズ、dev 〜{ODDS_DEV_END.date()} / test 06〜） ===")
    print(res2.round(3).to_string(index=False))
    (OUT / "info.json").write_text(json.dumps({
        "model": MODEL, "win_source": WIN,
        "tri_races": int(tri["race_id"].nunique()), "trio_races": int(trio["race_id"].nunique()),
        "odds_races": int(tri_o["race_id"].nunique())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
