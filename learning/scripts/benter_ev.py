"""市場オッズとモデルの合成（Benter 型）＋期待値で絞る、全券種（2026-10-13）。

これまでの EV 戦略はモデル確率だけで EV を出していたため、モデルが市場と食い違う穴馬ほど
EV が大きく見え、実現率と逆相関していた（place-model-and-ev-strategy-findings）。
ここでは単勝の確定オッズから市場確率 q を作り、
    log π_i = a·log p_model_i + b·log q_i （レース内 softmax）
の a, b を dev（2026-02〜05）の勝ち馬で最尤推定してから、π で各券種の組の確率を出し
EV = π(組) × オッズ ≥ t の組を買う。t と「全部買う／レースで EV 最大の1点だけ」は dev で選び、
test（2026-06〜09）で1回測る。比較として a=1,b=0（モデルだけ）と a=0,b=1（市場だけ）も同じ手順で。

組の確率は Harville（上位 9 頭まで）。複勝・ワイドのオッズは範囲なので
「下限 + α(上限−下限)」（α は dev の実払戻の平均位置）。**オッズはすべて確定値**なので楽観的。
券種: 単勝・複勝・ワイド・馬連・馬単・3連複・3連単（枠は馬の確率からの集約が要るので対象外）。
"""

from __future__ import annotations

import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.eval.place import harville_topk, place_slots, race_log_softmax  # noqa: E402
from nar.transform import silver as S  # noqa: E402

OUT = ROOT / "artifacts/benter_ev"
DEV_END = pd.Timestamp("2026-05-31")
T_GRID = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
TOP = 9
CODE = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬複", "exacta": "馬単",
        "trio": "３連複", "trifecta": "３連単"}
PAY = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単",
       "trio": "3連複", "trifecta": "3連単"}
JP = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単",
      "trio": "3連複", "trifecta": "3連単"}
K = {"win": 1, "place": 1, "wide": 2, "quinella": 2, "exacta": 2, "trio": 3, "trifecta": 3}
ORDERED = {"exacta", "trifecta"}
N_BOOT = 2000


def load_odds(races: set) -> dict[str, pd.DataFrame]:
    master = S.TrackMaster()
    parts = {bt: [] for bt in CODE}
    inv = {v: k for k, v in CODE.items()}
    for f in sorted(glob.glob(str(ROOT / "data_real/bronze/odds/ym=*/*.parquet"))):
        b = pd.read_parquet(f)
        b = b[b["賭式"].isin(inv)]
        if b.empty:
            continue
        baba = master.codes(b["競馬場"])
        ymd = S._ymd(b["競走年月日"])
        rn = S._digits(b["レース番号"]).astype("Int64")
        rid = np.array([S.make_race_id(x, d, r) for x, d, r in zip(baba, ymd, rn)])
        keep = np.isin(rid, list(races))
        b, rid = b[keep], rid[keep]
        for code, g in b.groupby("賭式"):
            bt = inv[code]
            idx = (b["賭式"] == code).to_numpy()
            o = pd.DataFrame({"race_id": rid[idx],
                              "h1": S._digits(g["番号1"]).to_numpy(), "h2": S._digits(g["番号2"]).to_numpy(),
                              "h3": S._digits(g["番号3"]).to_numpy(),
                              "omin": S._digits(g["オッズ"]).to_numpy(),
                              "omax": S._digits(g["オッズ（最大）"]).to_numpy()})
            parts[bt].append(o)
    out = {}
    for bt, ps in parts.items():
        o = pd.concat(ps, ignore_index=True)
        cols = ["h1", "h2", "h3"][:K[bt]]
        o = o.dropna(subset=cols + ["omin"])
        o = o[o["omin"] > 0]
        arr = o[cols].astype(int).to_numpy()
        if bt not in ORDERED and K[bt] > 1:
            arr = np.sort(arr, axis=1)
        for i, c in enumerate(cols):
            o[c] = arr[:, i]
        out[bt] = o.drop_duplicates(["race_id", *cols])
    return out


def load_payout(bt: str) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "comb_3", "payout_yen"])
    cols = ["comb_1", "comb_2", "comb_3"][:K[bt]]
    d = pay[pay["bet_type"] == PAY[bt]].dropna(subset=cols)
    arr = d[cols].astype(int).to_numpy()
    if bt not in ORDERED and K[bt] > 1:
        arr = np.sort(arr, axis=1)
    names = ["h1", "h2", "h3"][:K[bt]]
    d = pd.DataFrame(arr, columns=names).assign(race_id=d["race_id"].to_numpy(),
                                                ret=d["payout_yen"].to_numpy() / 100.0)
    return d.groupby(["race_id", *names])["ret"].sum().reset_index()


def fit_blend(df: pd.DataFrame) -> tuple[float, float]:
    rid = df["race_id"].to_numpy()
    lp, lq, y = np.log(df["p_model"].to_numpy()), np.log(df["q"].to_numpy()), df["is_win"].to_numpy()

    def nll(th):
        pi = np.exp(race_log_softmax(th[0] * lp + th[1] * lq, rid))
        return -np.log(np.clip(pi[y == 1], 1e-12, None)).mean()
    res = minimize(nll, np.array([0.5, 0.5]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return float(res.x[0]), float(res.x[1])


def combo_probs(df: pd.DataFrame, pcol: str) -> dict[str, pd.DataFrame]:
    """馬の確率 pcol から各券種の組の確率（Harville、上位 TOP 頭）。"""
    out = {bt: [] for bt in CODE}
    for rid, g in df.sort_values(["race_id", pcol], ascending=[True, False]).groupby("race_id", sort=False):
        h, p = g["horse_no"].to_numpy(), g[pcol].to_numpy()
        n = len(h)
        out["win"].append(pd.DataFrame({"race_id": rid, "h1": h, "P": p}))
        k = int(place_slots(n))
        if k:
            out["place"].append(pd.DataFrame({"race_id": rid, "h1": h,
                                              "P": harville_topk(p, np.full(n, rid), np.full(n, k))}))
        m = min(n, TOP)
        hh, pp = h[:m], p[:m]
        ex = {}
        for i, j in itertools.permutations(range(m), 2):
            ex[(i, j)] = pp[i] * pp[j] / max(1 - pp[i], 1e-12)
        out["exacta"].append(pd.DataFrame([(rid, hh[i], hh[j], v) for (i, j), v in ex.items()],
                                          columns=["race_id", "h1", "h2", "P"]))
        out["quinella"].append(pd.DataFrame([(rid, *sorted((hh[i], hh[j])), ex[(i, j)] + ex[(j, i)])
                                             for i, j in itertools.combinations(range(m), 2)],
                                            columns=["race_id", "h1", "h2", "P"]))
        tri, wide = {}, {}
        for i, j, l in itertools.permutations(range(m), 3):
            v = ex[(i, j)] * pp[l] / max(1 - pp[i] - pp[j], 1e-12)
            tri[(i, j, l)] = v
            for a, b in ((i, j), (i, l), (j, l)):
                key = tuple(sorted((a, b)))
                wide[key] = wide.get(key, 0.0) + v
        out["trifecta"].append(pd.DataFrame([(rid, hh[i], hh[j], hh[l], v) for (i, j, l), v in tri.items()],
                                            columns=["race_id", "h1", "h2", "h3", "P"]))
        trio = {}
        for (i, j, l), v in tri.items():
            key = tuple(sorted((hh[i], hh[j], hh[l])))
            trio[key] = trio.get(key, 0.0) + v
        out["trio"].append(pd.DataFrame([(rid, *k_, v) for k_, v in trio.items()],
                                        columns=["race_id", "h1", "h2", "h3", "P"]))
        out["wide"].append(pd.DataFrame([(rid, *sorted((hh[a], hh[b])), v) for (a, b), v in wide.items()],
                                        columns=["race_id", "h1", "h2", "P"]))
    return {bt: pd.concat(v, ignore_index=True) for bt, v in out.items() if v}


def boot(b: pd.DataFrame) -> tuple[float, float]:
    d = b.groupby("race_date").agg(ret=("ret", "sum"), n=("ret", "size"))
    r, n = d["ret"].to_numpy(), d["n"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    bs = r[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(ROOT / "artifacts/dev_flat_pop/oos_win_reg.parquet")
    w = w[w["race_id"].str[:2].astype(int) > 4].rename(columns={"ensemble": "p_model"})
    odds = load_odds(set(w["race_id"]))
    win_o = odds["win"][["race_id", "h1", "omin"]].rename(columns={"h1": "horse_no", "omin": "odds"})
    w = w.merge(win_o, on=["race_id", "horse_no"], how="inner")
    full = w.groupby("race_id")["horse_no"].transform("size") == w.groupby("race_id")["horse_no"].transform("size").max()
    n_by = w.groupby("race_id").size()
    w = w[w["race_id"].map(n_by) >= 2].copy()
    w["race_date"] = pd.to_datetime(w["race_date"])
    w["q"] = (1 / w["odds"]) / w.groupby("race_id")["odds"].transform(lambda s: (1 / s).sum())
    w["p_model"] = w.groupby("race_id")["p_model"].transform(lambda s: s / s.sum())
    dev = w[w["race_date"] <= DEV_END]
    a, b = fit_blend(dev)
    variants = {"モデルだけ": (1.0, 0.0), "市場だけ": (0.0, 1.0), "合成(Benter)": (a, b)}
    info = {"a": a, "b": b, "races_dev": int(dev.race_id.nunique()),
            "races_test": int(w[w.race_date > DEV_END].race_id.nunique())}
    rid = w["race_id"].to_numpy()
    for v, (aa, bb) in variants.items():
        w[f"pi_{v}"] = np.exp(race_log_softmax(aa * np.log(w["p_model"]) + bb * np.log(w["q"]), rid))
        # 単勝の確率の当てはまり（参考）
        for per, g in w.groupby(w.race_date > DEV_END):
            info[f"nll_{v}_{'test' if per else 'dev'}"] = float(
                -np.log(np.clip(g.loc[g.is_win == 1, f"pi_{v}"], 1e-12, None)).mean())
    pays = {bt: load_payout(bt) for bt in CODE}
    dates = w.drop_duplicates("race_id").set_index("race_id")["race_date"]

    rows = []
    for v in variants:
        cp = combo_probs(w, f"pi_{v}")
        for bt, t in cp.items():
            names = ["h1", "h2", "h3"][:K[bt]]
            t = t.merge(odds[bt], on=["race_id", *names], how="inner")
            t = t.merge(pays[bt], on=["race_id", *names], how="left")
            t["ret"] = t["ret"].fillna(0.0)
            t["race_date"] = t["race_id"].map(dates)
            dev_m = t["race_date"] <= DEV_END
            if bt in ("place", "wide"):
                hit = dev_m & (t["ret"] > 0) & (t["omax"] > t["omin"])
                alpha = float(((t.loc[hit, "ret"] - t.loc[hit, "omin"]) /
                               (t.loc[hit, "omax"] - t.loc[hit, "omin"])).clip(0, 1).mean())
                t["o_est"] = t["omin"] + alpha * (t["omax"] - t["omin"])
            else:
                t["o_est"] = t["omin"]
            t["ev"] = t["P"] * t["o_est"]
            for thr in T_GRID:
                s = t[t["ev"] >= thr]
                for mode in ("全部", "レース最大1点"):
                    ss = s if mode == "全部" else s.loc[s.groupby("race_id")["ev"].idxmax()]
                    for per, g in ss.groupby(ss["race_date"] > DEV_END):
                        rows.append({"variant": v, "bet": JP[bt], "ev_min": thr, "mode": mode,
                                     "period": "test" if per else "dev", "n_bets": len(g),
                                     "n_races": g["race_id"].nunique(), "roi": g["ret"].sum() / len(g),
                                     "hit": float((g["ret"] > 0).mean())})
            t.to_parquet(OUT / f"tickets_{bt}_{v}.parquet", index=False)
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)

    # 券種×確率ごとに dev 最良（dev 30 ベット以上）を選び test を1回
    sel = []
    d = grid[(grid.period == "dev") & (grid.n_bets >= 30)]
    for (v, bet), g in d.groupby(["variant", "bet"]):
        r = g.loc[g["roi"].idxmax()]
        bt = {vv: kk for kk, vv in JP.items()}[bet]
        t = pd.read_parquet(OUT / f"tickets_{bt}_{v}.parquet")
        s = t[(t["ev"] >= r["ev_min"]) & (t["race_date"] > DEV_END)]
        if r["mode"] != "全部" and len(s):
            s = s.loc[s.groupby("race_id")["ev"].idxmax()]
        lo, hi = boot(s) if len(s) else (np.nan, np.nan)
        sel.append({"券種": bet, "確率": v, "ev_min": r["ev_min"], "買い方": r["mode"],
                    "dev_bets": int(r["n_bets"]), "dev_roi": r["roi"],
                    "test_bets": len(s), "test_roi": float(s["ret"].sum() / len(s)) if len(s) else np.nan,
                    "test_hit": float((s["ret"] > 0).mean()) if len(s) else np.nan,
                    "ci_lo": lo, "ci_hi": hi})
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "selected.csv", index=False)
    (OUT / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    pd.set_option("display.width", 250)
    print(json.dumps(info, ensure_ascii=False))
    print(sel.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
