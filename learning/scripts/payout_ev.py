"""払戻（オッズ）の予測モデルと、全券種の「全馬券 EV ≥ t」買いの評価（2026-10-17）。

締切前オッズは 2026-02〜（OOS 期間）しか無いので、fold（2019〜2023）で EV を使う買い方を検証するには
払戻を予測するしかない。

払戻モデル（券種ごと・fold ごと）
  目的変数  log(払戻/100)。払戻は「当たった組」でしか観測できない
  特徴量    組の各馬の人気（単勝人気、順序のある券種は着順位置ごと）、走った頭数、
            人気だけの勝率表から Harville で出した組の確率 log P_pop、競馬場、複勝/ワイドは払戻枠数
  重み      1 / P_pop（逆確率重み）。当たった組は人気側に偏るので、そのままだと「同じ人気構成の組の
            オッズ」を過小に推定する。逆確率で全組の母集団に戻す（99 パーセンタイルで頭打ち）
  学習期間  fold 開始の 8 年前〜31 日前（fold の検証期間を含まない）
  平均化    exp(予測) × 残差の重み付き平均 exp（smearing）で E[オッズ] に
買い方
  各レースで、アンサンブル確率（ensemble_wf）の上位 9 頭から作れる全組について
  EV = P_ens(組) × 予測オッズ を出し、EV ≥ t（t ∈ T_GRID）の組を
    all   全部買う / top1  レースで EV 最大の1点だけ
  の 2 通りで買う。1点 100 円。比較として P を人気だけの確率にしたもの（市場の確率で EV を出すと
  理屈上は控除率ぶん 1 を下回るはず＝払戻モデルの検算）も出す。
選択は fold2〜5 の平均、nested（fold k を fold<k で選ぶ）も出す。枠連・枠単は枠の人気が無いので対象外。
OOS は触らない。

出力: artifacts/payout_ev/{payout_model_quality.csv, grid.csv, selected.csv, nested.csv, ev_calibration.csv}
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402
import strategy_select as S  # noqa: E402

from nar.eval.deviation import log_p_pop  # noqa: E402
from nar.eval.place import harville_topk, place_slots  # noqa: E402

ROOT = R.ROOT
OUT = ROOT / "artifacts/payout_ev"
BETS = ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta")
K = {"win": 1, "place": 1, "quinella": 2, "exacta": 2, "wide": 2, "trio": 3, "trifecta": 3}
ORDERED = {"exacta", "trifecta"}
T_GRID = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
FOLDS = (2, 3, 4, 5)
TOP = 9
TRAIN_YEARS = 8
PAY_JP = R.BET_JP


# ====================================================================== 組の確率（Harville）
def combo_prob(bt: str, p: np.ndarray, idx: tuple[int, ...], topk: np.ndarray | None = None) -> float:
    """1 レース分の確率ベクトル p（総和 1、全頭）での組 idx（レース内位置）の確率。"""
    if bt == "win":
        return p[idx[0]]
    if bt == "place":
        return topk[idx[0]]
    if bt in ("exacta", "quinella"):
        i, j = idx
        ex = lambda a, b: p[a] * p[b] / max(1 - p[a], 1e-12)  # noqa: E731
        return ex(i, j) if bt == "exacta" else ex(i, j) + ex(j, i)
    tri = lambda a, b, c: p[a] * p[b] / max(1 - p[a], 1e-12) * p[c] / max(1 - p[a] - p[b], 1e-12)  # noqa: E731
    if bt == "trifecta":
        return tri(*idx)
    if bt == "trio":
        return sum(tri(*q) for q in itertools.permutations(idx))
    if bt == "wide":
        i, j = idx
        return sum(tri(*q) for k in range(len(p)) if k not in (i, j)
                   for q in itertools.permutations((i, j, k)))
    raise ValueError(bt)


def harville_all(bt: str, P: np.ndarray, mask: np.ndarray | None = None):
    """運用と共有（nar.eval.payout.harville_all）。"""
    from nar.eval.payout import harville_all as _h
    return _h(bt, P)


# ====================================================================== データ
def starters_with_pop(table_start: str) -> pd.DataFrame:
    st = R.starters()
    st = st[st["race_id"].str[:2].astype(int) > 4].copy()
    tab = R.pop_table_before(st, table_start)
    st["p_pop"] = np.exp(log_p_pop(st["popularity"].to_numpy(dtype=float), st["n"].to_numpy(),
                                   st["race_id"].to_numpy(), tab))
    st["prank"] = st["popularity"].fillna(st["n"]).clip(1, 16).astype(int)
    st["track"] = st["race_id"].str[:2].astype(int)
    return st


def payouts(bt: str) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "comb_3", "payout_yen"])
    cols = ["comb_1", "comb_2", "comb_3"][:K[bt]]
    d = pay[pay["bet_type"] == PAY_JP[bt]].dropna(subset=cols)
    arr = d[cols].astype(int).to_numpy()
    if bt not in ORDERED and K[bt] > 1:
        arr = np.sort(arr, axis=1)
    names = ["h1", "h2", "h3"][:K[bt]]
    out = pd.DataFrame(arr, columns=names).assign(race_id=d["race_id"].to_numpy(),
                                                  odds=d["payout_yen"].to_numpy() / 100.0)
    return out.groupby(["race_id", *names])["odds"].sum().reset_index()


def features_for(bt: str, ranks, n, logp, track) -> pd.DataFrame:
    """運用と共有（nar.eval.payout.payout_features）。"""
    from nar.eval.payout import payout_features
    return payout_features(bt, ranks, n, logp, track)


def training_rows(bt: str, st: pd.DataFrame, pays: pd.DataFrame, start, end) -> pd.DataFrame:
    """当たった組（払戻あり）の特徴量・重み。"""
    s = st[(st["race_date"] >= start) & (st["race_date"] <= end)]
    races = {rid: g for rid, g in s.groupby("race_id", sort=False)}
    names = ["h1", "h2", "h3"][:K[bt]]
    pw = pays[pays["race_id"].isin(races.keys())]
    rows = []
    for rec in pw.itertuples(index=False):
        g = races[rec.race_id]
        hn = g["horse_no"].to_numpy()
        pos = {h: i for i, h in enumerate(hn)}
        hs = [getattr(rec, c) for c in names]
        if any(h not in pos for h in hs):
            continue
        idx = tuple(pos[h] for h in hs)
        p = g["p_pop"].to_numpy()
        n = len(p)
        topk = harville_topk(p, np.zeros(n), np.full(n, place_slots(n))) if bt == "place" else None
        if bt == "place" and place_slots(n) == 0:
            continue
        pr = combo_prob(bt, p, idx, topk)
        rk = g["prank"].to_numpy()[list(idx)]
        if bt not in ORDERED:
            rk = np.sort(rk)
        rows.append((rec.race_id, *rk, n, np.log(max(pr, 1e-9)), int(g["track"].iloc[0]), rec.odds, pr))
    cols = ["race_id", *[f"r{i + 1}" for i in range(K[bt])], "n", "logp_pop", "track", "odds", "p_pop_combo"]
    return pd.DataFrame(rows, columns=cols)


def fit_payout_model(tr: pd.DataFrame, bt: str):
    import lightgbm as lgb

    X = features_for(bt, tr[[f"r{i + 1}" for i in range(K[bt])]].to_numpy(), tr["n"].to_numpy(),
                     tr["logp_pop"].to_numpy(), tr["track"].to_numpy())
    y = np.log(tr["odds"].to_numpy())
    w = 1.0 / np.clip(tr["p_pop_combo"].to_numpy(), 1e-6, 1)
    w = np.minimum(w, np.quantile(w, 0.99))
    prm = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               verbosity=-1, seed=0, num_threads=4)
    m = lgb.train(prm, lgb.Dataset(X, label=y, weight=w, categorical_feature=["track"]), 600)
    resid = y - m.predict(X)
    smear = float(np.average(np.exp(resid), weights=w))
    return m, smear


# ====================================================================== 予測側（fold の全組）
def fold_combos(bt: str, win: pd.DataFrame, fold: int, st: pd.DataFrame, pays: pd.DataFrame,
                model, smear) -> pd.DataFrame:
    d = win[win["fold"] == fold].merge(st[["race_id", "horse_no", "prank", "track"]], on=["race_id", "horse_no"], how="left")
    d = d.sort_values(["race_id", "p_reg"], ascending=[True, False])
    d["pos"] = d.groupby("race_id").cumcount()
    rids = d["race_id"].unique()
    N = int(d["pos"].max()) + 1
    shape = (len(rids), N)
    ri = pd.Index(rids).get_indexer(d["race_id"])
    Pe = np.zeros(shape); Pp = np.zeros(shape); RK = np.full(shape, 16); HN = np.zeros(shape, int)
    Pe[ri, d["pos"]] = d["p_reg"].to_numpy(); Pp[ri, d["pos"]] = d["p_pop"].to_numpy()
    RK[ri, d["pos"]] = d["prank"].fillna(16).to_numpy(); HN[ri, d["pos"]] = d["horse_no"].to_numpy()
    n = (Pe > 0).sum(1)
    track = d.drop_duplicates("race_id").set_index("race_id").loc[rids, "track"].to_numpy()
    out = []
    B = 4000
    for s0 in range(0, len(rids), B):
        sl = slice(s0, s0 + B)
        if bt == "place":
            pe = np.zeros_like(Pe[sl]); pp = np.zeros_like(Pp[sl])
            for r in range(pe.shape[0]):
                k = place_slots(n[sl][r]); nn = n[sl][r]
                if k:
                    pe[r, :nn] = harville_topk(Pe[sl][r, :nn], np.zeros(nn), np.full(nn, k))
                    pp[r, :nn] = harville_topk(Pp[sl][r, :nn], np.zeros(nn), np.full(nn, k))
            m = min(TOP, N)
            ce, combos = pe[:, :m], [(i,) for i in range(m)]
            cp = pp[:, :m]
        else:
            ce, combos = harville_all(bt, Pe[sl], Pe[sl] > 0)
            cp, _ = harville_all(bt, Pp[sl], Pp[sl] > 0)
        idx = np.array(combos)                               # (C, K)
        rr = np.repeat(np.arange(ce.shape[0]), len(combos))
        cc = np.tile(np.arange(len(combos)), ce.shape[0])
        valid = (idx[cc] < n[sl][rr, None]).all(1)
        if bt == "place":
            valid &= place_slots(n[sl][rr]) > 0
        rr, cc = rr[valid], cc[valid]
        pos = idx[cc]
        ranks = RK[sl][rr[:, None], pos]
        hns = HN[sl][rr[:, None], pos]
        if bt not in ORDERED:
            o = np.argsort(hns, axis=1)
            hns = np.take_along_axis(hns, o, 1)
            ranks = np.sort(ranks, axis=1)
        pe_c = ce[rr, cc]; pp_c = cp[rr, cc]
        X = features_for(bt, ranks, n[sl][rr], np.log(np.maximum(pp_c, 1e-9)), track[sl][rr])
        odds_hat = np.exp(model.predict(X)) * smear
        df = pd.DataFrame({"race_id": rids[sl][rr], "p_ens": pe_c, "p_pop": pp_c, "odds_hat": odds_hat})
        for i in range(K[bt]):
            df[f"h{i + 1}"] = hns[:, i]
        out.append(df)
    res = pd.concat(out, ignore_index=True)
    names = ["h1", "h2", "h3"][:K[bt]]
    sold = set(pays["race_id"])
    res = res[res["race_id"].isin(sold)]
    res = res.merge(pays, on=["race_id", *names], how="left")
    res["ret"] = res["odds"].fillna(0.0)
    res["fold"] = fold
    res["ev"] = res["p_ens"] * res["odds_hat"]
    res["ev_pop"] = res["p_pop"] * res["odds_hat"]
    return res.drop(columns=["odds"])


def evaluate(res: pd.DataFrame, bt: str) -> pd.DataFrame:
    rows = []
    for col in ("ev", "ev_pop"):
        for t in T_GRID:
            s = res[res[col] >= t]
            for mode in ("all", "top1"):
                ss = s if mode == "all" else s.loc[s.groupby("race_id")[col].idxmax()] if len(s) else s
                for f, g in ss.groupby("fold"):
                    rows.append({"bet": bt, "prob": "アンサンブル" if col == "ev" else "人気だけ", "t": t, "mode": mode,
                                 "fold": f, "n": len(g), "races": g["race_id"].nunique(),
                                 "roi": g["ret"].sum() / len(g) if len(g) else np.nan,
                                 "hit": float((g["ret"] > 0).mean()) if len(g) else np.nan})
    return pd.DataFrame(rows)


def calib(res: pd.DataFrame, bt: str) -> pd.DataFrame:
    q = pd.qcut(res["ev"], 10, labels=False, duplicates="drop")
    g = res.groupby(q).agg(ev_pred=("ev", "mean"), roi=("ret", "mean"), n=("ret", "size"))
    return g.assign(bet=bt).reset_index(drop=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    win = S.load_win()
    bets = sys.argv[1].split(",") if len(sys.argv) > 1 else BETS
    quality, grids, cals = [], [], []
    for bt in bets:
        pays = payouts(bt)
        parts = []
        for f in FOLDS:
            start = pd.Timestamp(R.FOLD_START[f])
            st = starters_with_pop(R.FOLD_START[f])
            tr = training_rows(bt, st, pays, start - pd.DateOffset(years=TRAIN_YEARS), start - pd.Timedelta(days=31))
            model, smear = fit_payout_model(tr, bt)
            # 払戻モデルの当てはまり（その fold の当たり組で。重みなしの素の誤差）
            va = training_rows(bt, st, pays, start, start + pd.DateOffset(months=11))
            Xv = features_for(bt, va[[f"r{i + 1}" for i in range(K[bt])]].to_numpy(), va["n"].to_numpy(),
                              va["logp_pop"].to_numpy(), va["track"].to_numpy())
            pv = model.predict(Xv)
            yv = np.log(va["odds"].to_numpy())
            quality.append({"bet": PAY_JP[bt], "fold": f, "n_train": len(tr), "n_valid": len(va),
                            "r2_log": 1 - np.var(yv - pv) / np.var(yv),
                            "median_abs_log_err": float(np.median(np.abs(yv - pv))), "smear": smear})
            res = fold_combos(bt, win, f, st, pays, model, smear)
            parts.append(res)
            print(PAY_JP[bt], f, len(tr), len(res), f"R2={quality[-1]['r2_log']:.3f}", flush=True)
        res = pd.concat(parts, ignore_index=True)
        res.to_parquet(OUT / f"combos_{bt}.parquet", index=False)
        grids.append(evaluate(res, PAY_JP[bt]))
        cals.append(calib(res, PAY_JP[bt]))
    q = pd.DataFrame(quality); q.to_csv(OUT / "payout_model_quality.csv", index=False)
    g = pd.concat(grids, ignore_index=True); g.to_csv(OUT / "grid.csv", index=False)
    c = pd.concat(cals, ignore_index=True); c.to_csv(OUT / "ev_calibration.csv", index=False)
    summarize(g, q, c)
    return 0


def summarize(g: pd.DataFrame, q: pd.DataFrame, c: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    w = g.pivot_table(index=["bet", "prob", "t", "mode"], columns="fold", values="roi")
    nmin = g.pivot_table(index=["bet", "prob", "t", "mode"], columns="fold", values="races").min(axis=1)
    w.columns = [f"f{c}" for c in w.columns]
    w["mean"] = w.mean(axis=1); w["min"] = w[[f"f{f}" for f in FOLDS]].min(axis=1); w["min_races"] = nmin
    w = w[w["min_races"] >= 30].reset_index()
    w.to_csv(OUT / "candidates.csv", index=False)
    sel = w.sort_values("mean", ascending=False).groupby(["bet", "prob"]).head(1).sort_values(["prob", "mean"], ascending=[True, False])
    sel.to_csv(OUT / "selected.csv", index=False)
    nest = []
    for (bet, prob), gg in w.groupby(["bet", "prob"]):
        r = {}
        for i, f in enumerate(FOLDS[1:], start=1):
            prior = [f"f{p}" for p in FOLDS[:i]]
            b = gg.loc[gg[prior].mean(axis=1).idxmax()]
            r[f"f{f}"] = b[f"f{f}"]
        nest.append({"bet": bet, "prob": prob, **r, "nested_mean": np.mean(list(r.values()))})
    nest = pd.DataFrame(nest); nest.to_csv(OUT / "nested.csv", index=False)
    print("=== 払戻モデルの当てはまり（当たり組の log 払戻、fold 平均）===")
    print(q.groupby("bet")[["r2_log", "median_abs_log_err"]].mean().round(3).to_string())
    print("\n=== EV 十分位ごとの実 ROI（アンサンブル）===")
    for bet, gg in c.groupby("bet"):
        print(bet, " ".join(f"{a:.2f}→{b:.2f}" for a, b in zip(gg["ev_pred"], gg["roi"])))
    print("\n=== fold2〜5 平均 ROI 最大（券種 × 確率）===")
    print(sel.round(3).to_string(index=False))
    print("\n=== nested ===")
    print(nest.round(3).to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
