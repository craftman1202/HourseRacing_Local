"""券種ごとの「買う／見送る」専用モデル（メタモデル、2026-10-13）。

roi_search の結論: セグメントを手で組み合わせて 5-fold で最良を選ぶと OOS で崩れる（選択の偶然）。
ここでは手のルールの代わりに、レースの特徴量から「その買い方をこのレースで買ったときの
期待払戻率」を学習する正則化した GBDT を券種・買い方ごとに作る。

  期待払戻率 = P(的中) × E[払戻率 | 的中]
    P(的中)            LightGBM 二値分類
    E[払戻率 | 的中]   LightGBM 回帰（log 払戻率、的中レースだけ）→ exp に分散補正
  特徴量（すべて運用時に出馬表と締切前オッズから作れるもの）:
    確率上位 p1〜p3・p1−p2・エントロピー・頭数・距離・競馬場・月・
    買い方の Harville 的中確率・点数・モデルの1位と人気1位の一致・モデルと人気の p1 の差

学習は walk-forward: fold k の予測は fold 1〜k−1 で学習（k=2〜5）、OOS の予測は fold 1〜5 で学習。
「期待払戻率 ≥ τ なら買う」の τ は fold 2〜5 の予測（全てアウトオブサンプル）で選び、OOS は1回だけ測る。
比較対象は同じ買い方の手のルール（軸の単勝確率 ≥ x、x も fold 2〜5 で選ぶ）。

対象の買い方: roi_search の候補のうち、券種ごとに 5-fold 平均 ROI 上位 2 つの（買い方, λ）
（セグメントで絞らない候補から）。
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402

from nar.eval.place import harville_topk, place_slots, race_log_softmax  # noqa: E402

ROOT = R.ROOT
# DNF=1（roi_search.DNF）: 競走中止・失格を戻した版。頭数は「走った頭数」を使う
# （gold の field_size は出走取消・除外も数えた登録頭数で、予測行との差が中止馬の数＝
#  レース後の情報になっていた）。
OUT = ROOT / ("artifacts/bet_selector_dnf2" if R.DNF else "artifacts/bet_selector")
FEATS = ["p1", "p2", "p3", "gap12", "entropy", "field_size", "distance", "track", "month",
         "p_hit", "n", "agree", "reg_minus_pop"]
MIN_RACES = 150
N_BOOT = 2000
PARAMS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=0,
              deterministic=True, force_row_wise=True)
ROUNDS = 300


def targets_from_search() -> list[tuple[str, str, float]]:
    c = pd.read_parquet(R.OUT / "candidates.parquet")
    c = c[(c["seg"] == "全体") & (c["lam"].isin([0.75, 1.0]))]
    best = (c.sort_values("fold_mean_roi", ascending=False)
            .drop_duplicates(["bet", "structure", "lam"]).groupby("bet").head(2))
    return [(r.bet, r.structure, float(r.lam)) for r in best.itertuples()]


def race_features(win: pd.DataFrame, lam: float, structs: list[str]) -> pd.DataFrame:
    """レース単位の特徴量と、指定した買い方の Harville 的中確率。"""
    rid = win["race_id"].to_numpy()
    lp = (1 - lam) * np.log(np.clip(win["p_reg"].to_numpy(), 1e-12, 1)) + \
        lam * np.log(np.clip(win["p_pop"].to_numpy(), 1e-12, 1))
    win = win.assign(p=np.exp(race_log_softmax(lp, rid)))
    S = R.structures()
    rows = []
    for r, g in win.sort_values(["race_id", "p"], ascending=[True, False]).groupby("race_id", sort=False):
        p = g["p"].to_numpy()
        n = len(p)
        m = min(n, R.TOP)
        pp = p[:m]
        top_reg = g["p_reg"].to_numpy()[0]
        top_pop = g["p_pop"].to_numpy()[0]
        ex = {(i, j): pp[i] * pp[j] / max(1 - pp[i], 1e-12) for i, j in itertools.permutations(range(m), 2)}
        tri = {(i, j, l): ex[(i, j)] * pp[l] / max(1 - pp[i] - pp[j], 1e-12)
               for i, j, l in itertools.permutations(range(m), 3)}
        k = int(place_slots(n))
        plc = harville_topk(p, np.zeros(n), np.full(n, k))[:m] if k else np.zeros(m)
        feat = {"race_id": r, "p1": pp[0], "p2": pp[1] if m > 1 else 0.0, "p3": pp[2] if m > 2 else 0.0,
                "gap12": pp[0] - (pp[1] if m > 1 else 0.0),
                "entropy": float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
                "reg_minus_pop": float(top_reg - top_pop)}
        for name in structs:
            bt, combos = S[name]
            ph = 0.0
            for c in combos:
                if max(c) >= m:
                    continue
                if bt == "win":
                    ph += pp[c[0]]
                elif bt == "place":
                    ph += plc[c[0]]
                elif bt == "exacta":
                    ph += ex[c]
                elif bt == "quinella":
                    ph += ex[c] + ex[(c[1], c[0])]
                elif bt == "trifecta":
                    ph += tri[c]
                elif bt == "trio":
                    ph += sum(tri[q] for q in itertools.permutations(c))
                elif bt == "wide":
                    ph += sum(v for q, v in tri.items() if c[0] in q and c[1] in q)
                else:                      # 枠は馬の確率からの集約が要るので持たない
                    ph = np.nan
                    break
            feat[f"phit::{name}"] = ph
        rows.append(feat)
    return pd.DataFrame(rows).set_index("race_id")


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    X, Xt = tr[FEATS], te[FEATS]
    clf = lgb.train({**PARAMS, "objective": "binary"},
                    lgb.Dataset(X, label=tr["hit"], categorical_feature=["track"]), ROUNDS)
    ph = clf.predict(Xt)
    h = tr[tr["hit"] == 1]
    ly = np.log(h["rr"].to_numpy())
    reg = lgb.train({**PARAMS, "objective": "regression", "min_data_in_leaf": 200},
                    lgb.Dataset(h[FEATS], label=ly, categorical_feature=["track"]), ROUNDS)
    resid_var = float(np.var(ly - reg.predict(h[FEATS])))
    pay = np.exp(reg.predict(Xt) + resid_var / 2)        # 対数正規の平均への補正
    return ph * pay


def roi_of(d: pd.DataFrame) -> tuple[float, int]:
    return (float(d["ret"].sum() / d["stake"].sum()) if len(d) else np.nan), int(len(d))


def boot(d: pd.DataFrame) -> tuple[float, float]:
    g = d.groupby("race_date").agg(ret=("ret", "sum"), stake=("stake", "sum"))
    r, s = g["ret"].to_numpy(), g["stake"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(g), size=(N_BOOT, len(g)))
    bs = r[idx].sum(1) / s[idx].sum(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    win = R.load_probs()
    attrs = pd.read_parquet(R.OUT / "race_attrs.parquet")
    gold = pd.read_parquet(ROOT / "data_real/gold/features_noodds_pop_cat/features.parquet",
                           columns=["race_id", "distance"]).drop_duplicates("race_id").set_index("race_id")
    gold["field_size"] = win.groupby("race_id").size().reindex(gold.index)
    targets = targets_from_search()
    print("対象:", targets)
    feats_by_lam = {}
    for lam in sorted({t[2] for t in targets}):
        structs = [s for _, s, l in targets if l == lam]
        cache = OUT / f"race_features_lam{lam}.parquet"
        if cache.exists():
            feats_by_lam[lam] = pd.read_parquet(cache)
        else:
            feats_by_lam[lam] = race_features(win, lam, structs)
            feats_by_lam[lam].to_parquet(cache)
    results, preds_all = [], []
    for bet, sname, lam in targets:
        rl = pd.read_parquet(R.OUT / f"race_level_lam{lam}.parquet")
        rl = rl[rl["structure"] == sname].drop(columns=["p1"]).set_index("race_id")
        f = feats_by_lam[lam]
        d = rl.join(f, how="inner").join(attrs[["fold", "race_date", "一致"]]).join(
            gold[["field_size", "distance"]])
        d["p_hit"] = d[f"phit::{sname}"]
        d["track"] = d.index.str[:2].astype(int)
        d["month"] = pd.to_datetime(d["race_date"]).dt.month
        d["agree"] = (d["一致"] == "一致").astype(int)
        d["stake"] = d["n"] * 100.0
        d["rr"] = d["ret"] / d["stake"]
        d["hit"] = (d["ret"] > 0).astype(int)
        d = d.reset_index()
        pred = pd.Series(np.nan, index=d.index)
        for k in (2, 3, 4, 5, 0):
            tr = d[(d["fold"] > 0) & (d["fold"] < k)] if k else d[d["fold"] > 0]
            te = d[d["fold"] == k]
            pred[te.index] = fit_predict(tr, te)
        d["pred"] = pred
        cv = d[d["fold"].between(2, 5)]
        oos = d[d["fold"] == 0]
        # メタモデル: τ を fold2〜5 で選ぶ
        best = None
        for tau in np.round(np.arange(0.6, 2.01, 0.05), 2):
            roi, n = roi_of(cv[cv["pred"] >= tau])
            if n >= MIN_RACES * 4 and (best is None or roi > best[1]):
                best = (tau, roi, n)
        # 手のルール: 軸の単勝確率 ≥ x を fold2〜5 で選ぶ
        hand = None
        for x in (0.0, 0.3, 0.4, 0.5, 0.6):
            roi, n = roi_of(cv[cv["p1"] >= x])
            if n >= MIN_RACES * 4 and (hand is None or roi > hand[1]):
                hand = (x, roi, n)
        sm = oos[oos["pred"] >= best[0]]
        sh = oos[oos["p1"] >= hand[0]]
        rm, nm = roi_of(sm)
        rh, nh = roi_of(sh)
        lom, him = boot(sm) if nm else (np.nan, np.nan)
        loh, hih = boot(sh) if nh else (np.nan, np.nan)
        # 予測の当てはまり（OOS、予測の 5 分位ごとの実 ROI）
        q = pd.qcut(oos["pred"], 5, labels=False, duplicates="drop")
        calib = oos.groupby(q).apply(lambda g: pd.Series({"pred": g["pred"].mean(), "roi": g["ret"].sum() / g["stake"].sum()}))
        results.append({
            "券種": R.BET_JP[bet], "買い方": sname, "λ": lam,
            "メタ τ": best[0], "メタ cv_roi": best[1], "メタ cv_races": best[2],
            "メタ OOS": rm, "メタ OOS races": nm, "メタ ci": f"[{lom:.2f}, {him:.2f}]",
            "手 x": hand[0], "手 cv_roi": hand[1], "手 cv_races": hand[2],
            "手 OOS": rh, "手 OOS races": nh, "手 ci": f"[{loh:.2f}, {hih:.2f}]",
            "OOS 予測5分位の実ROI": "/".join(f"{v:.2f}" for v in calib["roi"]),
            "OOS 予測と実ROIの順位相関": float(oos["pred"].corr(oos["rr"], method="spearman")),
        })
        preds_all.append(d.assign(bet=bet, structure=sname, lam=lam)[
            ["race_id", "bet", "structure", "lam", "fold", "race_date", "pred", "p1", "n", "ret", "stake"]])
        print(results[-1])
    res = pd.DataFrame(results)
    res.to_csv(OUT / "comparison.csv", index=False)
    pd.concat(preds_all).to_parquet(OUT / "predictions.parquet", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 28)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
