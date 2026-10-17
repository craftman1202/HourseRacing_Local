"""事前登録した買い方の OOS 評価（2026-10-17、artifacts/strategy_ens/preregistration.md）。

OOS（2024-02-01〜）の予測は all_models.py oos で作ったもの（開封は oos_access.log に記録済み）。
ここでは買い方を選び直さない。登録した規則をそのまま当てて ROI・的中率・日単位ブロック
ブートストラップ 95%CI・上位払戻 10 件を除いた ROI を出す。

  A. オッズ不要の規則（strategy_select の selected）
  B. 予測払戻 × EV の規則（payout_ev、払戻モデルは 2024-02-01 の 8 年前〜31 日前で学習）
  C. B と同じ規則を、2026-02〜の実オッズ（確定値）で EV を出して当てたもの
比較: 人気だけ（λ=1）の同じ手順、旧本番モデル（dev reg）の roi_search_dnf2 OOS 値
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_models as A  # noqa: E402
import payout_ev as PE  # noqa: E402
import roi_search as R  # noqa: E402
import strategy_select as S  # noqa: E402

from nar.eval import metrics  # noqa: E402
from nar.models.ensemble import ConstrainedStacker  # noqa: E402

OUT = R.ROOT / "artifacts/strategy_ens/oos"
MODELS = ("tabm", "lgbm", "catboost")
ODDS_FREE = [("複勝 1位", 0.5, 0.6), ("単勝 1位", 0.5, 0.6), ("枠単 1点", 0.0, 0.6), ("枠連 1点", 0.0, 0.6),
             ("馬単 1点", 0.25, 0.6), ("馬連 1点", 0.25, 0.6), ("ワイド 1点", 0.0, 0.6),
             ("3連単 1・2着固定 3着2頭", 0.0, 0.6), ("3連複 BOX3頭", 0.0, 0.5)]
EV_RULES = [("trio", 2.0, "top1"), ("quinella", 2.0, "top1"), ("wide", 2.0, "top1"),
            ("exacta", 2.0, "top1"), ("trifecta", 2.0, "all"), ("place", 1.3, "top1"), ("win", 2.0, "all")]
N_BOOT = 4000


def oos_frame() -> pd.DataFrame:
    base = None
    for m in MODELS:
        f = pd.read_parquet(A.OUT / "pop/oos" / f"{m}.parquet")
        base = f if base is None else base.merge(f[["race_id", "horse_no", m]], on=["race_id", "horse_no"])
    w = json.loads((A.OUT / "pop/ensemble.json").read_text())["all_fold_weights"]
    ww = np.array([w[m] for m in MODELS]); ww = ww / ww.sum()
    st = ConstrainedStacker(list(MODELS)); st.weights = ww
    base = base.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    base["ensemble"] = st.predict_proba(base[list(MODELS)], base["race_id"].to_numpy())
    base["fold"] = 0
    return base


def boot(ret: np.ndarray, stake: np.ndarray, day: np.ndarray) -> tuple[float, float]:
    d = pd.DataFrame({"r": ret, "s": stake, "d": day}).groupby("d").sum()
    r, s = d["r"].to_numpy(), d["s"].to_numpy()
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    bs = r[idx].sum(1) / s[idx].sum(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def summary(name: str, ret: np.ndarray, stake: np.ndarray, day: np.ndarray, races: int) -> dict:
    if len(ret) == 0:
        return {"規則": name, "races": 0}
    lo, hi = boot(ret, stake, day)
    trimmed = (ret.sum() - np.sort(ret)[::-1][:10].sum()) / stake.sum()
    unit = 100.0 if stake.max() >= 100 else 1.0
    return {"規則": name, "races": races, "点数": int(stake.sum() / unit), "ROI": ret.sum() / stake.sum(),
            "CI_lo": lo, "CI_hi": hi, "上位10除く": trimmed, "的中率": float((ret > 0).mean())}


ONLY_EV = "--only-ev" in sys.argv


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    o = oos_frame()
    o.to_parquet(OUT / "oos_ensemble.parquet", index=False)
    rid, y = o["race_id"].to_numpy(), o["is_win"].to_numpy()
    mets = {m: metrics.summary(o[m].to_numpy(), y, o["finish_pos"].to_numpy(), rid) for m in (*MODELS, "ensemble")}
    win = S.load_win(OUT / "oos_ensemble.parquet", "ensemble")
    mets["popularity_only"] = metrics.summary(win["p_pop"].to_numpy(), win["is_win"].to_numpy(),
                                              win["finish_pos"].to_numpy(), win["race_id"].to_numpy())
    (A.OUT / "pop/oos_summary.json").write_text(json.dumps(mets, default=float, indent=1))
    day = win.drop_duplicates("race_id").set_index("race_id")["race_date"]
    rows = []

    # A. オッズ不要
    pays = {} if ONLY_EV else R.payout_dicts()
    for lam in ([] if ONLY_EV else sorted({l for _, l, _ in ODDS_FREE} | {1.0})):
        names = sorted({s for s, l, _ in ODDS_FREE if l == lam} | ({s for s, _, _ in ODDS_FREE} if lam == 1.0 else set()))
        rl = R.race_level(win, lam, pays, names=names)
        for s, l, x in ODDS_FREE:
            if l == lam:
                d = rl[(rl["structure"] == s) & (rl["p1"] >= x)]
                rows.append({"区分": "A オッズ不要", **summary(f"{s} λ={l} 軸≥{x}", d["ret"].to_numpy(float),
                                                            d["n"].to_numpy(float) * 100, d["race_id"].map(day).to_numpy(), len(d))})
            if lam == 1.0:
                d = rl[(rl["structure"] == s) & (rl["p1"] >= x)]
                rows.append({"区分": "比較: 人気だけ", **summary(f"{s} λ=1 軸≥{x}", d["ret"].to_numpy(float),
                                                             d["n"].to_numpy(float) * 100, d["race_id"].map(day).to_numpy(), len(d))})

    # B / C. EV（予測払戻 / 実オッズ）
    start = pd.Timestamp(R.FOLD_START[0])
    st = PE.starters_with_pop(R.FOLD_START[0])
    from benter_ev import load_odds
    real = load_odds(set(win["race_id"]))
    alpha = 0.257
    for bt, t, mode in EV_RULES:
        cf = OUT / f"combos_{bt}.parquet"
        if cf.exists():
            res = pd.read_parquet(cf)
        else:
            payouts = PE.payouts(bt)
            tr = PE.training_rows(bt, st, payouts, start - pd.DateOffset(years=PE.TRAIN_YEARS), start - pd.Timedelta(days=31))
            model, smear = PE.fit_payout_model(tr, bt)
            res = PE.fold_combos(bt, win, 0, st, payouts, model, smear)
            res.to_parquet(cf, index=False)
        for label, col, df in [("B 予測払戻 EV", "ev", res)]:
            sel = df[df[col] >= t]
            if mode == "top1" and len(sel):
                sel = sel.loc[sel.groupby("race_id")[col].idxmax()]
            rows.append({"区分": label, **summary(f"{PE.PAY_JP[bt]} EV≥{t} {mode}", sel["ret"].to_numpy(float),
                                                  np.ones(len(sel)), sel["race_id"].map(day).to_numpy(), sel["race_id"].nunique())})
        # 実オッズ（2026-02〜）
        names = ["h1", "h2", "h3"][:PE.K[bt]]
        ro = real.get({"win": "win", "place": "place", "quinella": "quinella", "exacta": "exacta",
                       "wide": "wide", "trio": "trio", "trifecta": "trifecta"}[bt])
        if ro is not None and len(ro):
            ro = ro.copy()
            ro["o_real"] = np.where(ro["omax"].notna() & (ro["omax"] > ro["omin"]),
                                    ro["omin"] + alpha * (ro["omax"] - ro["omin"]), ro["omin"]) \
                if bt in ("place", "wide") else ro["omin"]
            m2 = res.merge(ro[["race_id", *names, "o_real"]], on=["race_id", *names], how="inner")
            m2["ev_real"] = m2["p_ens"] * m2["o_real"]
            for lab, period in [("C 実オッズ EV（2026-02〜05）", m2["race_id"].map(day) <= pd.Timestamp("2026-05-31")),
                                ("C 実オッズ EV（2026-06〜）", m2["race_id"].map(day) > pd.Timestamp("2026-05-31")),
                                ("B' 予測払戻 EV（同じ 2026-02〜 のレース）", pd.Series(True, index=m2.index))]:
                col = "ev" if lab.startswith("B'") else "ev_real"
                dd = m2[period.to_numpy()]
                sel = dd[dd[col] >= t]
                if mode == "top1" and len(sel):
                    sel = sel.loc[sel.groupby("race_id")[col].idxmax()]
                rows.append({"区分": lab, **summary(f"{PE.PAY_JP[bt]} EV≥{t} {mode}", sel["ret"].to_numpy(float),
                                                    np.ones(len(sel)), sel["race_id"].map(day).to_numpy(), sel["race_id"].nunique())})
        print(bt, "done", flush=True)
    r = pd.DataFrame(rows)
    if ONLY_EV and (OUT / "oos_results.csv").exists():
        prev = pd.read_csv(OUT / "oos_results.csv")
        r = pd.concat([prev[prev["区分"].str.startswith(("A ", "比較"))], r], ignore_index=True)
    r.to_csv(OUT / "oos_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(json.dumps({k: round(v["race_nll"], 5) for k, v in mets.items()}))
    print(r.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
