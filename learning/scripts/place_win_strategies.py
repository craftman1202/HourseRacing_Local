"""単勝モデル × 複勝モデルの組み合わせベット戦略の ROI 比較。

入力（すべて保存済みのものを読むだけ。OOS を開封しない）:
  artifacts/oos_predictions.parquet          単勝モデル（評価用配布物）の OOS 予測 `ensemble`
  artifacts/place/oos_place_predictions.parquet  複勝モデルの OOS 予測 `pp_ensemble`
                                              と、単勝モデルを Harville で変換した `pp_from_win`
  data_real/silver/odds.parquet              単勝オッズ（確定値）
  data_real/bronze/odds/                     複勝オッズの下限・上限（確定値）
  data_real/silver/payout.parquet            実払戻（単勝・複勝）

期間はオッズのある期間だけ。平地は 2026-02-02〜2026-09-02、ばんえいは帯広ばのみ
オッズがあり 2026-02-02〜2026-08-31。前半（〜05-31）で閾値を選び、後半はその閾値で
1回測るだけにする。

系統は `PLACE_FAMILY` 環境変数で切り替える（既定は平地 flat）。ばんえいは件数が
1桁少ないので `MIN_DEV_BETS` を下げてある（`place_model.py` と同じ規約）。

**オッズはどちらも確定オッズ**（締切前には手に入らない値）。ROI はその分だけ楽観的。
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.transform import silver as S  # noqa: E402

FAMILY = os.environ.get("PLACE_FAMILY", "flat")
if FAMILY not in ("flat", "banei"):
    raise ValueError(f"PLACE_FAMILY は flat/banei のいずれか（受領: {FAMILY!r}）")

_PATHS = {
    "flat": dict(out="artifacts/place", win_oos_pred="artifacts/oos_predictions.parquet",
                min_dev_bets=300),
    "banei": dict(out="artifacts/place_banei",
                 win_oos_pred="artifacts/banei/oos_predictions.parquet",
                 min_dev_bets=30),
}[FAMILY]

OUT = ROOT / _PATHS["out"]
WIN_OOS_PRED = ROOT / _PATHS["win_oos_pred"]
DEV_END = pd.Timestamp("2026-05-31")
T_GRID = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0)
X_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_DEV_BETS = _PATHS["min_dev_bets"]
N_BOOT = 2000


# --------------------------------------------------------------------- 読み込み
def place_odds() -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / "data_real/bronze/odds/ym=*/*.parquet")))
    b = pd.concat([pd.read_parquet(f, columns=["競馬場", "競走年月日", "レース番号", "賭式",
                                               "番号1", "オッズ", "オッズ（最大）"])
                   for f in fs])
    b = b[b["賭式"] == "複勝"]
    master = S.TrackMaster()
    baba = master.codes(b["競馬場"])
    ymd = S._ymd(b["競走年月日"])
    rn = S._digits(b["レース番号"]).astype("Int64")
    out = pd.DataFrame({
        "race_id": [S.make_race_id(x, d, r) for x, d, r in zip(baba, ymd, rn)],
        "horse_no": S._digits(b["番号1"]).astype("Int64").to_numpy(),
        "pl_min": S._digits(b["オッズ"]).to_numpy(),
        "pl_max": S._digits(b["オッズ（最大）"]).to_numpy(),
    })
    out = out[(out["pl_min"] > 0) & out["horse_no"].notna()]
    return out.drop_duplicates(["race_id", "horse_no"])


def realized(bet_type: str) -> pd.DataFrame:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "payout_yen"])
    pay = pay[pay["bet_type"] == bet_type]
    return (pay.groupby(["race_id", "comb_1"])["payout_yen"].sum() / 100.0).rename(
        "ret_win" if bet_type == "単勝" else "ret_place").reset_index().rename(
        columns={"comb_1": "horse_no"})


def load() -> pd.DataFrame:
    win = pd.read_parquet(WIN_OOS_PRED,
                          columns=["race_id", "horse_no", "race_date", "is_win", "ensemble"])
    plc = pd.read_parquet(OUT / "oos_place_predictions.parquet",
                          columns=["race_id", "horse_no", "n_runners", "k_place", "is_place",
                                   "pp_ensemble", "pp_from_win"])
    odds = pd.read_parquet(ROOT / "data_real/silver/odds.parquet",
                           columns=["race_id", "horse_no", "odds_win", "popularity"])
    df = (win.rename(columns={"ensemble": "pw"})
          .merge(plc.rename(columns={"pp_ensemble": "pp"}), on=["race_id", "horse_no"])
          .merge(odds, on=["race_id", "horse_no"])
          .merge(place_odds(), on=["race_id", "horse_no"], how="left")
          .merge(realized("単勝"), on=["race_id", "horse_no"], how="left")
          .merge(realized("複勝"), on=["race_id", "horse_no"], how="left"))
    df["ret_win"] = df["ret_win"].fillna(0.0)
    df["ret_place"] = df["ret_place"].fillna(0.0)
    df["race_date"] = pd.to_datetime(df["race_date"])
    # オッズが付いているレースは全馬に付いている前提。欠けたレースは丸ごと落とす
    full = df.groupby("race_id")["odds_win"].transform("size") == df["n_runners"]
    df = df[full].reset_index(drop=True)
    df["period"] = np.where(df["race_date"] <= DEV_END, "dev", "test")
    return df


# ----------------------------------------------------------------------- 戦略
def add_ev(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["pl_est"] = df["pl_min"] + alpha * (df["pl_max"] - df["pl_min"])
    df["ev_win"] = df["pw"] * df["odds_win"]
    df["ev_pl_min"] = df["pp"] * df["pl_min"]
    df["ev_pl_est"] = df["pp"] * df["pl_est"]
    df["ev_pl_est_h"] = df["pp_from_win"] * df["pl_est"]
    return df


def bets(df: pd.DataFrame, family: str, t: float, x: float) -> pd.DataFrame:
    """(race_date, stake, ret, hit) のベット一覧。1ベット 1 単位（100円）。"""
    has_pl = df["pl_min"].notna()

    def win(mask):
        s = df[mask]
        return pd.DataFrame({"race_date": s["race_date"], "ret": s["ret_win"],
                             "hit": s["is_win"] == 1, "kind": "win"})

    def place(mask):
        s = df[mask & has_pl]
        return pd.DataFrame({"race_date": s["race_date"], "ret": s["ret_place"],
                             "hit": s["ret_place"] > 0, "kind": "place"})

    ev_w, ev_p = df["ev_win"], df["ev_pl_est"]
    if family == "W: 単勝EV":
        return win((ev_w >= t) & (df["pw"] >= x))
    if family == "P: 複勝EV(下限オッズ)":
        return place((df["ev_pl_min"] >= t) & (df["pp"] >= x))
    if family == "P: 複勝EV(推定オッズ)":
        return place((ev_p >= t) & (df["pp"] >= x))
    if family == "Ph: 複勝EV(単勝モデル→Harville)":
        return place((df["ev_pl_est_h"] >= t) & (df["pp_from_win"] >= x))
    if family == "W|P: 単勝EV & 複勝確率≥x":
        return win((ev_w >= t) & (df["pp"] >= x))
    if family == "P|W: 複勝EV & 単勝確率≥x":
        return place((ev_p >= t) & (df["pw"] >= x))
    if family == "WP: 単複とも EV≥t なら両方":
        m = (ev_w >= t) & (ev_p >= t) & (df["pp"] >= x)
        return pd.concat([win(m), place(m)])
    if family == "Max: EVの高い方を1点":
        pick_w = ev_w >= ev_p.fillna(-np.inf)
        prob = np.where(pick_w, df["pw"], df["pp"])
        ev = np.where(pick_w, ev_w, ev_p)
        m = (ev >= t) & (prob >= x)
        return pd.concat([win(m & pick_w), place(m & ~pick_w)])
    raise ValueError(family)


FAMILIES = ("W: 単勝EV", "P: 複勝EV(下限オッズ)", "P: 複勝EV(推定オッズ)",
            "Ph: 複勝EV(単勝モデル→Harville)", "W|P: 単勝EV & 複勝確率≥x",
            "P|W: 複勝EV & 単勝確率≥x", "WP: 単複とも EV≥t なら両方", "Max: EVの高い方を1点")


def summarize(b: pd.DataFrame) -> dict:
    n = len(b)
    if n == 0:
        return {"n_bets": 0, "roi": np.nan, "hit_rate": np.nan, "n_days": 0}
    return {"n_bets": n, "roi": float(b["ret"].sum() / n), "hit_rate": float(b["hit"].mean()),
            "n_days": int(b["race_date"].nunique())}


def day_bootstrap(b: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """開催日ブロックブートストラップ（同日のレースは相関するので日単位で引く）。"""
    g = b.groupby("race_date").agg(ret=("ret", "sum"), n=("ret", "size"))
    r, n = g["ret"].to_numpy(), g["n"].to_numpy()
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    return r[idx].sum(1) / n[idx].sum(1)


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def main() -> int:
    df = load()
    dev_placers = df[(df["period"] == "dev") & (df["ret_place"] > 0) & df["pl_min"].notna()
                     & (df["pl_max"] > df["pl_min"])]
    alpha = float(((dev_placers["ret_place"] - dev_placers["pl_min"])
                   / (dev_placers["pl_max"] - dev_placers["pl_min"])).clip(0, 1).mean())
    df = add_ev(df, alpha)
    rng = np.random.default_rng(0)

    info = {"alpha_place_odds": alpha,
            "rows": {p: int((df["period"] == p).sum()) for p in ("dev", "test")},
            "races": {p: int(df.loc[df["period"] == p, "race_id"].nunique())
                      for p in ("dev", "test")},
            "dates": {p: [str(df.loc[df["period"] == p, "race_date"].min().date()),
                          str(df.loc[df["period"] == p, "race_date"].max().date())]
                      for p in ("dev", "test")}}

    # 較正の確認（期間別）: 複勝確率の予測平均と実現率
    calib = []
    for p, g in df.groupby("period"):
        for col in ("pp", "pp_from_win"):
            bins = pd.qcut(g[col], 10, duplicates="drop")
            c = g.groupby(bins, observed=True).agg(pred=(col, "mean"), obs=("is_place", "mean"),
                                                   n=("is_place", "size"))
            calib.append(c.assign(period=p, model=col).reset_index(drop=True))
    pd.concat(calib).to_csv(OUT / "strategy_place_calibration.csv", index=False)

    # 基準線
    base = []
    d = {p: g for p, g in df.groupby("period")}
    for p, g in d.items():
        fav = g[g["popularity"] == 1]
        top_w = g.loc[g.groupby("race_id")["pw"].idxmax()]
        top_p = g.loc[g.groupby("race_id")["pp"].idxmax()]
        has = g["pl_min"].notna()
        rows = {
            "全馬 単勝": (g["ret_win"], g["is_win"] == 1),
            "全馬 複勝": (g.loc[has, "ret_place"], g.loc[has, "ret_place"] > 0),
            "1番人気 単勝": (fav["ret_win"], fav["is_win"] == 1),
            "1番人気 複勝": (fav["ret_place"], fav["ret_place"] > 0),
            "単勝モデル1位 単勝": (top_w["ret_win"], top_w["is_win"] == 1),
            "複勝モデル1位 複勝": (top_p["ret_place"], top_p["ret_place"] > 0),
        }
        for name, (r, h) in rows.items():
            base.append({"period": p, "baseline": name, "n_bets": len(r),
                         "roi": float(r.mean()), "hit_rate": float(h.mean())})
    base = pd.DataFrame(base)

    # グリッド（dev / test の両方を出すが、選択は dev のみで行う）
    grid = []
    for fam in FAMILIES:
        for t in T_GRID:
            for x in X_GRID:
                for p, g in d.items():
                    grid.append({"family": fam, "t": t, "x": x, "period": p,
                                 **summarize(bets(g, fam, t, x))})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "strategy_grid.csv", index=False)

    dev = grid[(grid["period"] == "dev") & (grid["n_bets"] >= MIN_DEV_BETS)]
    chosen = dev.loc[dev.groupby("family")["roi"].idxmax()]
    rows = []
    for _, c in chosen.iterrows():
        bt = bets(d["test"], c["family"], c["t"], c["x"])
        s = summarize(bt)
        boot = day_bootstrap(bt, rng) if len(bt) else np.array([np.nan])
        rows.append({"family": c["family"], "t": c["t"], "x": c["x"],
                     "dev_n": int(c["n_bets"]), "dev_roi": c["roi"], "dev_hit": c["hit_rate"],
                     "test_n": s["n_bets"], "test_roi": s["roi"], "test_hit": s["hit_rate"],
                     "test_ci_lo": float(np.nanquantile(boot, 0.025)),
                     "test_ci_hi": float(np.nanquantile(boot, 0.975)),
                     # H0: ROI ≤ 1 の片側ブートストラップ p 値
                     "p_roi_gt1": float((np.sum(boot <= 1.0) + 1) / (len(boot) + 1))})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res["p_roi_gt1"].to_numpy())
    m = len(res)
    # Bonferroni 同時 95% 区間（family 数ぶん）
    for i, r in res.iterrows():
        bt = bets(d["test"], r["family"], r["t"], r["x"])
        boot = day_bootstrap(bt, np.random.default_rng(1))
        res.loc[i, "test_ci_bonf_lo"] = float(np.quantile(boot, 0.025 / m))
        res.loc[i, "test_ci_bonf_hi"] = float(np.quantile(boot, 1 - 0.025 / m))
    res = res.sort_values("test_roi", ascending=False)
    res.to_csv(OUT / "strategy_selected.csv", index=False)
    base.to_csv(OUT / "strategy_baselines.csv", index=False)
    (OUT / "strategy_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 250)
    print(json.dumps(info, ensure_ascii=False))
    print(base.pivot(index="baseline", columns="period", values=["n_bets", "roi"]).round(3))
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
