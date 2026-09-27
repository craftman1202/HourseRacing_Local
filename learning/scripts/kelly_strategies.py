"""単勝・複勝それぞれ独立の予算でケリー配分したときの ROI。

ルール（1レースあたり）:
  - 単勝: EV = p_win × 単勝オッズ ≥ 1 かつ p_win ≥ x_win の馬に賭ける
  - 複勝: EV = p_place × 推定複勝オッズ ≥ 1 かつ p_place ≥ x_place の馬に賭ける
  - 金額: ケリー比率 f = (p·o − 1)/(o − 1) × 予算 1万円（券種ごと）。同一レースで
          Σf > 1 のときは Σf = 1 に縮める。100円単位に切り捨て、100円未満は見送り
  - x_win / x_place は 0.0〜0.9 の 10% 刻み。dev（〜2026-05-31）で選び、test で1回測る

推定複勝オッズは place_win_strategies と同じ（下限 + α(上限−下限)、α は dev の実払戻から）。
オッズは確定値なので ROI は楽観側に偏る。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from place_win_strategies import OUT, add_ev, bh, load  # noqa: E402

BUDGET = 10_000
UNIT = 100
X_GRID = tuple(round(0.1 * i, 1) for i in range(10))
KELLY_SCALES = (1.0, 0.25)       # フルケリーと 1/4 ケリー（感度確認）
MIN_DEV_BETS = 100
N_BOOT = 2000


def kelly_stakes(df: pd.DataFrame, p: str, o: str, x: float, scale: float) -> pd.Series:
    ok = (df[p] * df[o] >= 1.0) & (df[p] >= x) & df[o].notna() & (df[o] > 1.0)
    f = np.where(ok, (df[p] * df[o] - 1.0) / (df[o] - 1.0), 0.0) * scale
    f = pd.Series(np.clip(f, 0.0, None), index=df.index)
    tot = f.groupby(df["race_id"]).transform("sum")
    f = np.where(tot > 1.0, f / tot, f)
    return pd.Series(np.floor(f * BUDGET / UNIT) * UNIT, index=df.index)


def run(df: pd.DataFrame, kind: str, x: float, scale: float) -> pd.DataFrame:
    if kind == "win":
        stake = kelly_stakes(df, "pw", "odds_win", x, scale)
        ret = stake * df["ret_win"]
    else:
        stake = kelly_stakes(df, "pp", "pl_est", x, scale)
        ret = stake * df["ret_place"]
    m = stake > 0
    return pd.DataFrame({"race_date": df.loc[m, "race_date"], "race_id": df.loc[m, "race_id"],
                         "stake": stake[m], "ret": ret[m]})


def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"n_bets": 0, "n_races": 0, "stake": 0.0, "ret": 0.0, "roi": np.nan,
                "profit": 0.0, "max_dd": 0.0, "hit_rate": np.nan}
    daily = b.groupby("race_date")[["stake", "ret"]].sum().sort_index()
    eq = (daily["ret"] - daily["stake"]).cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:] - eq))
    return {"n_bets": len(b), "n_races": int(b["race_id"].nunique()),
            "stake": float(b["stake"].sum()), "ret": float(b["ret"].sum()),
            "roi": float(b["ret"].sum() / b["stake"].sum()),
            "profit": float(b["ret"].sum() - b["stake"].sum()), "max_dd": dd,
            "hit_rate": float((b["ret"] > 0).mean())}


def boot(b: pd.DataFrame, seed: int = 0) -> np.ndarray:
    g = b.groupby("race_date")[["stake", "ret"]].sum()
    s, r = g["stake"].to_numpy(), g["ret"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(N_BOOT, len(g)))
    return r[idx].sum(1) / s[idx].sum(1)


def main() -> int:
    info = json.loads((OUT / "strategy_info.json").read_text())
    df = add_ev(load(), info["alpha_place_odds"])
    d = {p: g for p, g in df.groupby("period")}

    grid = []
    for scale in KELLY_SCALES:
        for kind in ("win", "place"):
            for x in X_GRID:
                for p, g in d.items():
                    grid.append({"kelly": scale, "kind": kind, "x": x, "period": p,
                                 **summarize(run(g, kind, x, scale))})
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "kelly_grid.csv", index=False)

    rows = []
    for scale in KELLY_SCALES:
        pick = {}
        for kind in ("win", "place"):
            dev = grid[(grid.kelly == scale) & (grid.kind == kind) & (grid.period == "dev")
                       & (grid.n_bets >= MIN_DEV_BETS)]
            pick[kind] = dev.loc[dev["roi"].idxmax()]
        parts = {k: run(d["test"], k, pick[k]["x"], scale) for k in pick}
        parts["win+place"] = pd.concat([parts["win"], parts["place"]])
        dev_parts = {k: run(d["dev"], k, pick[k]["x"], scale) for k in pick}
        dev_parts["win+place"] = pd.concat([dev_parts["win"], dev_parts["place"]])
        for k, bt in parts.items():
            s, sd = summarize(bt), summarize(dev_parts[k])
            bs = boot(bt) if len(bt) else np.array([np.nan])
            rows.append({"kelly": scale, "kind": k,
                         "x_win": pick["win"]["x"] if k != "place" else None,
                         "x_place": pick["place"]["x"] if k != "win" else None,
                         "dev_roi": sd["roi"], "dev_stake": sd["stake"], "dev_profit": sd["profit"],
                         "test_roi": s["roi"], "test_ci_lo": float(np.nanquantile(bs, .025)),
                         "test_ci_hi": float(np.nanquantile(bs, .975)),
                         "p_roi_gt1": float((np.sum(bs <= 1) + 1) / (len(bs) + 1)),
                         "test_bets": s["n_bets"], "test_races": s["n_races"],
                         "test_stake": s["stake"], "test_profit": s["profit"],
                         "test_max_dd": s["max_dd"], "test_hit": s["hit_rate"]})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res["p_roi_gt1"].to_numpy())
    res.to_csv(OUT / "kelly_selected.csv", index=False)

    pd.set_option("display.width", 250)
    for scale in KELLY_SCALES:
        print(f"\n=== kelly x{scale} : dev / test ROI by cutoff ===")
        print(grid[grid.kelly == scale].pivot_table(index="x", columns=["kind", "period"],
                                                   values=["roi", "n_bets"]).round(3))
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
