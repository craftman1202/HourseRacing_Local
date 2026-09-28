"""「EV の低い方が1以上・的中確率も条件を満たすとき」に賭ける対照戦略の ROI。

本番戦略（`nar.eval.place.max_ev_bets`、EV の高い方を選ぶ）と条件は完全に対称にし、
違いは高低の選び方だけにする。保存済みの OOS 予測を読むだけ（OOS を開封しない）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kelly_strategies import boot, summarize  # noqa: E402
from place_win_strategies import OUT, add_ev, load  # noqa: E402

from nar.eval.place import max_ev_bets, max_ev_bets_with_fallback, min_ev_bets  # noqa: E402


def backtest(rule, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, g in df.groupby("period"):
        g = g.reset_index(drop=True)
        b = rule(g["race_id"].to_numpy(), g["pw"], g["odds_win"], g["pp"], g["pl_est"])
        m = b["stake_yen"] > 0
        ret = np.where(b["bet_type"] == "単勝", g["ret_win"], g["ret_place"]) * b["stake_yen"]
        bets = pd.DataFrame({"race_date": g.loc[m, "race_date"], "race_id": g.loc[m, "race_id"],
                             "stake": b.loc[m, "stake_yen"], "ret": ret[m],
                             "bet_type": b.loc[m, "bet_type"]})
        for kind, part in [("合計", bets), *bets.groupby("bet_type")]:
            s = summarize(part)
            bs = boot(part) if len(part) else np.array([np.nan])
            rows.append({"period": period, "kind": kind, **s,
                         "ci_lo": float(np.nanquantile(bs, .025)),
                         "ci_hi": float(np.nanquantile(bs, .975)),
                         "avg_stake": s["stake"] / max(s["n_bets"], 1),
                         "per_day_stake": s["stake"] / max(part["race_date"].nunique(), 1)})
    return pd.DataFrame(rows)


def main() -> int:
    info = json.loads((OUT / "strategy_info.json").read_text())
    df = add_ev(load(), info["alpha_place_odds"])

    max_res = backtest(max_ev_bets, df).assign(strategy="EVの高い方（本番）")
    min_res = backtest(min_ev_bets, df).assign(strategy="EVの低い方（対照）")
    fb_res = backtest(max_ev_bets_with_fallback, df).assign(strategy="高い方→確率不足なら低い方へfallback")
    res = pd.concat([max_res, min_res, fb_res], ignore_index=True)
    res.to_csv(OUT / "min_ev_backtest.csv", index=False)

    # フォールバックが実際にどれだけ起きたか（高い方は確率不足で見送り、低い方に切り替えた件数）
    fb_rows = []
    for period, g in df.groupby("period"):
        g = g.reset_index(drop=True)
        b = max_ev_bets_with_fallback(g["race_id"].to_numpy(), g["pw"], g["odds_win"],
                                      g["pp"], g["pl_est"])
        fb = b[b["fallback"]]
        ret = np.where(fb["bet_type"] == "単勝", g.loc[fb.index, "ret_win"],
                       g.loc[fb.index, "ret_place"]) * fb["stake_yen"]
        fb_rows.append({"period": period, "n_fallback_bets": len(fb),
                        "fallback_stake": int(fb["stake_yen"].sum()),
                        "fallback_roi": float((ret.sum() / fb["stake_yen"].sum())
                                              if fb["stake_yen"].sum() else np.nan)})
    print("\n--- フォールバックで成立したベットだけの内訳 ---")
    print(pd.DataFrame(fb_rows).round(3).to_string(index=False))

    pd.set_option("display.width", 250)
    cols = ["strategy", "period", "kind", "n_bets", "n_races", "stake", "ret", "roi",
           "profit", "hit_rate", "ci_lo", "ci_hi", "avg_stake", "per_day_stake"]
    print(res[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
