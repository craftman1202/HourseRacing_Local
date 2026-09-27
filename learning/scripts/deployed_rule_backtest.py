"""運用に載せたルール（nar.eval.place.max_ev_bets）そのものでの ROI。

EV の高い方を1点・EV≥1・的中確率≥0.6・券種ごと1レース1万円のフルケリー。
保存済みの OOS 予測を読むだけ（OOS を開封しない）。
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

from nar.eval.place import max_ev_bets  # noqa: E402


def main() -> int:
    info = json.loads((OUT / "strategy_info.json").read_text())
    df = add_ev(load(), info["alpha_place_odds"])
    rows = []
    for period, g in df.groupby("period"):
        g = g.reset_index(drop=True)
        b = max_ev_bets(g["race_id"].to_numpy(), g["pw"], g["odds_win"], g["pp"], g["pl_est"])
        m = b["stake_yen"] > 0
        ret = np.where(b["bet_type"] == "単勝", g["ret_win"], g["ret_place"]) * b["stake_yen"]
        bets = pd.DataFrame({"race_date": g.loc[m, "race_date"], "race_id": g.loc[m, "race_id"],
                             "stake": b.loc[m, "stake_yen"], "ret": ret[m],
                             "bet_type": b.loc[m, "bet_type"]})
        for kind, part in [("合計", bets), *bets.groupby("bet_type")]:
            s = summarize(part)
            bs = boot(part)
            rows.append({"period": period, "kind": kind, **s,
                         "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)),
                         "avg_stake": s["stake"] / max(s["n_bets"], 1),
                         "per_day_stake": s["stake"] / max(part["race_date"].nunique(), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "deployed_rule_backtest.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
