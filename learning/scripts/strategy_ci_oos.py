"""strategy_ci_select.py が選んだ戦略を OOS で確認する（選択には使わない）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402
import strategy_ci_select as C  # noqa: E402
import strategy_select as S  # noqa: E402

BT = {"単勝": "win", "複勝": "place", "馬連": "quinella", "馬単": "exacta", "ワイド": "wide",
      "3連複": "trio", "3連単": "trifecta", "枠連": "bracket_q", "枠単": "bracket_e"}
sel = pd.read_csv(C.OUT / "selected.csv")
rng = np.random.default_rng(1)
days = C.day_map()
rows = []
win_oos = None
for r in sel.itertuples():
    if not isinstance(r.family, str):
        rows.append({"bet": r.bet, "採用": "なし"}); continue
    bt = BT[r.bet]
    if r.family == "EV":
        d = pd.read_parquet(C.OOSD / f"combos_{bt}.parquet", columns=["race_id", "p_ens", "ev", "ret", "fold"])
        s = d[(d["ev"] >= r.t) & (d["p_ens"] >= r.p_min)]
        if r.mode == "top1":
            s = s.sort_values("ev", ascending=False).drop_duplicates("race_id")
        s = s.assign(n=1)
        out = C.summarize(s[["race_id", "ret", "n"]], days, 1.0, rng)
        yearly = s.assign(y=s["race_id"].map(days).dt.year).groupby("y").apply(lambda x: round(x["ret"].sum() / len(x), 3), include_groups=False).to_dict()
    else:
        if win_oos is None:
            win_oos = S.load_win(C.OOSD / "oos_ensemble.parquet", "ensemble")
            pays = R.payout_dicts()
        rl = R.race_level(win_oos, r.lam, pays, names=[r.structure])
        s = rl[rl["p1"] >= r.x]
        out = C.summarize(s[["race_id", "ret", "n"]], days, 100.0, rng)
        yearly = s.assign(y=s["race_id"].map(days).dt.year).groupby("y").apply(lambda x: round(x["ret"].sum() / (100.0 * x["n"].sum()), 3), include_groups=False).to_dict()
    rows.append({"bet": r.bet, "選択期間ROI": round(r.roi, 3), "選択期間CI下限": round(r.ci_lo, 3),
                 **({} if out is None else {"OOS_ROI": round(out["roi"], 3), "OOS_CI": f"[{out['ci_lo']:.2f},{out['ci_hi']:.2f}]",
                                             "OOS_races": out["races"], "OOS_的中率": round(out["hit_rate"], 3)}), "OOS年別": yearly})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(C.OUT / "oos_check.csv", index=False)
