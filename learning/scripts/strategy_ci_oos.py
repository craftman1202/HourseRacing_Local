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


def stats(s: pd.DataFrame, stake: str) -> dict:
    """s: race_id, ret, stake。開催日単位ブートストラップ。"""
    if s.empty:
        return {}
    g = s.assign(day=s["race_id"].map(days)).groupby("day").agg(ret=("ret", "sum"), st=(stake, "sum"))
    pt, med, lo, hi = C.boot(g["ret"].to_numpy(float), g["st"].to_numpy(float), rng, 10000)
    yearly = s.assign(y=s["race_id"].map(days).dt.year).groupby("y").apply(
        lambda x: round(x["ret"].sum() / x[stake].sum(), 3), include_groups=False).to_dict()
    return {"OOS_ROI": round(pt, 3), "OOS_CI": f"[{lo:.2f},{hi:.2f}]", "OOS_races": s["race_id"].nunique(),
            "OOS_的中率": round(float(s.groupby("race_id")["ret"].sum().gt(0).mean()), 3), "OOS年別": yearly}


rows, win_oos, pays = [], None, None
for r in sel.itertuples():
    if not isinstance(r.family, str):
        rows.append({"bet": r.bet, "採用": "なし"}); continue
    bt = BT[r.bet]
    if r.family == "EV":
        d = pd.read_parquet(C.OOSD / f"combos_{bt}.parquet", columns=["race_id", "p_ens", "ev", "ret"])
        s = d[(d["ev"] >= r.t - 1e-9) & (d["p_ens"] >= r.p_min - 1e-9)]
        if r.mode == "top1":
            s = s.sort_values("ev", ascending=False).drop_duplicates("race_id")
        out = stats(s.assign(stake=1.0)[["race_id", "ret", "stake"]], "stake")
    else:
        if win_oos is None:
            win_oos = S.load_win(C.OOSD / "oos_ensemble.parquet", "ensemble")
            pays = R.payout_dicts()
        rl = R.race_level(win_oos, r.lam, pays, names=[r.structure])
        s = rl[rl["p1"] >= r.x - 1e-9]
        out = stats(s.assign(stake=s["n"] * 100.0)[["race_id", "ret", "stake"]], "stake")
    rows.append({"bet": r.bet, "選択期間ROI": round(r.roi, 3), "選択期間CI": f"[{r.ci_lo:.2f},{r.ci_hi:.2f}]", **out})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(C.OUT / "oos_check.csv", index=False)
