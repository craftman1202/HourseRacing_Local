"""全モデルアンサンブル（all_models.py）の確率で、全券種の買い方を fold だけで選ぶ（2026-10-17）。

  確率      π = softmax((1−λ)·log p_ens + λ·log p_pop)、λ ∈ {0, .25, .5, .75, 1}
            p_ens: アンサンブル（ensemble_wf: fold k の重みは fold < k の OOF から）
            p_pop: 人気×走った頭数の勝率表（fold 開始日より前のデータだけ、roi_search.pop_table_before）
  買い方    roi_search.structures() の 42 通り（1点・流し・BOX・フォーメーション、枠含む）
  軸の確信  軸（π の1位）の π ≥ x、x ∈ {0, .3, .4, .5, .6}
  セグメントでの絞り込みはしない（roi_search_dnf2 で OOS 再現しないと分かっている）

ラベル・レースは中止馬リークを塞いだ版（all_models のデータは競走中止・失格の馬を含む）。
アンサンブルが fold2〜5 にしか無いので、選択は fold2〜5。
  - 「fold2〜5 平均 ROI 最大」の候補を券種ごとに1つ
  - nested: fold k の買い方を fold < k だけで選んだときの fold k の ROI（手順の汎化性能）
  - 同じ手順を λ=1（人気だけ）に限ったものと比べ、モデルが人気に何を足しているかを見る
OOS はここでは触らない（all_models.py oos の後に strategy_oos で1回だけ測る）。

出力: artifacts/strategy_ens/{race_level_lam*.parquet, candidates.csv, selected.csv, nested.csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_search as R  # noqa: E402

ROOT = R.ROOT
OUT = ROOT / "artifacts/strategy_ens"
SRC = ROOT / "artifacts/all_models/pop/oof_all.parquet"
FOLDS = (2, 3, 4, 5)
MIN_RACES = 30


def load_win(src: Path = SRC, prob_col: str = "ensemble_wf") -> pd.DataFrame:
    from nar.eval.deviation import log_p_pop

    o = pd.read_parquet(src)
    o = o[o[prob_col].notna()].copy()
    e = pd.read_parquet(ROOT / "data_real/silver/entry.parquet", columns=["race_id", "horse_no", "waku", "popularity"])
    e["horse_no"] = e["horse_no"].astype(int)
    o = o.merge(e, on=["race_id", "horse_no"], how="left")
    o["n"] = o.groupby("race_id")["horse_no"].transform("size")
    st = R.starters()
    o["p_pop"] = np.nan
    for f in sorted(o["fold"].unique()):
        m = (o["fold"] == f).to_numpy()
        tab = R.pop_table_before(st, R.FOLD_START[int(f)])
        o.loc[m, "p_pop"] = np.exp(log_p_pop(o.loc[m, "popularity"].to_numpy(dtype=float), o.loc[m, "n"].to_numpy(),
                                             o.loc[m, "race_id"].to_numpy(), tab))
    o = o.rename(columns={prob_col: "p_reg"})
    return o[["race_id", "horse_no", "race_date", "fold", "p_reg", "p_pop", "waku", "is_win", "finish_pos", "dnf"]]


def candidates(rl_all: pd.DataFrame) -> pd.DataFrame:
    rl = rl_all.copy()
    rl["stake"] = rl["n"] * 100.0
    rl["bet"] = rl["structure"].map({k: v[0] for k, v in R.structures().items()})
    parts = []
    for x in R.X_GRID:
        d = rl[rl["p1"] >= x]
        g = d.groupby(["lam", "structure", "bet", "fold"]).agg(stake=("stake", "sum"), ret=("ret", "sum"),
                                                              races=("race_id", "size"),
                                                              hits=("ret", lambda s: int((s > 0).sum()))).reset_index()
        g["x"] = x
        parts.append(g)
    long = pd.concat(parts, ignore_index=True)
    long["roi"] = long["ret"] / long["stake"]
    w = long.pivot_table(index=["bet", "structure", "lam", "x"], columns="fold", values=["roi", "races", "hits"])
    w.columns = [f"{a}_f{b}" for a, b in w.columns]
    w = w.reset_index()
    rc = [f"roi_f{f}" for f in FOLDS]
    ok = (w[[f"races_f{f}" for f in FOLDS]].min(axis=1) >= MIN_RACES)
    w = w[ok].copy()
    w["mean_roi"] = w[rc].mean(axis=1)
    w["min_roi"] = w[rc].min(axis=1)
    w["folds_gt1"] = (w[rc] > 1).sum(axis=1)
    w["hit_rate"] = sum(w[f"hits_f{f}"] for f in FOLDS) / sum(w[f"races_f{f}"] for f in FOLDS)
    return w


def nested(w: pd.DataFrame, lam_only: float | None = None) -> pd.DataFrame:
    rows = []
    for bet, g in w.groupby("bet"):
        if lam_only is not None:
            g = g[g["lam"] == lam_only]
        if g.empty:
            continue
        res = {}
        for i, f in enumerate(FOLDS[1:], start=1):
            prior = [f"roi_f{p}" for p in FOLDS[:i]]
            b = g.loc[g[prior].mean(axis=1).idxmax()]
            res[f"f{f}"] = b[f"roi_f{f}"]
        res["nested_mean"] = float(np.mean(list(res.values())))
        rows.append({"bet": R.BET_JP[bet], "lam": "1.0のみ" if lam_only is not None else "全λ", **res})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    win = load_win()
    pays = R.payout_dicts()
    rls = []
    for lam in R.LAMBDAS:
        path = OUT / f"race_level_lam{lam}.parquet"
        if path.exists():
            rl = pd.read_parquet(path)
        else:
            rl = R.race_level(win, lam, pays).assign(lam=lam)
            rl.to_parquet(path, index=False)
        rls.append(rl)
        # 2026-10-17 追加の買い方（軸1頭＋相手3〜5頭）。既存のキャッシュに無いものだけ計算する
        extra = [k for k in R.structures() if k not in set(rl["structure"])]
        if extra:
            ep = OUT / f"race_level_extra_lam{lam}.parquet"
            if ep.exists() and set(pd.read_parquet(ep, columns=["structure"])["structure"]) >= set(extra):
                ex = pd.read_parquet(ep)
            else:
                ex = R.race_level(win, lam, pays, names=extra).assign(lam=lam)
                ex.to_parquet(ep, index=False)
            rls.append(ex)
    meta = win.drop_duplicates("race_id").set_index("race_id")[["fold", "race_date"]]
    rl_all = pd.concat(rls, ignore_index=True).join(meta, on="race_id")
    w = candidates(rl_all)
    w.to_csv(OUT / "candidates.csv", index=False)
    sel = (w.sort_values("mean_roi", ascending=False).groupby("bet").head(1)
           .assign(券種=lambda d: d["bet"].map(R.BET_JP)))
    cols = ["券種", "structure", "lam", "x", *[f"roi_f{f}" for f in FOLDS], "mean_roi", "min_roi", "folds_gt1", "hit_rate"]
    sel[cols].to_csv(OUT / "selected.csv", index=False)
    pop_only = (w[w["lam"] == 1.0].sort_values("mean_roi", ascending=False).groupby("bet").head(1)
                .assign(券種=lambda d: d["bet"].map(R.BET_JP)))
    nest = pd.concat([nested(w), nested(w, lam_only=1.0)], ignore_index=True)
    nest.to_csv(OUT / "nested.csv", index=False)
    pd.set_option("display.width", 250)
    print("=== fold2〜5 平均 ROI 最大（券種ごと、全 λ）===")
    print(sel[cols].round(3).to_string(index=False))
    print("\n=== 同じ、人気だけ（λ=1）===")
    print(pop_only[cols].round(3).to_string(index=False))
    print("\n=== nested（fold k を fold<k で選んだときの fold k の ROI）===")
    print(nest.round(3).to_string(index=False))
    flow = w[w["structure"].str.contains("軸1頭流し 相手[345]頭|1着固定流し 相手[345]頭|軸1頭マルチ")
             & w["bet"].isin(["wide", "trio", "trifecta"])]
    fb = flow.sort_values("mean_roi", ascending=False).groupby("structure").head(1).sort_values(["bet", "mean_roi"], ascending=[True, False])
    fb.to_csv(OUT / "flow_patterns.csv", index=False)
    print("\n=== 軸1頭＋相手3〜5頭（ワイド・3連複・3連単、買い方ごとに最良の λ・x）===")
    print(fb.assign(券種=fb["bet"].map(R.BET_JP))[cols].round(3).to_string(index=False))
    print("\n全候補", len(w), " fold2〜5 平均>1:", int((w["mean_roi"] > 1).sum()), " 4fold 全部>1:", int((w["folds_gt1"] == 4).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
