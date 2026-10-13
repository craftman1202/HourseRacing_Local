"""全券種の ROI 改善策の総当たり（オッズ不要の施策、2026-10-13）。

試す施策（すべての組み合わせを、walk-forward 5-fold で選んで OOS で1回測る）:
  確率の作り方  本番モデル（乖離回帰）と人気だけの勝率を対数で混ぜる  λ ∈ {0, .25, .5, .75, 1}
               （λ=0 が本番モデル、λ=1 が人気だけ）
  買い方       券種ごとの代表的な組み方（1点・流し・BOX・フォーメーション）
  軸の確信度   軸（確率1位）の単勝確率 ≥ x   x ∈ {0, .3, .4, .5, .6}
  レース選び   セグメント1つで絞る: 頭数・競馬場・クラス・馬場・距離・
               「モデルの1位と1番人気が一致するか」
券種: 単勝・複勝・ワイド・馬連・馬単・3連複・3連単・枠連・枠単。1点 100 円の均等買い。

期間: fold1〜5 = 2019〜2023 の各年（walk-forward の valid）、OOS = 2024-02〜。
確率はどちらも学習区間だけで作ったもの（scripts/deviation_models.py walkforward と評価用モデル）。
人気は確定時点の値なので、人気を使う確率は実運用より楽観的。

出力: artifacts/roi_search/race_level_lam{λ}.parquet（レース×買い方の点数と払戻）と、
      candidates.parquet（全候補 × 期間の ROI）。集計と選択は roi_search_report.py。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.eval.place import race_log_softmax  # noqa: E402

OUT = ROOT / "artifacts/roi_search"
DEV = ROOT / "artifacts/dev_flat_pop"
TOP = 9
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
X_GRID = (0.0, 0.3, 0.4, 0.5, 0.6)

BET_JP = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単",
          "trio": "3連複", "trifecta": "3連単", "bracket_q": "枠連", "bracket_e": "枠単"}


# ------------------------------------------------------------------ 買い方（順位の組）
def _p(*xs):
    return [tuple(x) for x in xs]


def structures() -> dict[str, tuple[str, list[tuple[int, ...]]]]:
    """名前 → (券種, 確率順位（0始まり）の組のリスト)。頭数が足りない組は使う側で落とす。"""
    S: dict[str, tuple[str, list]] = {}
    S["単勝 1位"] = ("win", [(0,)])
    S["単勝 1・2位"] = ("win", [(0,), (1,)])
    S["複勝 1位"] = ("place", [(0,)])
    S["複勝 1・2位"] = ("place", [(0,), (1,)])
    S["複勝 2位"] = ("place", [(1,)])
    for bt, jp in (("wide", "ワイド"), ("quinella", "馬連")):
        S[f"{jp} 1点"] = (bt, [(0, 1)])
        for n in (2, 4, 8):
            S[f"{jp} 軸1頭流し 相手{n}頭"] = (bt, [(0, j) for j in range(1, n + 1)])
        S[f"{jp} BOX3頭"] = (bt, list(itertools.combinations(range(3), 2)))
        S[f"{jp} BOX4頭"] = (bt, list(itertools.combinations(range(4), 2)))
    S["馬単 1点"] = ("exacta", [(0, 1)])
    S["馬単 裏表"] = ("exacta", [(0, 1), (1, 0)])
    for n in (2, 4, 8):
        S[f"馬単 1着固定流し 相手{n}頭"] = ("exacta", [(0, j) for j in range(1, n + 1)])
    for n in (2, 4, 7):
        S[f"3連複 軸2頭流し 相手{n}頭"] = ("trio", [(0, 1, k) for k in range(2, n + 2)])
    for n in (3, 5, 8):
        S[f"3連複 軸1頭流し 相手{n}頭"] = ("trio", list((0,) + c for c in itertools.combinations(range(1, n + 1), 2)))
    for n in (3, 4, 5):
        S[f"3連複 BOX{n}頭"] = ("trio", list(itertools.combinations(range(n), 3)))
    for n in (3, 5, 8):
        S[f"3連単 1着固定流し 相手{n}頭"] = ("trifecta", [(0,) + c for c in itertools.permutations(range(1, n + 1), 2)])
    for n in (2, 5):
        S[f"3連単 1・2着固定 3着{n}頭"] = ("trifecta", [(0, 1, k) for k in range(2, n + 2)])
    S["3連単 フォーメーション 1位/上位3/上位7"] = ("trifecta", [(0, j, k) for j in (1, 2) for k in range(1, 7) if k != j])
    S["3連単 BOX3頭"] = ("trifecta", list(itertools.permutations(range(3), 3)))
    S["枠連 1点"] = ("bracket_q", [(0, 1)])
    S["枠連 軸流し 相手3頭"] = ("bracket_q", [(0, j) for j in (1, 2, 3)])
    S["枠単 1点"] = ("bracket_e", [(0, 1)])
    S["枠単 1着固定流し 相手3頭"] = ("bracket_e", [(0, j) for j in (1, 2, 3)])
    return S


ORDERED = {"exacta", "trifecta", "bracket_e"}


# ------------------------------------------------------------------ データ
def load_probs() -> pd.DataFrame:
    def both(name):
        f = pd.read_parquet(DEV / f"oof_win_{name}.parquet")[["race_id", "horse_no", "race_date", "fold", "ensemble"]]
        o = pd.read_parquet(DEV / f"oos_win_{name}.parquet")[["race_id", "horse_no", "race_date", "ensemble"]].assign(fold=0)
        return pd.concat([f, o], ignore_index=True)
    reg = both("reg").rename(columns={"ensemble": "p_reg"})
    pop = both("pop").rename(columns={"ensemble": "p_pop"})[["race_id", "horse_no", "p_pop"]]
    w = reg.merge(pop, on=["race_id", "horse_no"])
    w = w[w["race_id"].str[:2].astype(int) > 4]
    e = pd.read_parquet(ROOT / "data_real/silver/entry.parquet", columns=["race_id", "horse_no", "waku"])
    e["horse_no"] = e["horse_no"].astype(int)
    return w.merge(e, on=["race_id", "horse_no"], how="left")


def race_attrs(win: pd.DataFrame) -> pd.DataFrame:
    """セグメント用のレース属性（すべて発走前に分かるもの）。"""
    g = pd.read_parquet(ROOT / "data_real/gold/features_noodds_pop_cat/features.parquet",
                        columns=["race_id", "field_size", "distance", "c_tier", "c_going"])
    g = g.drop_duplicates("race_id").set_index("race_id")
    r = win.groupby("race_id").agg(fold=("fold", "first"), race_date=("race_date", "first"))
    top_reg = win.loc[win.groupby("race_id")["p_reg"].idxmax()].set_index("race_id")["horse_no"]
    top_pop = win.loc[win.groupby("race_id")["p_pop"].idxmax()].set_index("race_id")["horse_no"]
    r["agree"] = np.where(top_reg.reindex(r.index) == top_pop.reindex(r.index), "一致", "不一致")
    r = r.join(g)
    fs = r["field_size"]
    r["頭数"] = pd.cut(fs, [0, 6, 9, 12, 99], labels=["〜6頭", "7〜9頭", "10〜12頭", "13頭〜"]).astype(str)
    r["競馬場"] = r.index.str[:2]
    r["クラス"] = r["c_tier"].map({4.0: "A", 3.0: "B", 2.0: "C", 1.0: "D"}).fillna("その他")
    r["馬場"] = np.where(r["c_going"] == 0, "良", np.where(r["c_going"].notna(), "稍重〜不良", "不明"))
    r["距離"] = pd.cut(r["distance"], [0, 1200, 1600, 9999], labels=["〜1200m", "1300〜1600m", "1700m〜"]).astype(str)
    r["一致"] = r["agree"]
    return r[["fold", "race_date", "頭数", "競馬場", "クラス", "馬場", "距離", "一致"]]


def payout_dicts() -> dict[str, dict]:
    pay = pd.read_parquet(ROOT / "data_real/silver/payout.parquet",
                          columns=["race_id", "bet_type", "comb_1", "comb_2", "comb_3", "payout_yen"])
    out = {}
    for bt, jp in BET_JP.items():
        d = pay[pay["bet_type"] == jp]
        k = {"win": 1, "place": 1, "wide": 2, "quinella": 2, "exacta": 2, "bracket_q": 2,
             "bracket_e": 2, "trio": 3, "trifecta": 3}[bt]
        cols = ["comb_1", "comb_2", "comb_3"][:k]
        d = d.dropna(subset=cols)
        arr = d[cols].astype(int).to_numpy()
        if bt not in ORDERED and k > 1:
            arr = np.sort(arr, axis=1)
        dd: dict = {}
        for rid, combo, y in zip(d["race_id"], map(tuple, arr), d["payout_yen"]):
            dd.setdefault(rid, {})
            dd[rid][combo] = dd[rid].get(combo, 0) + int(y)
        out[bt] = dd
    return out


# ------------------------------------------------------------------ レース単位の集計
def race_level(win: pd.DataFrame, lam: float, pays: dict) -> pd.DataFrame:
    rid = win["race_id"].to_numpy()
    lp = (1 - lam) * np.log(np.clip(win["p_reg"].to_numpy(), 1e-12, 1)) + \
        lam * np.log(np.clip(win["p_pop"].to_numpy(), 1e-12, 1))
    win = win.assign(p=np.exp(race_log_softmax(lp, rid)))
    S = structures()
    rows = []
    for r, g in win.sort_values(["race_id", "p"], ascending=[True, False]).groupby("race_id", sort=False):
        horse = g["horse_no"].to_numpy()[:TOP]
        waku = g["waku"].to_numpy()[:TOP]
        m = len(horse)
        p1 = float(g["p"].iloc[0])
        for name, (bt, combos) in S.items():
            dd = pays[bt].get(r)
            if dd is None:                       # その券種の発売が無い
                continue
            src = waku if bt.startswith("bracket") else horse
            tickets = set()
            for c in combos:
                if max(c) >= m:
                    continue
                t = tuple(int(src[i]) for i in c)
                if bt not in ORDERED and len(t) > 1:
                    t = tuple(sorted(t))
                if bt.startswith("bracket") and any(np.isnan(x) for x in t):
                    continue
                tickets.add(t)
            if not tickets:
                continue
            ret = sum(dd.get(t, 0) for t in tickets)
            rows.append((r, name, len(tickets), ret, p1))
    return pd.DataFrame(rows, columns=["race_id", "structure", "n", "ret", "p1"])


def main() -> int:
    lam = float(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    win = load_probs()
    attrs = race_attrs(win)
    attrs.to_parquet(OUT / "race_attrs.parquet")
    pays = payout_dicts()
    rl = race_level(win, lam, pays)
    rl["lam"] = lam
    rl.to_parquet(OUT / f"race_level_lam{lam}.parquet", index=False)
    print(f"λ={lam}: {len(rl):,} 行（{rl.race_id.nunique():,} レース × {rl.structure.nunique()} 買い方）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
