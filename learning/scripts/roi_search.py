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

import os  # noqa: E402

# DNF=1: 競走中止・失格の馬（走ったのに予測行が無い）をレースに戻す版（2026-10-13）。
# 予測行は完走馬だけで作られており、レース後にしか分からない「誰が完走したか」が
# 買い目から中止馬を除く形で漏れていた（scripts/bet_selector_diag.py で発見）。
DNF = os.environ.get("DNF") == "1"
OUT = ROOT / ("artifacts/roi_search_dnf2" if DNF else "artifacts/roi_search")
DEV = ROOT / "artifacts/dev_flat_pop"
TOP = 9
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
X_GRID = (0.0, 0.3, 0.4, 0.5, 0.6)

BET_JP = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単",
          "trio": "3連複", "trifecta": "3連単", "bracket_q": "枠連", "bracket_e": "枠単"}


# ------------------------------------------------------------------ 買い方（順位の組）
# 定義は運用と共有（nar.eval.recommend、2026-10-13 に移設。内容は同じ）
from nar.eval.recommend import structures  # noqa: E402,F401

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
    w = w.merge(e, on=["race_id", "horse_no"], how="left")
    return add_dnf(w) if DNF else w


FOLD_START = {1: "2019-02-01", 2: "2020-02-01", 3: "2021-02-01", 4: "2022-02-01", 5: "2023-02-01",
              0: "2024-02-01"}
POP_SMOOTH, MAX_RANK = 20.0, 16


def starters() -> pd.DataFrame:
    """走った馬（完走 + 競走中止 + 失格）。取消・除外は人気が無く、馬券は返還なので含めない。"""
    e = pd.read_parquet(ROOT / "data_real/silver/entry.parquet",
                        columns=["race_id", "race_date", "horse_no", "waku", "finish_pos", "is_win",
                                 "popularity", "margin"])
    e = e[e["finish_pos"].notna() | e["margin"].isin(["競走中止", "失格"])].copy()
    e["horse_no"] = e["horse_no"].astype(int)
    e["race_date"] = pd.to_datetime(e["race_date"])
    e["n"] = e.groupby("race_id")["horse_no"].transform("size")
    e["is_win"] = e["is_win"].fillna(0)
    return e


def pop_table_before(st: pd.DataFrame, start: str) -> dict:
    """人気 × 走った頭数の勝率表（start より前だけ）。deviation_models.fit_pop_table と同じ収縮。"""
    d = st[st["race_date"] < pd.Timestamp(start)]
    r = d["popularity"].fillna(d["n"]).clip(1, MAX_RANK).astype(int)
    g = d.assign(r=r).groupby(["r", "n"])["is_win"].agg(["sum", "count"]).reset_index()
    p = (g["sum"] + POP_SMOOTH / g["n"]) / (g["count"] + POP_SMOOTH)
    return {f"{int(a)}_{int(b)}": float(v) for a, b, v in zip(g["r"], g["n"], p)}


def add_dnf(w: pd.DataFrame) -> pd.DataFrame:
    """中止・失格馬を戻し、人気の勝率 p_pop を走った頭数で引き直す。

    中止馬の p_reg（本番モデル）は無い（学習・予測とも完走馬だけ）。近似として中止馬には
    その馬の p_pop を与え、完走馬の p_reg は比を保ったまま残りの確率を分け合う。
    """
    from nar.eval.deviation import log_p_pop

    st_all = starters()                       # 勝率表は全期間の過去分から（fold ごとに開始日より前）
    st = st_all[st_all["race_id"].isin(set(w["race_id"]))]
    meta = w.drop_duplicates("race_id").set_index("race_id")[["fold", "race_date"]]
    add = st.merge(w[["race_id", "horse_no"]].assign(_in=1), on=["race_id", "horse_no"], how="left")
    add = add[add["_in"].isna()]
    add = pd.DataFrame({"race_id": add["race_id"], "horse_no": add["horse_no"],
                        "race_date": add["race_id"].map(meta["race_date"]),
                        "fold": add["race_id"].map(meta["fold"]), "p_reg": np.nan,
                        "waku": add["waku"]})
    w = pd.concat([w.drop(columns=["p_pop"]), add], ignore_index=True)
    w = w.merge(st[["race_id", "horse_no", "popularity", "n"]], on=["race_id", "horse_no"], how="left")
    w["p_pop"] = np.nan
    for f, start in FOLD_START.items():
        m = (w["fold"] == f).to_numpy()
        tab = pop_table_before(st_all, start)
        w.loc[m, "p_pop"] = np.exp(log_p_pop(w.loc[m, "popularity"].to_numpy(dtype=float),
                                             w.loc[m, "n"].to_numpy(), w.loc[m, "race_id"].to_numpy(), tab))
    miss = w["p_reg"].isna()
    w.loc[miss, "p_reg"] = w.loc[miss, "p_pop"]
    dnf_mass = w["p_reg"].where(miss, 0.0).groupby(w["race_id"]).transform("sum")
    w.loc[~miss, "p_reg"] = w.loc[~miss, "p_reg"] * (1 - dnf_mass[~miss])
    w["dnf"] = miss.astype(int)
    return w.drop(columns=["popularity", "n"])


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
def race_level(win: pd.DataFrame, lam: float, pays: dict, names: list[str] | None = None) -> pd.DataFrame:
    rid = win["race_id"].to_numpy()
    lp = (1 - lam) * np.log(np.clip(win["p_reg"].to_numpy(), 1e-12, 1)) + \
        lam * np.log(np.clip(win["p_pop"].to_numpy(), 1e-12, 1))
    win = win.assign(p=np.exp(race_log_softmax(lp, rid)))
    S = structures()
    if names is not None:
        S = {k: S[k] for k in names}
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
