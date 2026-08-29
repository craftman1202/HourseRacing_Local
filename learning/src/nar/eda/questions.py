"""EDA 第一・三〜六問。

設計書 §7 の「モデルを作る前に必ず答えを出しておく質問リスト」を、答えが数値で
返る関数にしたもの。第二問（リーク検証）だけは分量が多いので leakage.py に分けた。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.economic import inverse_odds_sum, NOMINAL_TAKEOUT


# ---------------------------------------------------------------- Q1 カバレッジ
def coverage_map(entry: pd.DataFrame, race: pd.DataFrame,
                 track_names: dict[int, str] | None = None) -> pd.DataFrame:
    """年×競馬場のレース数と主要列の欠損率。どの年から使えるかをここで決める。"""
    e = entry.copy()
    e["year"] = pd.to_datetime(e["race_date"]).dt.year
    candidates = [c for c in ("weight_kg", "time_sec", "sire_sk", "odds_win") if c in e.columns]
    agg = {"race_id": ("race_id", "nunique"), "n_entries": ("race_id", "size")}
    # 欠損率と併せて欠損「件数」を持つ。年で束ねるときに行数で重み付けするために要る
    # （場ごとの率を単純に max/mean すると、小さい場の1年で全場の年が落ちる）。
    agg |= {f"{c}_null_pct": (c, lambda s: 100.0 * s.isna().mean()) for c in candidates}
    agg |= {f"{c}_null_n": (c, lambda s: int(s.isna().sum())) for c in candidates}
    out = e.groupby(["year", "baba_code"]).agg(**agg).reset_index()
    out = out.rename(columns={"race_id": "n_races"})
    if track_names:
        out.insert(2, "track", out["baba_code"].map(track_names))
    return out.round(3)


# 市場情報。トラックA（オッズ非使用）の学習開始年を決める条件には入れない。
# NAR のオッズ配信は 2026-02 以降しか無く、これを条件に含めると 28 年分すべてが
# 「使えない年」になる。カバレッジは別項目として報告する。
MARKET_COLS = ("odds_win", "popularity")


def usable_from_year(coverage: pd.DataFrame, max_null_pct: float = 10.0,
                     min_races_per_year: int | None = None,
                     exclude_cols: tuple[str, ...] = MARKET_COLS) -> dict:
    """学習の主対象をどの年から取るかの提案。

    設計書は2010年以降を想定しているが、その根拠を実測で置き換えるのがここの役目。

    レース数の下限は絶対値で持たない。開催規模は年代で変わるうえ、絶対値を置くと
    データ規模が変わったときに黙って「該当年なし」を返す。中央値の半分を下限にして、
    「他の年に比べて明らかに薄い年」を落とす相対基準にする。
    """
    null_cols = [c for c in coverage.columns
                 if c.endswith("_null_pct")
                 and c.removesuffix("_null_pct") not in exclude_cols]
    # 年の判定は行数で重み付けした全体欠損率で行う。場ごとの率の最大値を採ると、
    # 小さい場の1年（実データでは 2021 年の水沢 13.3%）だけで、その年より前の
    # 全場・全年が学習対象から外れる。年全体では 2.8% でしかない。
    # 場ごとの悪化は by_track_worst に別途残して、年の採否とは分けて見る。
    count_cols = {c: c.replace("_null_pct", "_null_n") for c in null_cols
                  if c.replace("_null_pct", "_null_n") in coverage.columns}
    grouped = coverage.groupby("year")
    by_year = grouped.agg(n_races=("n_races", "sum"),
                          n_entries=("n_entries", "sum")).reset_index()
    for pct_col, n_col in count_cols.items():
        by_year[pct_col] = (grouped[n_col].sum().to_numpy()
                            / by_year["n_entries"].to_numpy() * 100.0)
        by_year[pct_col.replace("_null_pct", "_worst_track_pct")] = (
            grouped[pct_col].max().to_numpy())
    for pct_col in null_cols:                      # 件数が無い列は従来どおり
        if pct_col not in by_year.columns:
            by_year[pct_col] = grouped[pct_col].max().to_numpy()
    # 判定からは外すが、どの年にオッズがあるかは残す
    market = {}
    for col in exclude_cols:
        pct, n = f"{col}_null_pct", f"{col}_null_n"
        if pct in coverage.columns and n in coverage.columns:
            by_year[pct] = (grouped[n].sum().to_numpy()
                            / by_year["n_entries"].to_numpy() * 100.0)
            market[col] = by_year[["year", pct]].round(2)
    if min_races_per_year is None:
        min_races_per_year = int(by_year["n_races"].median() * 0.5)

    ok = by_year[
        (by_year["n_races"] >= min_races_per_year)
        & (by_year[null_cols].max(axis=1) <= max_null_pct)
    ]
    # 一度基準を満たしたら以降は満たし続ける年を採る。途中の1年だけ薄い場合に
    # 学習開始年が不必要に後ろへ動かないようにする。
    proposed = None
    if len(ok):
        years = sorted(by_year["year"])
        ok_years = set(ok["year"])
        for i, y in enumerate(years):
            if all(yy in ok_years for yy in years[i:]):
                proposed = int(y)
                break
    return {
        "proposed_train_from": proposed,
        "criteria": f"行数で重み付けした年間欠損率 <= {max_null_pct}% かつ "
                    f"年間 {min_races_per_year} レース以上"
                    "（レース数の下限は年間中央値の50%）",
        "by_year": by_year.round(3),
        "note": "これ以前は種牡馬事前分布の推定にのみ使う二段構えを想定。",
        "excluded_from_criteria": list(exclude_cols),
        "market_coverage": market,
    }


# ------------------------------------------------------------------ Q3 名寄せ
def identity_audit(entry: pd.DataFrame, horse_col: str = "horse_sk",
                   date_col: str = "race_date") -> dict:
    """horse_sk / horse_alias の健全性。この品質が過去成績特徴量の精度を直接決める。"""
    e = entry.copy()
    e[date_col] = pd.to_datetime(e[date_col])
    g = e.groupby(horse_col)[date_col]
    career = (g.max() - g.min()).dt.days / 365.25

    same_day = (
        e.groupby([horse_col, e[date_col].dt.date]).size().rename("n")
        .reset_index().query("n > 1")
    )
    gaps = e.sort_values([horse_col, date_col]).groupby(horse_col)[date_col].diff().dt.days.dropna()

    return {
        "n_horses": int(e[horse_col].nunique()),
        "career_years_p50": round(float(career.median()), 2),
        "career_years_p99": round(float(career.quantile(0.99)), 2),
        "n_career_over_12y": int((career > 12).sum()),
        "over_12y_pct": round(100.0 * float((career > 12).mean()), 3),
        "n_same_day_duplicate_starts": int(len(same_day)),
        "gap_days_p50": round(float(gaps.median()), 1) if len(gaps) else None,
        "gap_days_p99": round(float(gaps.quantile(0.99)), 1) if len(gaps) else None,
        # 現役期間が12年を超える個体が多数出たら結合条件が緩すぎる（TR-06）
        "verdict": "過剰結合の疑い" if (career > 12).mean() > 0.01 or len(same_day) else "PASS",
    }


# -------------------------------------------------------------- Q4 ターゲット
def target_basics(entry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """頭数分布、枠番別勝率（場ごと）、人気別勝率。"""
    out: dict[str, pd.DataFrame] = {}
    field_size = entry.groupby("race_id").size().rename("n_runners")
    out["field_size"] = field_size.value_counts().sort_index().rename("n_races").reset_index()

    if "waku" in entry.columns:
        out["waku_winrate"] = (
            entry.groupby(["baba_code", "waku"])["is_win"]
            .agg(["mean", "size"]).rename(columns={"mean": "win_rate", "size": "n"})
            .reset_index().round(4)
        )
    if "popularity" in entry.columns:
        out["popularity_winrate"] = (
            entry.groupby("popularity")["is_win"]
            .agg(["mean", "size"]).rename(columns={"mean": "win_rate", "size": "n"})
            .reset_index().round(4)
        )
    return out


def favourite_longshot_bias(entry: pd.DataFrame, takeout: float = 0.20,
                            n_bins: int = 12, by: str | None = None) -> pd.DataFrame:
    """市場暗黙確率と実測勝率の較正曲線。

    地方でこのバイアスがどの程度あるかは市場効率性の指標そのもの。
    低確率帯で実測 < 暗黙（穴馬が過剰に買われている）なら典型的な longshot bias。
    """
    e = entry.dropna(subset=["odds_win"]).copy()
    raw = (1.0 - takeout) / e["odds_win"]
    e["q_market"] = raw / raw.groupby(e["race_id"]).transform("sum")
    e["bin"] = pd.qcut(e["q_market"], n_bins, duplicates="drop")

    keys = ["bin"] + ([by] if by else [])
    out = (
        e.groupby(keys, observed=True)
        .agg(q_mean=("q_market", "mean"), win_rate=("is_win", "mean"),
             n=("is_win", "size"), odds_mean=("odds_win", "mean"))
        .reset_index()
    )
    out["bias"] = out["win_rate"] - out["q_mean"]
    out["roi_flat"] = out["win_rate"] * out["odds_mean"]
    return out.round(4)


# --------------------------------------------------------------- Q5 分布シフト
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index。

    ゼロ割を避けるため各ビンに小さな下限を置く。0.10 で警告、0.25 で構造変化とみなす。
    """
    exp = pd.to_numeric(expected, errors="coerce").dropna()
    act = pd.to_numeric(actual, errors="coerce").dropna()
    if exp.empty or act.empty or exp.nunique() < 2:
        return np.nan
    edges = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(exp, edges)[0] / len(exp)
    a_pct = np.histogram(act, edges)[0] / len(act)
    e_pct, a_pct = np.clip(e_pct, 1e-6, None), np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def yearly_drift(feat: pd.DataFrame, columns: list[str], date_col: str = "race_date",
                 bins: int = 10, warn: float = 0.10, fail: float = 0.25) -> pd.DataFrame:
    """年ごとの PSI。基準は直前年。構造変化点が walk-forward 分割の設計に直結する。"""
    f = feat.copy()
    f["year"] = pd.to_datetime(f[date_col]).dt.year
    years = sorted(f["year"].unique())
    rows = []
    for prev, cur in zip(years, years[1:]):
        for col in columns:
            if col not in f.columns:
                continue
            v = psi(f.loc[f["year"] == prev, col], f.loc[f["year"] == cur, col], bins)
            rows.append({
                "year": cur, "baseline_year": prev, "column": col, "psi": round(v, 4),
                "status": "FAIL" if v >= fail else ("WARN" if v >= warn else "PASS"),
            })
    return pd.DataFrame(rows)


def change_points(drift: pd.DataFrame, fail: float = 0.25, min_columns: int = 3) -> pd.DataFrame:
    """複数列が同時にシフトした年を構造変化点として拾う。

    1列だけの PSI 上昇はその列固有の事情なので、分割設計を変える理由にはしない。
    """
    hits = drift[drift["psi"] >= fail]
    if hits.empty:
        return pd.DataFrame(columns=["year", "n_columns_shifted", "columns"])
    g = hits.groupby("year")["column"].agg(list).reset_index()
    g["n_columns_shifted"] = g["column"].map(len)
    return (
        g[g["n_columns_shifted"] >= min_columns]
        .rename(columns={"column": "columns"})[["year", "n_columns_shifted", "columns"]]
    )


# ---------------------------------------------------------------- Q6 控除率
def measured_takeout(entry: pd.DataFrame, bet_type: str = "単勝",
                     by: list[str] | None = None) -> pd.DataFrame:
    """券種別・場別の控除率実測。

    公称値ではなく実測値を期待値計算に使う。主催者ごとに設定が違うため。
    """
    keys = by or ["baba_code"]
    rows = []
    for rid, g in entry.dropna(subset=["odds_win"]).groupby("race_id"):
        rec = {"race_id": rid, "inv_sum": inverse_odds_sum(g["odds_win"].to_numpy())}
        for k in keys:
            rec[k] = g[k].iloc[0]
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = df.groupby(keys)["inv_sum"].agg(["mean", "std", "count"]).reset_index()
    out["takeout_measured"] = 1.0 - 1.0 / out["mean"]
    out["takeout_nominal"] = NOMINAL_TAKEOUT[bet_type]
    out["diff_pt"] = ((out["takeout_measured"] - out["takeout_nominal"]) * 100).round(2)
    out["status"] = np.where(out["diff_pt"].abs() <= 2.0, "PASS", "REVIEW")
    return out.round(4)
