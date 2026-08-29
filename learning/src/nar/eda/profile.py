"""programmatic-eda の手順 1〜5。

1. 構造の把握（粒度の確認） 2. 欠損プロファイル 3. 外れ値 4. 分布 5. 相関。
閾値の既定は skills/programmatic-eda/references/quality_thresholds.md に準拠し、
conf/eda.yaml の overrides だけがプロジェクト固有の判断として記録に残る。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Thresholds:
    null_warn_pct: float = 5.0
    null_fail_pct: float = 30.0
    dup_full_row_fail_pct: float = 1.0
    outlier_z: float = 3.0
    outlier_iqr_k: float = 1.5
    skew_warn: float = 1.0
    skew_fail: float = 2.0
    corr_flag: float = 0.80
    corr_near_perfect: float = 0.95
    per_column: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_conf(cls, conf: dict) -> "Thresholds":
        t = dict(conf.get("thresholds", {}))
        per_col = {
            o["column"]: {k: v for k, v in o.items() if k not in ("column", "reason")}
            for o in conf.get("overrides", [])
        }
        return cls(**t, per_column=per_col)

    def null_limits(self, column: str) -> tuple[float, float]:
        o = self.per_column.get(column, {})
        warn = float(o.get("null_warn_pct", self.null_warn_pct))
        fail = float(o.get("null_fail_pct", self.null_fail_pct))
        if warn > fail:
            # 列別に WARN だけ緩めて FAIL を据え置くと、WARN を飛ばして FAIL に
            # 落ちる区間ができる。設定ミスを黙って通さない。
            raise ValueError(
                f"{column}: null_warn_pct={warn} が null_fail_pct={fail} を超えています。"
                "WARN を緩めるなら FAIL も併せて上げてください。"
            )
        return warn, fail


def overview(df: pd.DataFrame, grain: str) -> dict:
    """手順1。粒度（1行が何を表すか）を明示させるため grain は必須引数にしてある。"""
    return {
        "grain": grain,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "unnamed_columns": [c for c in df.columns if str(c).startswith("Unnamed")],
        "duplicated_column_names": sorted(
            {c for c in df.columns if list(df.columns).count(c) > 1}
        ),
    }


def grain_check(df: pd.DataFrame, key: list[str]) -> dict:
    """想定外の重複キーが無いか。集計・JOIN の前に必ず行う（~/.claude/CLAUDE.md）。"""
    dup = df.duplicated(key, keep=False)
    return {
        "key": key,
        "n_rows": len(df),
        "n_unique_keys": int(df[key].drop_duplicates().shape[0]),
        "n_duplicate_rows": int(dup.sum()),
        "is_unique": bool(not dup.any()),
        "examples": df.loc[dup, key].head(5).to_dict("records"),
    }


# 高い欠損率に構造的な理由がある列。理由を書けるならそれは欠陥ではないので
# チェックリストの FAIL からは外す。空文字は「説明が無い」を意味する。
EXPECTED_NULLS: dict[str, str] = {
    "odds_win": "NAR のオッズ配信は 2026-02 以降のみ。それ以前は原理的に取得できない。"
                "トラックA（オッズ非使用）はこの列に依存しない。",
    "popularity": "オッズと同じ配信制約。単勝人気はオッズの順位なので同じ期間しか無い。",
    "time_sec": "取消・除外・中止で走っていない馬。0 埋めすると「極端に遅い馬」になる。",
    "finish_pos": "同上。走っていない馬に着順は無い。",
    "weight_kg": "計量前の取消と、馬体重を公表しない開催。",
}


def null_profile(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """手順2。欠損は列ごとに割合を出し、パターンを一言で評価する。

    構造的な理由が書ける列（EXPECTED_NULLS）は、割合が高くても FAIL にしない。
    「説明のある欠損」と「説明できない欠損」を混ぜると、後者が埋もれる。
    """
    rows = []
    for c in df.columns:
        n_null = int(df[c].isna().sum())
        pct = 100.0 * n_null / max(len(df), 1)
        warn, fail = th.null_limits(c)
        rows.append({
            "column": c,
            "n_null": n_null,
            "null_pct": round(pct, 3),
            "n_zero": int((df[c] == 0).sum()) if pd.api.types.is_numeric_dtype(df[c]) else 0,
            "warn_pct": warn,
            "fail_pct": fail,
            "status": FAIL if pct >= fail else (WARN if pct >= warn else PASS),
            "expected_reason": EXPECTED_NULLS.get(c, ""),
        })
    out = pd.DataFrame(rows)
    explained = out["expected_reason"].astype(str).str.strip() != ""
    out.loc[explained & (out["status"] == FAIL), "status"] = WARN
    return out.sort_values("null_pct", ascending=False)


def missingness_pattern(df: pd.DataFrame, by: str | None = None, top_n: int = 8) -> pd.DataFrame:
    """欠損が MCAR/MAR/MNAR のどれらしいかを見るための、他列条件付き欠損率。

    by（例: race_date の年）で欠損率が動くなら MCAR ではない。動かない保証は
    与えられないが、「暗黙に drop してよいか」の判断材料にはなる。
    """
    cols = df.columns[df.isna().any()][:top_n]
    if by is None or not len(cols):
        return pd.DataFrame()
    return (
        df.assign(**{by: df[by]})
        .groupby(by)[list(cols)]
        .apply(lambda g: g.isna().mean() * 100)
        .round(2)
        .reset_index()
    )


def outliers(df: pd.DataFrame, th: Thresholds, columns: list[str] | None = None) -> pd.DataFrame:
    """手順3。IQR と z-score の両方。除去はせず分類だけを求める。"""
    cols = columns or df.select_dtypes("number").columns.tolist()
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty or s.nunique() < 3:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - th.outlier_iqr_k * iqr, q3 + th.outlier_iqr_k * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        sd = s.std(ddof=0)
        n_z = int((np.abs(s - s.mean()) > th.outlier_z * sd).sum()) if sd > 0 else 0
        rows.append({
            "column": c, "n_iqr_outliers": n_iqr, "iqr_pct": round(100 * n_iqr / len(s), 3),
            "n_zscore_outliers": n_z, "iqr_low": lo, "iqr_high": hi,
            "min": s.min(), "max": s.max(),
            "classification": "未分類",  # real signal / data error / structural を人手で埋める
        })
    return pd.DataFrame(rows)


def distributions(df: pd.DataFrame, th: Thresholds, columns: list[str] | None = None) -> pd.DataFrame:
    """手順4。歪度は変換の要否判断に直結するので必ず出す。"""
    cols = columns or df.select_dtypes("number").columns.tolist()
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        skew = float(s.skew()) if s.nunique() > 2 else 0.0
        rows.append({
            "column": c, "n": len(s), "mean": s.mean(), "median": s.median(),
            "std": s.std(), "p5": s.quantile(0.05), "p95": s.quantile(0.95),
            "skew": round(skew, 3),
            "skew_status": FAIL if abs(skew) > th.skew_fail else (
                WARN if abs(skew) > th.skew_warn else PASS),
            "suggested_transform": _transform_hint(s, skew, th),
        })
    return pd.DataFrame(rows)


def _transform_hint(s: pd.Series, skew: float, th: Thresholds) -> str:
    if abs(skew) <= th.skew_warn:
        return "なし"
    if (s > 0).all():
        return "log"
    return "yeo-johnson"


# 定義上そうなる相関。説明を持たない強相関だけがチェックリストの FAIL 対象になる。
# ここに書くのは「なぜそうなるか」であって「無視してよい」ではない。
KNOWN_CORRELATIONS: dict[frozenset[str], str] = {
    frozenset({"horse_no", "waku"}):
        "枠番は馬番を頭数に応じて束ねた値なので定義上ほぼ線形。"
        "特徴量には相対枠順（draw_rel）だけを使い、両方は入れない。",
    frozenset({"finish_pos", "is_win"}):
        "is_win は finish_pos == 1 の指示変数。どちらも目的変数側で、"
        "特徴量には入らない。",
    frozenset({"odds_win", "popularity"}):
        "人気はオッズの順位。市場情報として同じものを別表現で持っている。",
    frozenset({"time_sec", "distance"}):
        "距離が伸びれば走破時計は伸びる。速度指数は場×距離で標準化して切り離す。",
}


def explain_correlation(a: str, b: str) -> str:
    return KNOWN_CORRELATIONS.get(frozenset({a, b}), "")


def correlations(df: pd.DataFrame, th: Thresholds, method: str = "spearman") -> pd.DataFrame:
    """手順5。

    既定を spearman にしているのは、競馬の特徴量（賞金、オッズ、出走間隔）が
    強く歪んでおり pearson だと外れ値1件で相関が立つため。
    """
    num = df.select_dtypes("number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    corr = num.corr(method=method)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.isna(r) or abs(r) < th.corr_flag:
                continue
            pairs.append({
                "col_a": cols[i], "col_b": cols[j], "r": round(float(r), 4),
                "flag": "near_perfect" if abs(r) >= th.corr_near_perfect else "strong",
                "action": "片方を落とす" if abs(r) >= th.corr_near_perfect else "VIF を確認",
                "explanation": explain_correlation(cols[i], cols[j]),
            })
    return pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
