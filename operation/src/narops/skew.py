"""train-serving skew の常時監視。

学習時と推論時の特徴量が同一手続きで計算されていることを、コードの共有だけでなく
**数値で**検証する。設計書のリーク対策を運用時に延長する仕組みであり、
静かな劣化を防ぐ唯一の手段（設計書 §2.4）。

日次 03:30 に、前日の全レースについて特徴量を「確定層のみから再計算」し、
推論時に保存した feature_snapshot と比較する。不一致が許容されるのは、
当日ライブ層に依存する3系統だけで、それ以外の乖離はバグ（SK-01 / SK-03）。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .errors import SkewError
from .shared import learning_package_root

TOLERANCE = 1e-9


@dataclass
class SkewReport:
    business_date: date
    n_compared: int
    mismatches: pd.DataFrame
    tolerated_columns: frozenset[str]

    @property
    def offending_columns(self) -> list[str]:
        if self.mismatches.empty:
            return []
        cols = set(self.mismatches["column"]) - set(self.tolerated_columns)
        return sorted(cols)

    @property
    def is_clean(self) -> bool:
        return not self.offending_columns

    @property
    def verdict(self) -> str:
        return "PASS" if self.is_clean else "FAIL"

    def summary(self) -> str:
        if self.is_clean:
            return f"{self.business_date}: {self.n_compared} 行を比較、許容外の乖離なし"
        return (f"{self.business_date}: 許容外の乖離が {len(self.mismatches)} 件 "
                f"（列 {self.offending_columns}）")


def compare(
    snapshot: pd.DataFrame,
    recomputed: pd.DataFrame,
    tolerated: frozenset[str],
    business_day: date,
    key: tuple[str, ...] = ("race_id", "horse_no"),
) -> SkewReport:
    """スナップショットと再計算値を突き合わせる（SK-01）。

    比較対象は両者に共通する数値列のみ。片方にしか無い列は「列構成が違う」
    という別の異常なので、mismatch として明示的に記録する。
    """
    k = list(key)
    merged = snapshot.merge(recomputed, on=k, suffixes=("_snap", "_recalc"), how="inner")

    snap_cols = _numeric_feature_columns(snapshot, k)
    recalc_cols = _numeric_feature_columns(recomputed, k)
    only_snap = sorted(snap_cols - recalc_cols)
    only_recalc = sorted(recalc_cols - snap_cols)

    rows: list[dict] = []
    for col in sorted(snap_cols & recalc_cols):
        a = pd.to_numeric(merged[f"{col}_snap"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_recalc"], errors="coerce")
        both_na = a.isna() & b.isna()
        # 相対誤差で判定する。値のスケールが列ごとに大きく違うため
        denom = np.maximum(np.abs(a.fillna(0.0)), 1.0)
        differs = (~both_na) & ((a.isna() ^ b.isna()) |
                                ((a.fillna(0.0) - b.fillna(0.0)).abs() / denom > TOLERANCE))
        for _, r in merged[differs].iterrows():
            rows.append({
                "race_id": r["race_id"], "horse_no": int(r["horse_no"]), "column": col,
                "snapshot": r[f"{col}_snap"], "recomputed": r[f"{col}_recalc"],
            })
    for col in only_snap + only_recalc:
        rows.append({"race_id": "-", "horse_no": -1, "column": col,
                     "snapshot": "存在" if col in snap_cols else "欠落",
                     "recomputed": "存在" if col in recalc_cols else "欠落"})

    return SkewReport(business_day, len(merged), pd.DataFrame(rows), tolerated)


def _numeric_feature_columns(df: pd.DataFrame, key: list[str]) -> set[str]:
    meta = set(key) | {"as_of_date", "computed_at", "as_of_ts", "db_watermark",
                       "model_release", "feature_spec_hash", "race_date", "start_ts",
                       "horse_sk", "jockey_sk", "trainer_sk", "sire_sk", "baba_code",
                       "finish_pos", "is_win"}
    return {c for c in df.columns
            if c not in meta and pd.api.types.is_numeric_dtype(df[c])}


def enforce(report: SkewReport) -> None:
    """許容外の乖離があれば止める（SK-03）。

    Blocker。ホワイトリスト列以外の差分は、その日の推奨配信を停止する理由になる。
    """
    if not report.is_clean:
        detail = report.mismatches[
            ~report.mismatches["column"].isin(report.tolerated_columns)].head(10)
        raise SkewError(
            f"学習時と推論時の特徴量が一致しません。{report.summary()}\n{detail.to_string(index=False)}")


# ---------------------------------------------------------------------- SK-02
def shared_code_paths() -> dict[str, str]:
    """推論が呼んでいる関数の定義位置。

    運用側に重複実装があると、ここが narops 配下を指す。学習側パッケージの
    実体を指していることをテストで固定する。
    """
    from . import shared

    targets = {
        "to_prerace": shared.to_prerace,
        "race_softmax": shared.race_softmax,
        "normalize_within_race": shared.normalize_within_race,
        "effective_odds": shared.effective_odds,
        "kelly_fraction": shared.kelly_fraction,
        "shrink": shared.shrink,
    }
    # 相対パス混じりの sys.path（`../learning/src` など）で読み込むと
    # `/operation/../learning/...` という形になり、前方一致の判定が誤って落ちる。
    # 起動方法で結果が変わってはいけないので、正規化してから返す。
    return {name: str(Path(inspect.getsourcefile(fn) or "").resolve())
            for name, fn in targets.items()}


def assert_no_duplicate_implementation() -> None:
    """共有すべき関数が学習側パッケージ由来であることを検証する（SK-02）。"""
    root = str(Path(learning_package_root()).resolve())
    offenders = {n: p for n, p in shared_code_paths().items() if not p.startswith(root)}
    if offenders:
        raise SkewError(
            "学習側と共有すべき関数が運用側で再実装されています: "
            f"{offenders}。同一実装を共有しないと Web と Discord の数値が乖離します。")
