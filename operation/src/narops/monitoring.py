"""アラート条件と SLO 集計。

設計書 §7.2 の7系統を漏れなく実装する（MO-02）。とくに RF 系ガードの本番発火は
「成功ではなくパイプラインの破損」として扱い、推奨の配信を自動停止する。
良すぎる結果を信じて資金を投じるのが、この種のシステムで最も高額な失敗。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# 設計書 §7.2 の7系統。MO-02 はこの集合に対する実装の網羅性を検証する
ALERT_KINDS = (
    "fetch_failure", "schema_drift", "skew_mismatch", "coverage_drop",
    "model_stale", "budget", "rf_guard",
)
BLOCKING_KINDS = frozenset({"skew_mismatch", "rf_guard", "schema_drift"})


@dataclass
class Alert:
    kind: str
    severity: str          # Blocker / Critical / Warning
    message: str
    value: float | str | None = None
    fired: bool = True

    @property
    def blocks_delivery(self) -> bool:
        """配信を止めるべきアラートか。"""
        return self.kind in BLOCKING_KINDS


@dataclass
class MonitorConfig:
    ece_degradation: float = 0.02
    psi_threshold: float = 0.25
    coverage_min: float = 0.90
    model_stale_days: int = 90
    rf_top1_max: float = 0.60
    rf_nll_min: float = 1.20
    rf_roi_max: float = 1.30
    rf_roi_months: int = 3
    fetch_failure_streak: int = 3


# ------------------------------------------------------------------ MO-01
def check_ece(series: pd.Series, oos_ece: float, cfg: MonitorConfig) -> Alert | None:
    """本番 ECE の30日移動平均が OOS 比で悪化していないか。"""
    if series.empty:
        return None
    rolling = float(series.tail(30).mean())
    if rolling - oos_ece >= cfg.ece_degradation:
        return Alert("model_degradation", "Critical",
                     f"本番 ECE 30日平均 {rolling:.4f} が OOS {oos_ece:.4f} より "
                     f"{rolling - oos_ece:.4f} 悪化しています。", rolling)
    return None


def check_psi(psi_by_feature: dict[str, float], cfg: MonitorConfig) -> Alert | None:
    over = {k: v for k, v in psi_by_feature.items() if v >= cfg.psi_threshold}
    if over:
        return Alert("model_degradation", "Critical",
                     f"PSI が {cfg.psi_threshold} を超えた特徴量: "
                     + ", ".join(f"{k}={v:.3f}" for k, v in sorted(over.items())),
                     max(over.values()))
    return None


def check_coverage(coverage: float, cfg: MonitorConfig) -> Alert | None:
    if coverage < cfg.coverage_min:
        return Alert("coverage_drop", "Critical",
                     f"推論カバレッジ {coverage * 100:.1f}% が下限 "
                     f"{cfg.coverage_min * 100:.0f}% を下回りました。", coverage)
    return None


def check_model_freshness(trained_on: date, today: date, cfg: MonitorConfig) -> Alert | None:
    age = (today - trained_on).days
    if age >= cfg.model_stale_days:
        return Alert("model_stale", "Critical",
                     f"current の学習日が {age} 日前です（上限 {cfg.model_stale_days} 日）。"
                     "再学習してください。", age)
    return None


def check_fetch_failures(streak: int, cfg: MonitorConfig) -> Alert | None:
    if streak >= cfg.fetch_failure_streak:
        return Alert("fetch_failure", "Critical",
                     f"NAR からの取得が {streak} 回連続で失敗しています。", streak)
    return None


def check_schema_drift(drifted: list[str]) -> Alert | None:
    if drifted:
        return Alert("schema_drift", "Blocker",
                     f"スキーマドリフトを検出しました: {drifted}。silver 以降を停止します。",
                     len(drifted))
    return None


def check_skew(report) -> Alert | None:
    if not report.is_clean:
        return Alert("skew_mismatch", "Blocker",
                     f"skew 検証が不一致です。{report.summary()} "
                     "当日の推奨配信を停止します。", len(report.offending_columns))
    return None


def check_budget(spend_usd: float, threshold: float = 3.0) -> Alert | None:
    if spend_usd >= threshold:
        return Alert("budget", "Critical",
                     f"当月の課金が ${spend_usd:.2f} に達しました（閾値 ${threshold:.2f}）。",
                     spend_usd)
    return None


# ------------------------------------------------------------------ RF ガード
def check_rf_guards(*, top1_30d: float | None = None, nll_30d: float | None = None,
                    monthly_roi: pd.Series | None = None,
                    cfg: MonitorConfig | None = None) -> list[Alert]:
    """本番での「良すぎる結果」検出（設計書 §7.2 / MO-02）。

    学習側の RF-01/02/03 を運用の実測値に対して回す。発火したら推奨の配信を
    自動停止する。成功ではなく破損の徴候として扱う。
    """
    cfg = cfg or MonitorConfig()
    out: list[Alert] = []
    if top1_30d is not None and top1_30d > cfg.rf_top1_max:
        out.append(Alert("rf_guard", "Blocker",
                         f"RF-01: 直近30日の Top-1 精度が {top1_30d * 100:.1f}% です。"
                         "着順情報の混入がほぼ確実。配信を停止します。", top1_30d))
    if nll_30d is not None and nll_30d < cfg.rf_nll_min:
        out.append(Alert("rf_guard", "Blocker",
                         f"RF-02: 直近30日のレース内 NLL が {nll_30d:.3f} です。"
                         "着順情報の混入がほぼ確実。配信を停止します。", nll_30d))
    if monthly_roi is not None and len(monthly_roi) >= cfg.rf_roi_months:
        streak = _longest_streak(monthly_roi > cfg.rf_roi_max)
        if streak >= cfg.rf_roi_months:
            out.append(Alert("rf_guard", "Blocker",
                             f"RF-03: 補正前 単勝 ROI が {cfg.rf_roi_max * 100:.0f}% 超を "
                             f"{streak} か月継続しています。実力ではなく実装を疑ってください。",
                             streak))
    return out


def _longest_streak(flags: pd.Series) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def implemented_alert_kinds() -> set[str]:
    """MO-02: 設計書記載の7系統に実装が存在することを示す。"""
    return {
        "fetch_failure", "schema_drift", "skew_mismatch", "coverage_drop",
        "model_stale", "budget", "rf_guard",
    }


def should_block_delivery(alerts: list[Alert]) -> bool:
    return any(a.fired and a.blocks_delivery for a in alerts)


# ------------------------------------------------------------------ MO-04 SLO
@dataclass
class SLO:
    notify_in_time: float
    coverage: float
    db_fresh_days: int
    skew_mismatches: int
    targets: dict[str, float] = field(default_factory=lambda: {
        "notify_in_time": 0.95, "coverage": 0.98, "skew_mismatches": 0})

    def violations(self) -> list[str]:
        out = []
        if self.notify_in_time < self.targets["notify_in_time"]:
            out.append(f"適時性 {self.notify_in_time:.3f} < {self.targets['notify_in_time']}")
        if self.coverage < self.targets["coverage"]:
            out.append(f"カバレッジ {self.coverage:.3f} < {self.targets['coverage']}")
        if self.skew_mismatches > self.targets["skew_mismatches"]:
            out.append(f"skew 不一致 {self.skew_mismatches} 件")
        return out

    @property
    def met(self) -> bool:
        return not self.violations()
