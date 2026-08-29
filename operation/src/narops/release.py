"""リリース運用：公開ゲート、シャドー推論、昇格条件。

承認なしでの本番切り替えは事故の温床なので、ゲートを機械的に強制する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .errors import PromotionRejected, PublishGateFailed
from .model.registry import ModelRegistry

# 学習側仕様書の Blocker 群。これが全 GREEN でないと publish させない（RL-01）
REQUIRED_GREEN = ("IG-14", "LK-05", "LK-06", "EV-03", "RF-01", "RF-02", "RF-03",
                  "RF-07", "RF-08", "CV-07")
SHADOW_DAYS = 14


# ---------------------------------------------------------------------- RL-01
def assert_publish_gate(test_results: dict[str, str]) -> None:
    """学習側の Blocker が全 GREEN でなければ publish 失敗。"""
    missing = [t for t in REQUIRED_GREEN if t not in test_results]
    failed = [t for t in REQUIRED_GREEN if test_results.get(t) not in (None, "GREEN")]
    if missing or failed:
        raise PublishGateFailed(
            "学習側の Blocker が GREEN ではないため publish できません。"
            f"未実施: {missing} / 失敗: {failed}")


# ---------------------------------------------------------------------- RL-02
@dataclass
class ShadowRun:
    """シャドー推論。

    新リリースを current にせず、既存の推論ジョブ内で並列に実行して
    prediction_shadow に書くだけ。**ベット候補も Discord 配信も一切生成しない。**
    """

    release_id: str
    predictions: list[dict] = field(default_factory=list)
    bet_candidates: list[dict] = field(default_factory=list)
    notifications: list[dict] = field(default_factory=list)

    def record(self, race_id: str, horse_no: int, p_win: float) -> None:
        self.predictions.append({"race_id": race_id, "horse_no": horse_no,
                                 "p_win": p_win, "is_shadow": True})

    def emit_bet(self, *args, **kwargs) -> None:
        raise PromotionRejected(
            "シャドー版はベット候補を生成できません（RL-02）。"
            "current に昇格してから配信してください。")

    def notify(self, *args, **kwargs) -> None:
        raise PromotionRejected("シャドー版は Discord 配信を行いません（RL-02）。")

    def comparison_report(self, production: pd.DataFrame) -> pd.DataFrame:
        """本番との予測乖離。切替判断に使う比較レポートのみを出力する。"""
        shadow = pd.DataFrame(self.predictions)
        if shadow.empty or production.empty:
            return pd.DataFrame(columns=["race_id", "horse_no", "p_shadow",
                                         "p_production", "delta"])
        m = shadow.merge(production, on=["race_id", "horse_no"],
                         suffixes=("_shadow", "_production"))
        m["delta"] = m["p_win_shadow"] - m["p_win_production"]
        return m[["race_id", "horse_no", "p_win_shadow", "p_win_production", "delta"]]


# ---------------------------------------------------------------------- RL-03
@dataclass
class ShadowMetrics:
    days: int
    nll: float
    ece: float
    top1: float


def assert_promotion_allowed(shadow: ShadowMetrics, production: ShadowMetrics,
                             min_days: int = SHADOW_DAYS) -> None:
    """2週間分のシャドー指標が現行版以上でなければ昇格を拒否する。"""
    problems = []
    if shadow.days < min_days:
        problems.append(f"シャドー期間 {shadow.days} 日が必要日数 {min_days} 日に足りません")
    if shadow.nll > production.nll:
        problems.append(f"NLL {shadow.nll:.4f} が現行 {production.nll:.4f} より悪い")
    if shadow.ece > production.ece + 1e-12:
        problems.append(f"ECE {shadow.ece:.4f} が現行 {production.ece:.4f} より悪い")
    if problems:
        raise PromotionRejected("昇格条件を満たしません: " + " / ".join(problems))


def promote(registry: ModelRegistry, release_id: str, shadow: ShadowMetrics,
            production: ShadowMetrics, actor: str, confirmed: bool = False) -> None:
    """current の切り替え。二段確認と監査ログを必須にする（WB-06）。"""
    if not confirmed:
        raise PromotionRejected(
            "昇格には二段確認が必要です（confirmed=True）。承認なしの本番切り替えは"
            "事故の温床です。")
    assert_promotion_allowed(shadow, production)
    registry.set_current(release_id, actor=actor,
                         reason=f"promotion nll={shadow.nll:.4f} ece={shadow.ece:.4f}")
