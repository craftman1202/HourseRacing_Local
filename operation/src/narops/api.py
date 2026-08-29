"""FastAPI: 運用エンドポイントと Web 向け API。

期待値計算・Kelly・自己インパクト補正は Python 側の単一実装を共有する。
TypeScript で再実装すると必ず乖離し、Web に表示される数値と Discord に届く数値が
違う、という最悪のバグを生む（設計書 §5.1）。OpenAPI から TS 型を生成すれば
型安全は保ったまま単一実装を維持できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ------------------------------------------------------------------ 契約（WB-01）


class HorsePrediction(BaseModel):
    horse_no: int = Field(ge=1, le=20)
    horse_name: str | None = None
    p_win: float = Field(ge=0.0, le=1.0)
    p_market: float | None = Field(default=None, ge=0.0, le=1.0)
    ev: float | None = None
    ev_adjusted: float | None = None
    stake_yen: int = Field(ge=0)


class RacePrediction(BaseModel):
    race_id: str = Field(pattern=r"^\d{12}$")
    race_date: date
    start_ts: datetime
    track_name: str
    race_no: int = Field(ge=1, le=12)
    model_release: str
    track_used: Literal["A", "A+B"]
    status: Literal["ok", "insufficient_data", "expired", "not_computed"]
    horses: list[HorsePrediction] = Field(default_factory=list)
    is_paper: bool = True

    @field_validator("horses")
    @classmethod
    def probabilities_sum_to_one(cls, v: list[HorsePrediction]) -> list[HorsePrediction]:
        """API 境界でも総和1を検証する（IN-01 を Web にも延長）。"""
        if v:
            total = sum(h.p_win for h in v)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"レース内確率の総和が {total} です（1 であるべき）")
        return v


class OOSMetrics(BaseModel):
    """保存済み評価テーブル由来。オンザフライ再計算はしない（WB-03）。"""

    model: str
    track: Literal["A", "B"]
    race_nll: float
    top1: float
    top3: float | None = None
    brier: float | None = None
    ece: float
    evaluated_at: datetime
    source: Literal["stored"] = "stored"


class DailyPnL(BaseModel):
    business_date: date
    n_races: int
    n_bets: int
    stake_yen: int
    return_yen: int
    roi: float
    hit_rate: float
    is_paper: bool          # ペーパー／実運用の区分を必ず明示する（WB-04）


class SystemHealth(BaseModel):
    db_watermark: datetime | None
    model_release: str | None
    model_trained_on: date | None
    skew_verdict: Literal["PASS", "FAIL", "UNKNOWN"]
    coverage_today: float
    delivery_blocked: bool
    blocked_reason: str | None = None


class TodayRace(BaseModel):
    """ダッシュボード `/` の本日開催タイムライン向け（設計書 §5.2(1)）。"""

    race_id: str = Field(pattern=r"^\d{12}$")
    race_no: int = Field(ge=1, le=12)
    track_name: str
    start_ts: datetime
    status: Literal["ok", "insufficient_data", "expired", "not_computed"]
    top_ev_adjusted: float | None = None


class ModelRelease(BaseModel):
    """`/models` 画面向け。gs://nar-model/releases/<id>/manifest.json を要約する。"""

    release_id: str
    is_current: bool
    model_id: str
    dataset_version: str
    train_period_start: date
    train_period_end: date
    git_commit: str
    track: Literal["A", "B"]
    purpose: str
    oos_metrics: dict[str, float]
    created_at: datetime


class PromoteRequest(BaseModel):
    confirmed: bool = False


# ------------------------------------------------------------------ 認可
Plan = Literal["free", "paid", "admin"]
# 有料機能は API レベルで止める。UI を隠すだけでは AU-01 を満たさない
PAID_FEATURES = frozenset({"today_prediction_detail", "betslip_text"})
ADMIN_FEATURES = frozenset({"model_promote"})


@dataclass
class Principal:
    user_id: str
    plan: Plan = "free"

    def can(self, feature: str) -> bool:
        if feature in ADMIN_FEATURES:
            return self.plan == "admin"
        if feature in PAID_FEATURES:
            return self.plan in ("paid", "admin")
        return True


class Forbidden(Exception):
    pass


class NotFound(Exception):
    pass


def require_feature(principal: Principal, feature: str) -> None:
    """AU-01: 無料プランで有料機能を API レベルでも取得不可にする。"""
    if not principal.can(feature):
        raise Forbidden(f"{feature} は {principal.plan} プランでは利用できません")


def require_owner(principal: Principal, resource_owner: str) -> None:
    """WB-07: 他ユーザのリソース ID を直接指定しても弾く（IDOR 対策）。"""
    if principal.plan != "admin" and principal.user_id != resource_owner:
        raise NotFound("リソースが見つかりません")


# ------------------------------------------------------------------ 課金 webhook
@dataclass
class BillingProcessor:
    """AU-02: 署名検証・重複配信の冪等処理・失効時の即時ダウングレード。"""

    secret: str
    processed: set[str] = field(default_factory=set)
    plans: dict[str, Plan] = field(default_factory=dict)

    def verify(self, payload: bytes, signature: str) -> bool:
        import hashlib
        import hmac

        expected = hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def handle(self, event_id: str, user_id: str, event_type: str,
               payload: bytes, signature: str) -> str:
        if not self.verify(payload, signature):
            raise Forbidden("課金 webhook の署名が不正です")
        if event_id in self.processed:
            return "duplicate"          # 冪等: 同一イベントの再配信は何もしない
        self.processed.add(event_id)
        if event_type in ("subscription.created", "subscription.renewed"):
            self.plans[user_id] = "paid"
            return "upgraded"
        if event_type in ("subscription.canceled", "payment.failed"):
            self.plans[user_id] = "free"    # 失効時は即時ダウングレード
            return "downgraded"
        return "ignored"

    def plan_of(self, user_id: str) -> Plan:
        return self.plans.get(user_id, "free")


# ------------------------------------------------------------------ 締切表示
def betting_ui_enabled(start_ts: datetime, now: datetime) -> bool:
    """WB-05: 発走後はベット提案を「参考」表示にし、投票補助 UI を無効化する。"""
    from .clock import to_utc

    return to_utc(now) < to_utc(start_ts)


def build_app(services: dict[str, Any] | None = None):
    """FastAPI アプリを組み立てる。

    services に注入することでテストから差し替え可能にする。ここでは契約
    （スキーマと認可）を確定させることが目的で、実処理は各モジュールが持つ。
    """
    from fastapi import Depends, FastAPI, HTTPException

    from .discord.betslip import Slip, render
    from .errors import PromotionRejected

    svc = services or {}
    app = FastAPI(title="nar-api", version="1.0.0")

    def principal() -> Principal:
        return svc.get("principal", Principal("anonymous", "free"))

    @app.get("/health", response_model=SystemHealth)
    def health() -> SystemHealth:
        return svc["health"]()

    @app.get("/races/today", response_model=list[TodayRace])
    def races_today() -> list[TodayRace]:
        return svc["races_today"]()

    @app.get("/races/{race_id}", response_model=RacePrediction)
    def race(race_id: str, who: Principal = Depends(principal)) -> RacePrediction:
        try:
            require_feature(who, "today_prediction_detail")
        except Forbidden as e:
            raise HTTPException(status_code=403, detail=str(e)) from None
        try:
            return svc["race"](race_id)
        except NotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from None

    @app.get("/performance/oos", response_model=list[OOSMetrics])
    def oos() -> list[OOSMetrics]:
        return svc["oos"]()

    @app.get("/performance/live", response_model=list[DailyPnL])
    def live() -> list[DailyPnL]:
        return svc["live"]()

    @app.get("/races/{race_id}/betslip")
    def betslip_text(race_id: str, site: Literal["rakuten", "spat4"] = "rakuten",
                      who: Principal = Depends(principal)) -> dict:
        """WB-08 相当: 楽天競馬/SPAT4 形式のコピー用テキストを返す。

        金額・馬番は `race` callable が返す確定済みの推奨（stake_yen）から作る。
        フォーマットは discord.betslip の単一実装をそのまま呼び、TS 側では
        再実装しない（設計書 §5.1 の要件）。
        """
        try:
            require_feature(who, "betslip_text")
        except Forbidden as e:
            raise HTTPException(status_code=403, detail=str(e)) from None
        try:
            pred = svc["race"](race_id)
        except NotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from None
        slips = [Slip(race_id=race_id, bet_type="単勝", horse_no=h.horse_no,
                      stake_yen=h.stake_yen)
                 for h in pred.horses if h.stake_yen > 0]
        return {"site": site, "text": render(slips, site)}

    @app.get("/models", response_model=list[ModelRelease])
    def models() -> list[ModelRelease]:
        return svc["models"]()

    @app.post("/models/{release_id}/promote")
    def promote_model(release_id: str, body: PromoteRequest,
                      who: Principal = Depends(principal)) -> dict:
        """WB-06: 昇格は admin のみ・二段確認必須。判定は release.py の既存ゲートを使う。"""
        try:
            require_feature(who, "model_promote")
        except Forbidden as e:
            raise HTTPException(status_code=403, detail=str(e)) from None
        try:
            svc["promote"](release_id, actor=who.user_id, confirmed=body.confirmed)
        except PromotionRejected as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        return {"status": "promoted", "release_id": release_id}

    return app
