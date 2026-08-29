"""定期ジョブの実行予算と記録。

更新頻度は「必要最低限に絞る」という要件があるので、超過実行を no-op にする
仕組みをコード側に持つ（DB-06）。Scheduler の設定ミスや手動再実行で
無駄な起動が積み上がるのを防ぐ。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from .clock import Clock, business_date
from .config import OpsConfig

# 1営業日あたりの許容実行回数（設計書 §2.2 の更新頻度表）
DAILY_LIMITS = {
    "final_merge": 1,        # 月次ファイル取り込み。NAR 側の更新が 02:00 頃
    "live_refresh": 3,       # 12:00 / 16:00 / 20:00
    "skew_check": 1,
    "plan_day": 1,
    "weekly_report": 1,
}


@dataclass
class UpdateBudget:
    """1日あたりの実行回数の上限。

    件数は `job_run` から数える。プロセス内のカウンタだけで持つと、
    Cloud Run のように**インスタンスが入れ替わる**環境では上限が効かない
    （新しいインスタンスは 0 から数え直す）。逆に同じインスタンスに当たれば
    効く、という再現しない挙動になる。

    wh を渡さない場合はプロセス内カウンタで動く（テスト用）。
    """

    cfg: OpsConfig
    limits: dict[str, int] = field(default_factory=lambda: dict(DAILY_LIMITS))
    counts: dict[tuple[str, date], int] = field(default_factory=dict)
    wh: object | None = None

    def allow(self, job: str, day: date) -> bool:
        """実行してよいか。超過なら False を返し、呼び出し側は no-op にする。"""
        limit = self.limits.get(job)
        if limit is None:
            return True
        if self.used(job, day) >= limit:
            return False
        self.counts[(job, day)] = self.counts.get((job, day), 0) + 1
        return True

    def used(self, job: str, day: date) -> int:
        local = self.counts.get((job, day), 0)
        if self.wh is None:
            return local
        return max(local, runs_today(self.wh, job, day))


def record_run(wh, job_name: str, clock: Clock, status: str = "ok",
               detail: str = "") -> None:
    now = clock.now()
    wh.insert_frame("job_run", pd.DataFrame([{
        "job_name": job_name, "run_at": now, "business_date": business_date(now),
        "status": status, "detail": detail,
    }]))


def runs_today(wh, job_name: str, day: date) -> int:
    row = wh.query(
        "SELECT COUNT(*) AS n FROM job_run WHERE job_name = ? AND business_date = ? "
        "AND status = 'ok'", [job_name, day], allow_full_scan=True)
    return int(row["n"].iloc[0]) if len(row) else 0


@dataclass
class RetryPolicy:
    """IN-11: 一時障害はリトライ、アサーション違反・データ欠損はリトライしない。

    同じ結果になるものを3回試すのは、失敗を3倍遅くするだけで何も得られない。
    """

    backoff_sec: tuple[int, ...] = (30, 90, 270)
    max_retries: int = 3

    RETRYABLE = ("BigQueryTransient", "HTTPError", "TimeoutError", "ServiceUnavailable")
    NON_RETRYABLE = ("AsOfViolation", "ZeroFillForbidden", "NormalizationError",
                     "InsufficientData", "FeatureSpecMismatch", "SkewError",
                     "StaleDataError", "RaceExpired")

    def should_retry(self, exc: BaseException, attempt: int) -> bool:
        name = type(exc).__name__
        if name in self.NON_RETRYABLE:
            return False
        if attempt >= self.max_retries:
            return False
        return name in self.RETRYABLE or isinstance(exc, (TimeoutError, ConnectionError))

    def wait_for(self, attempt: int) -> int:
        idx = min(attempt, len(self.backoff_sec)) - 1
        return self.backoff_sec[max(idx, 0)]
