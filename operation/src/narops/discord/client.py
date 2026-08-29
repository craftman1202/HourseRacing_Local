"""Discord Webhook 送信。

制約: 1メッセージ 10 embed、全 embed の合計 6,000 文字、Webhook あたり毎分30リクエスト。
3場同時開催では短時間に通知が集中するので、素朴に実装すると 429 を踏む（設計書 §4.2）。

秘密情報（webhook URL）は例外文にも標準出力にも出さない（DC-01）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import httpx
import pandas as pd

from ..clock import Clock, SystemClock, to_utc
from ..config import redact
from ..errors import SecretLeak
from .format import Embed, batch

log = logging.getLogger(__name__)

RATE_WINDOW_SEC = 60.0
RATE_MAX_REQUESTS = 30


@dataclass
class RateLimiter:
    """毎分30リクエストを超えないためのトークンバケット。"""

    rps: float = 0.4
    clock: Clock = field(default_factory=SystemClock)
    sleep: Callable[[float], None] = time.sleep
    _last: float | None = None
    waits: list[float] = field(default_factory=list)

    def acquire(self) -> None:
        interval = 1.0 / self.rps
        now = self.clock.now().timestamp()
        if self._last is not None:
            wait = interval - (now - self._last)
            if wait > 0:
                self.sleep(wait)
                self.waits.append(wait)
                now += wait
        self._last = now


@dataclass
class SendResult:
    sent: int
    chunks: int
    retries: int
    dead_lettered: list[Embed] = field(default_factory=list)


class DiscordSender:
    def __init__(self, webhook: str, *, rate_limiter: RateLimiter | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 max_attempts: int = 3, timeout: float = 10.0) -> None:
        # URL はインスタンス内に閉じ込め、repr にも出さない
        self._webhook = webhook
        self._limiter = rate_limiter or RateLimiter()
        self._sleep = sleep
        self.max_attempts = max_attempts
        self._client = httpx.Client(transport=transport, timeout=timeout)
        self.dead_letter: list[Embed] = []

    def __repr__(self) -> str:          # DC-01: repr から URL を消す
        return "<DiscordSender webhook=***REDACTED***>"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DiscordSender":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send(self, embeds: list[Embed]) -> SendResult:
        """分割 → レート制限 → 429 の Retry-After 尊重（DC-03/04/09）。"""
        chunks = batch(embeds)
        sent = retries = 0
        for chunk in chunks:
            ok, n_retry = self._send_chunk(chunk)
            retries += n_retry
            if ok:
                sent += len(chunk)
            else:
                # 黙って捨てない。ここに落ちるのは Discord の 5xx が続いた場合で、
                # 送れなかったのがアラートだと「停止したこと自体が通知されない」
                # という最悪の状態になる。URL は載せない（伏せても付けない）。
                log.error("Discord へ %d 件を送れませんでした（未送信として保持）: %s",
                          len(chunk), [e.title for e in chunk])
                self.dead_letter.extend(chunk)
        return SendResult(sent, len(chunks), retries, list(self.dead_letter))

    def _send_chunk(self, chunk: list[Embed]) -> tuple[bool, int]:
        payload = {"embeds": [e.to_dict() for e in chunk]}
        retries = 0
        for attempt in range(1, self.max_attempts + 1):
            self._limiter.acquire()
            try:
                r = self._client.post(self._webhook, json=payload)
            except httpx.HTTPError as exc:
                # 例外文に URL が載るので必ず伏せる
                raise RuntimeError(f"Discord 送信に失敗しました: {redact(str(exc))}") from None
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 5)) + 0.5
                self._sleep(wait)
                retries += 1
                continue
            if r.status_code >= 500:
                self._sleep(min(2 ** attempt, 30))
                retries += 1
                continue
            if r.status_code >= 400:
                raise RuntimeError(
                    f"Discord が {r.status_code} を返しました: {redact(r.text)[:200]}")
            return True, retries
        return False, retries


# ------------------------------------------------------------------ DC-07
def dedupe_key(race_id: str, model_release: str, revision: int = 0) -> str:
    """同一レース同一版の再送を防ぐ鍵。

    モデル版が変わったときだけ再送し、その場合は訂正表記を付ける。
    """
    return f"{race_id}:{model_release}:{revision}"


def already_sent(wh, race_id: str, channel: str, key: str) -> bool:
    row = wh.query(
        "SELECT COUNT(*) AS n FROM notification_log "
        "WHERE race_id=? AND channel=? AND dedupe_key=?",
        [race_id, channel, key], allow_full_scan=True)
    return bool(len(row) and int(row["n"].iloc[0]) > 0)


def record_sent(wh, race_id: str, channel: str, key: str, sent_at: datetime,
                start_ts: datetime | None, status: str = "ok") -> None:
    wh.insert_frame("notification_log", pd.DataFrame([{
        "race_id": race_id, "channel": channel, "dedupe_key": key,
        "sent_at": to_utc(sent_at),
        "start_ts": to_utc(start_ts) if start_ts is not None else None,
        "status": status,
    }]))


def assert_no_secret(payload: object, secrets: list[str]) -> None:
    """送信ペイロード・ログに webhook URL が混じっていないか（DC-01）。"""
    from ..config import contains_secret

    if contains_secret(str(payload), secrets):
        raise SecretLeak("秘密情報が出力に含まれています")
