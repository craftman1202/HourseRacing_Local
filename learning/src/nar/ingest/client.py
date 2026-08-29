"""HTTP 取得層。

レート制限・リトライ・Content-Type 検証を1か所に閉じ込める。
待機時間の系列はテストで assert できるよう、外から観測可能にしてある（IG-05/06）。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ..errors import ContentTypeError


class TokenBucket:
    """最低間隔を強制する。sleep 関数を差し替えられるのはテストのため。"""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def acquire(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last = now


@dataclass
class RetryPolicy:
    statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    max_attempts: int = 5
    base_sec: float = 2.0
    factor: float = 2.0
    jitter: float = 0.3

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """attempt は 1 始まり。2s, 4s, 8s, 16s, 32s に ±jitter を掛ける。"""
        base = self.base_sec * (self.factor ** (attempt - 1))
        return base * (1.0 + rng.uniform(-self.jitter, self.jitter))


@dataclass
class FetchResult:
    content: bytes
    status_code: int
    content_type: str
    url: str
    params: dict[str, Any]


@dataclass
class NarClient:
    user_agent: str
    min_interval_sec: float = 3.0
    timeout_sec: float = 60.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    expect_content_type: str = "application/zip"
    seed: int = 0
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._bucket = TokenBucket(self.min_interval_sec, self.clock, self.sleep)
        self.request_times: list[float] = []
        self.waits: list[float] = []
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout_sec,
            transport=self.transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NarClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, url: str, params: dict[str, Any]) -> FetchResult:
        last_status = 0
        for attempt in range(1, self.retry.max_attempts + 2):
            self._bucket.acquire()
            self.request_times.append(self.clock())
            resp = self._client.get(url, params=params)
            last_status = resp.status_code

            if resp.status_code in self.retry.statuses:
                if attempt > self.retry.max_attempts:
                    raise httpx.HTTPStatusError(
                        f"{url} が {resp.status_code} を {attempt} 回返しました",
                        request=resp.request, response=resp,
                    )
                wait = self.retry.backoff(attempt, self._rng)
                self.waits.append(wait)
                self.sleep(wait)
                continue

            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "").split(";")[0].strip()
            if ctype != self.expect_content_type:
                # HTML のエラーページが 200 で返る可能性があるため、
                # ステータスではなく Content-Type を信頼する（IG-07）
                raise ContentTypeError(
                    f"{url} が {ctype!r} を返しました（期待: {self.expect_content_type!r}）。"
                    "raw への保存は行いません。"
                )
            return FetchResult(resp.content, resp.status_code, ctype, url, params)

        raise RuntimeError(f"到達不能: {url} status={last_status}")
