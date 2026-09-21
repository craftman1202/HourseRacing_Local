"""設定と秘密情報の解決。

ローカルは `.env`、GCP は Secret Manager から解決する。**未設定時は起動失敗**にし、
無音の送信スキップにはしない（DC-02）。「通知が来ない」を「異常が無い」と
取り違えるのが運用で最も危険な失敗なので、ここは fail-fast にする。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONF_DIR = Path(__file__).resolve().parents[2] / "conf"

# ログ・レスポンスから伏せる値を持つ変数（DC-01 / MO-05）
SECRET_KEYS = ("DISCORD_WEBHOOK_PREDICTION", "DISCORD_WEBHOOK_ALERT",
               "DISCORD_WEBHOOK_DAILY", "DISCORD_WEBHOOK_DEADLETTER")

# webhook URL の形。ID 部分を `\d+` に限定すると、数字以外の ID や
# 別ホストの webhook が素通りする。パスに /webhooks/ を含む URL は
# まとめて伏せる（伏せすぎて困るものではない）。
_WEBHOOK_RE = re.compile(
    r"https?://[\w.\-]+/(?:api/)?webhooks?/[\w\-./]+", re.IGNORECASE)


def load_dotenv(path: str | Path) -> dict[str, str]:
    """.env を読む。値に `=` が含まれても壊れないよう1回だけ分割する。"""
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


log = logging.getLogger(__name__)


class SecretResolver:
    """`.env` → 環境変数 → Secret Manager の順に解決する。"""

    def __init__(self, dotenv_path: str | Path | None = None,
                 secret_manager: Any = None, project: str | None = None) -> None:
        self._dotenv = load_dotenv(dotenv_path) if dotenv_path else {}
        self._sm = secret_manager
        self._project = project
        self._cache: dict[str, str] = {}

    def get(self, key: str, required: bool = True) -> str:
        if key in self._cache:
            return self._cache[key]
        value = self._dotenv.get(key) or os.environ.get(key)
        if not value and self._sm is not None:
            try:
                value = self._sm.access(
                    f"projects/{self._project}/secrets/{key}/versions/latest")
            except Exception as exc:  # noqa: BLE001
                # 権限が無い・秘密が無いは「未設定」と同じ扱い。required=False の
                # 呼び出しまで例外にすると、通知を担当しないサービスが起動できない。
                # required=True なら下で明示的に止める。
                log.warning("%s を Secret Manager から取れません: %s",
                            key, type(exc).__name__)
                value = ""
        if not value:
            if required:
                raise RuntimeError(
                    f"{key} が未設定です。.env か Secret Manager に設定してください。"
                    "未設定のまま黙って送信をスキップする挙動にはしません（DC-02）。")
            return ""
        self._cache[key] = value
        return value

    def secret_values(self) -> list[str]:
        """伏せ字化の対象。実値そのものは決してログに出さない。"""
        return [v for k, v in self._cache.items() if k in SECRET_KEYS]


def redact(text: str, secrets: list[str] | None = None) -> str:
    """秘密情報を伏せる（DC-01 / MO-05）。

    既知の値を消すだけでは不十分（未キャッシュの値が漏れる）なので、
    Discord webhook の URL 形状そのものも正規表現で潰す。
    """
    out = str(text)
    for s in secrets or []:
        if s:
            out = out.replace(s, "***REDACTED***")
    return _WEBHOOK_RE.sub("***REDACTED***", out)


def contains_secret(text: str, secrets: list[str] | None = None) -> bool:
    haystack = str(text)
    if any(s and s in haystack for s in (secrets or [])):
        return True
    return bool(_WEBHOOK_RE.search(haystack))


@dataclass
class OpsConfig:
    project: str
    region: str
    infer_lead_minutes: int
    infer_lead_tolerance_sec: int
    infer_max_retries: int
    infer_retry_backoff_sec: tuple[int, ...]
    discord_min_ev: float
    discord_rate_limit_rps: float
    max_bet_per_race: int
    max_bet_per_day: int
    kelly_fraction: float
    live_refresh_hours: tuple[int, ...]
    odds_snapshot_interval_min: int
    odds_snapshot_close_interval_min: int
    tolerated_skew_columns: frozenset[str]
    model_stale_days: int
    max_bytes_billed: int
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "OpsConfig":
        raw = yaml.safe_load(Path(path or CONF_DIR / "ops.yaml").read_text(encoding="utf-8"))
        inf, disc, bet = raw["inference"], raw["discord"], raw["betting"]
        return cls(
            project=raw["gcp"]["project"], region=raw["gcp"]["region"],
            infer_lead_minutes=int(inf["lead_minutes"]),
            infer_lead_tolerance_sec=int(inf["lead_tolerance_sec"]),
            # 2026-09-21 まで conf/ops.yaml に書いてあるだけで読まれておらず、
            # 推論失敗時のリトライが一度も実装されていなかった
            # （b_body_weight の掲載待ちで無リトライのまま止まっていた実例）。
            infer_max_retries=int(inf.get("max_retries", 3)),
            infer_retry_backoff_sec=tuple(int(s) for s in inf.get(
                "retry_backoff_sec", (30, 90, 270))),
            discord_min_ev=float(disc["min_ev"]),
            discord_rate_limit_rps=float(disc["rate_limit_rps"]),
            max_bet_per_race=int(bet["max_per_race"]),
            max_bet_per_day=int(bet["max_per_day"]),
            kelly_fraction=float(bet["kelly_fraction"]),
            live_refresh_hours=tuple(raw["update"]["live_refresh_hours_jst"]),
            odds_snapshot_interval_min=int(raw["update"]["odds_snapshot_interval_min"]),
            odds_snapshot_close_interval_min=int(
                raw["update"]["odds_snapshot_close_interval_min"]),
            tolerated_skew_columns=frozenset(raw["skew"]["tolerated_columns"]),
            model_stale_days=int(raw["monitoring"]["model_stale_days"]),
            max_bytes_billed=int(raw["gcp"]["max_bytes_billed"]),
            raw=raw,
        )
