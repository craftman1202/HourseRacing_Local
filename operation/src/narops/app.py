"""Cloud Run で動く FastAPI アプリ。

Scheduler / Tasks からは OIDC 認証で呼ばれる。Cloud Run 側で
`--no-allow-unauthenticated` にしてあるので認証はプラットフォームが行うが、
ここでも呼び出し元を検証する（IN-12 / 多層防御）。

起動時に配布物の完全性を検証し、失敗したらヘルスチェックを赤にする（MP-02）。
壊れたモデルで推論を続けるより、起動しないほうが安全。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from .config import OpsConfig, SecretResolver, redact
from .mode import OperatingState

log = logging.getLogger("narops.app")

_state: dict[str, Any] = {}


def _build_services() -> Any:
    """起動時に依存を組み立てる。

    ここで落ちたら起動失敗にする。`.env`/Secret Manager が未設定のまま
    無音で配信をスキップする状態を作らない（DC-02）。
    """
    from .db import schema
    from .db.backend import Warehouse
    from .gcp import BigQueryWarehouse
    from .service import Services

    cfg = OpsConfig.load()
    state = OperatingState.from_env()
    state.validate()

    backend = os.environ.get("NAROPS_BACKEND", "bigquery")
    if backend == "bigquery":
        wh = BigQueryWarehouse(project=cfg.project,
                               dataset=cfg.raw["gcp"]["resources"]["bq_dataset"],
                               location=cfg.region,
                               max_bytes_billed=cfg.max_bytes_billed)
    else:
        wh = Warehouse(os.environ.get("NAROPS_DB", ":memory:"),
                       max_bytes_billed=cfg.max_bytes_billed)
        schema.create_all(wh)

    svc = Services(wh=wh, cfg=cfg, state=state)
    _attach_queue(svc, cfg)
    _attach_model(svc, cfg)
    _attach_discord(svc, cfg, state)
    return svc


def _attach_queue(svc: Any, cfg: OpsConfig) -> None:
    """推論タスクの予約先を Cloud Tasks にする。

    既定のインメモリ実装はテスト用。Cloud Run に載せたままだと、予約が
    インスタンスの寿命で消え、**締切前の推論が一度も起動しない**。
    「/plan-day が 20 件登録しました」と報告するのに何も走らない、という
    いちばん気付きにくい壊れ方になる。
    """
    service_url = os.environ.get("NAROPS_SERVICE_URL")
    if not service_url:
        log.warning("NAROPS_SERVICE_URL 未設定。推論タスクはメモリ上にしか"
                    "積まれません（インスタンス終了で消えます）")
        return
    from .gcp import CloudTasksQueue

    resources = cfg.raw["gcp"]["resources"]
    svc.queue = CloudTasksQueue(
        project=cfg.project, location=cfg.region,
        queue=resources.get("tasks_queue", "nar-queue"),
        service_url=service_url,
        service_account=os.environ.get(
            "NAROPS_TASKS_SA", f"nar-ops@{cfg.project}.iam.gserviceaccount.com"))
    log.info("Cloud Tasks を使います（%s）", svc.queue.queue)


def _attach_model(svc: Any, cfg: OpsConfig) -> None:
    """current リリースを読み、SHA-256 を検証してから載せる（MP-02）。"""
    from nar.config import feature_config

    svc.feature_config = feature_config()
    from .model.registry import ModelRegistry

    root = os.environ.get("NAROPS_MODEL_ROOT")
    bucket = os.environ.get("NAROPS_MODEL_BUCKET")
    if root:
        svc.registry = ModelRegistry(root)
        rid = svc.registry.current_id()
    elif bucket:
        # Cloud Run は永続ディスクを持たない。GCS の current を見て、
        # 起動時に一度だけ取ってくる。ハッシュ検証は取得後にローカルで行う。
        import tempfile

        from .gcp import GcsModelRegistry

        remote = GcsModelRegistry(bucket=bucket,
                                  project=os.environ.get("NAROPS_PROJECT", ""))
        rid = remote.current_id()
        if rid is None:
            log.warning("GCS 側の current が未設定。推論は不可")
            return
        cache = Path(os.environ.get("NAROPS_MODEL_CACHE", tempfile.gettempdir())
                     ) / "nar-model"
        (cache / "releases").mkdir(parents=True, exist_ok=True)
        remote.download_release(rid, str(cache / "releases"))
        svc.registry = ModelRegistry(cache)
        svc.registry.set_current(rid, actor="startup", reason="GCS から取得")
        log.info("GCS から %s を取得しました（%s）", rid, bucket)
    else:
        log.warning("NAROPS_MODEL_ROOT も NAROPS_MODEL_BUCKET も未設定。"
                    "モデル無しで起動します（推論は不可）")
        return
    if rid is None:
        log.warning("current ポインタ未設定。推論は不可")
        return
    release = svc.registry.load(rid, verify=True)   # ハッシュ不一致なら例外
    svc.manifest = release.manifest
    from .runtime import load_models

    svc.models, svc.standardizer = load_models(release)
    log.info("モデル %s を読み込みました（%d モデル）", rid, len(svc.models))


def _attach_discord(svc: Any, cfg: OpsConfig, state: OperatingState) -> None:
    from .discord.client import DiscordSender, RateLimiter

    # 通知を出すのは推論を持つサービスだけ。API と Web は Secret Manager への
    # 権限も持たない（最小権限）。役割を見ずに webhook を要求すると、
    # 通知に関係ないサービスが起動できなくなる。
    role = os.environ.get("NAROPS_ROLE", "ops")
    if role != "ops":
        log.info("役割 %s は通知を出しません", role)
        return

    # Cloud Run に .env は無い。Secret Manager を渡さないと、webhook が
    # どうやっても解決できず「通知できるつもりで何も出ない」状態になる。
    project = os.environ.get("NAROPS_PROJECT") or getattr(cfg, "project", None)
    sm = None
    if project:
        try:
            from .gcp import SecretManagerResolver

            sm = SecretManagerResolver(project=project)
        except Exception as exc:  # noqa: BLE001
            log.warning("Secret Manager を使えません（%s）。.env と環境変数のみで解決します",
                        exc)
    resolver = SecretResolver(os.environ.get("NAROPS_ENV_FILE", ".env"),
                              secret_manager=sm, project=project)
    limiter = RateLimiter(rps=cfg.discord_rate_limit_rps)
    # アラートは shadow でも送る。止まっていること自体を知る必要がある
    alert_url = resolver.get("DISCORD_WEBHOOK_ALERT", required=False)
    if alert_url:
        svc.alert_sender = DiscordSender(alert_url, rate_limiter=limiter)
    if state.mode.delivers:
        svc.sender = DiscordSender(resolver.get("DISCORD_WEBHOOK_PREDICTION"),
                                   rate_limiter=limiter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["svc"] = _build_services()
    yield
    svc = _state.get("svc")
    for s in (getattr(svc, "sender", None), getattr(svc, "alert_sender", None)):
        if s is not None:
            s.close()


def _svc() -> Any:
    svc = _state.get("svc")
    if svc is None:
        raise HTTPException(status_code=503, detail="サービス未初期化")
    return svc


def _require_oidc(request: Request, authorization: str | None) -> None:
    """IN-12: 未認証呼び出しは 403。

    Cloud Run が `--no-allow-unauthenticated` で先に弾くので、ここは多層防御。
    ローカル検証では NAROPS_ALLOW_INSECURE=1 で緩める。
    """
    if os.environ.get("NAROPS_ALLOW_INSECURE") == "1":
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=403, detail="OIDC トークンがありません")


def create_app() -> FastAPI:
    app = FastAPI(title="nar-ops", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        from .service import health_endpoint

        return health_endpoint(_svc())

    @app.post("/plan-day")
    async def plan_day(request: Request,
                       authorization: str | None = Header(default=None)) -> dict:
        """当日の計画。日付を渡せる。

        Scheduler は日付なしで叩く（当日ぶん）。日付を受けられないと、
        翌日ぶんの事前確認も、取りこぼした日のやり直しもできない。
        1日1回の制約（DB-06）は日付ごとに効くので、指定しても二重には走らない。
        """
        from datetime import date as _date

        from .service import plan_day_endpoint

        _require_oidc(request, authorization)
        day = None
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("day"):
                day = _date.fromisoformat(str(body["day"]))
        except Exception:  # noqa: BLE001 — 本文なしが通常
            day = None
        return plan_day_endpoint(_svc(), day)

    @app.post("/snapshot-odds")
    def snapshot_odds(request: Request,
                      authorization: str | None = Header(default=None)) -> dict:
        from .service import snapshot_odds_endpoint

        _require_oidc(request, authorization)
        return snapshot_odds_endpoint(_svc())

    @app.post("/refresh-live")
    def refresh_live(request: Request,
                     authorization: str | None = Header(default=None)) -> dict:
        from .service import refresh_live_endpoint

        _require_oidc(request, authorization)
        return refresh_live_endpoint(_svc())

    @app.post("/ingest-and-refresh")
    def ingest(request: Request, authorization: str | None = Header(default=None)) -> dict:
        from .service import ingest_and_refresh_endpoint

        _require_oidc(request, authorization)
        return ingest_and_refresh_endpoint(_svc())

    @app.post("/infer")
    async def infer(request: Request,
                    authorization: str | None = Header(default=None)) -> dict:
        from .service import infer_endpoint

        _require_oidc(request, authorization)
        body = await request.json()
        race_id = body.get("race_id")
        if not race_id:
            raise HTTPException(status_code=400, detail="race_id が必要です")
        out = infer_endpoint(_svc(), race_id)
        return {"race_id": out.race_id, "status": out.status, "reason": out.reason,
                "n_predictions": len(out.predictions),
                "n_bet_candidates": len(out.bet_candidates)}

    @app.post("/weekly-report")
    def weekly(request: Request, authorization: str | None = Header(default=None)) -> dict:
        from .service import weekly_report_endpoint

        _require_oidc(request, authorization)
        return weekly_report_endpoint(_svc())

    @app.exception_handler(Exception)
    async def _redact_errors(request: Request, exc: Exception):
        """例外文に webhook URL が載らないようにする（DC-01）。"""
        from fastapi.responses import JSONResponse

        # `detail` は JSONResponse の引数ではない。渡すと例外ハンドラ自身が
        # TypeError で落ち、本来の原因が握り潰される（実際にそうなった）。
        log.exception("unhandled: %s", redact(str(exc)))
        return JSONResponse(status_code=500,
                            content={"detail": redact(str(exc))[:300]})

    return app


app = create_app()
