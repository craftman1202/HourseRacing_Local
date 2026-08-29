"""nar-api を実データに接続する。

`narops.api.build_app()` は契約（Pydantic モデルとルーティング）だけを持ち、
`services` に渡された callable が実処理を行う。ここでその callable を
BigQuery / GCS（本番）または DuckDB / ローカル FS（テスト・開発）に対して実装する。

`create_app()` は `wh`/`registry`/`cfg`/`clock` を省略すると本番構成
（BigQueryWarehouse + GcsModelRegistry）を自分で組み立てる。テストからは
これらを直接渡してインメモリ実装に差し替えられる — `narops.api.build_app` と
同じ DI の考え方をそのまま踏襲している。

`nar-api` は BigQuery 読み取り専用（`infra/services.json` の
`nar-api@...` は `bigquery.dataViewer`/`jobUser` のみ）。書き込みを一切しない
（`/models/{id}/promote` だけが例外で、GCS の `current.json` を書き換える —
これは `ModelRegistry`/`GcsModelRegistry` が担うので、ここからは直接 GCS に
書き込まない）。
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

from .clock import Clock, SystemClock, business_date, to_utc
from .config import OpsConfig
from .mode import OperatingState

UTC = timezone.utc

_TRACK_NAMES: dict[int, str] | None = None


def _track_names() -> dict[int, str]:
    """baba_code → 競馬場名。学習側の TrackMaster（名前→コード）の逆引き。

    ここで再実装せず既存のマスタをそのまま借りる（廃止場は含まれないので、
    未知コードは `f"場{code}"` にフォールバックする — 実在しない名前を
    でっち上げるより、コードのまま出すほうが安全）。
    """
    global _TRACK_NAMES
    if _TRACK_NAMES is None:
        from nar.transform.keys import TrackMaster

        _TRACK_NAMES = {code: name for name, code in TrackMaster().mapping.items()}
    return _TRACK_NAMES


def _manifest_of(registry: Any, release_id: str):
    """`ModelRegistry`（ローカル）と `GcsModelRegistry`（本番）の両方に対応する。

    ローカル版は `.load(id, verify=...).manifest`、GCS 版は軽量な
    `.load_manifest(id)` を持つ（アーティファクト本体を落とさず manifest.json
    だけ読む）。ダックタイピングで吸収する。
    """
    if hasattr(registry, "load_manifest"):
        return registry.load_manifest(release_id)
    return registry.load(release_id, verify=False).manifest


def create_app(*, wh: Any = None, registry: Any = None, cfg: OpsConfig | None = None,
               clock: Clock | None = None, state: OperatingState | None = None,
               principal: Any = None):
    """`principal` を省略すると `Principal("web-viewer", "admin")` を使う。

    nar-api は `--no-allow-unauthenticated`（services.json）で、呼び出せるのは
    nar-web の SA だけ（run.invoker のみ、他は何も持たない）。さらに nar-web
    自身が Auth.js + 許可リストで人間を弾いてから初めて nar-api を呼ぶ。
    つまり nar-api に届いた時点で「許可リストに載った本人」であることは
    二重に保証済みで、billing が無い（課金プランという概念自体が無い）この
    構成では、それ以上の per-user 判定をここに持ち込む必要がない。
    テストで free/anonymous 相当の挙動（AU-01/WB-07 の 403/404）を検証したい
    場合は `principal=Principal("x", "free")` を明示的に渡す。
    """
    from . import api

    cfg = cfg or OpsConfig.load()
    clock = clock or SystemClock()
    state = state or OperatingState.from_env()
    if principal is None:
        principal = api.Principal("web-viewer", "admin")

    if wh is None:
        wh = _default_warehouse(cfg)
    if registry is None:
        registry = _default_registry(cfg)

    services = _build_services(wh, registry, cfg, clock, state)
    services["principal"] = principal
    return api.build_app(services)


def _default_warehouse(cfg: OpsConfig) -> Any:
    """`duckdb` は軽量な nar-api イメージに入れていない。

    BigQuery 経路では `narops.db.backend`（duckdb 依存）を import しないよう、
    分岐の中でだけ import する — トップレベルで import すると本番イメージが
    duckdb 無しで ImportError になる。
    """
    backend = os.environ.get("NAROPS_BACKEND", "bigquery")
    if backend == "bigquery":
        from .gcp import BigQueryWarehouse

        return BigQueryWarehouse(project=cfg.project,
                                 dataset=cfg.raw["gcp"]["resources"]["bq_dataset"],
                                 location=cfg.region, max_bytes_billed=cfg.max_bytes_billed)
    from .db import schema
    from .db.backend import Warehouse

    wh = Warehouse(os.environ.get("NAROPS_DB", ":memory:"),
                   max_bytes_billed=cfg.max_bytes_billed)
    schema.create_all(wh)
    return wh


def _default_registry(cfg: OpsConfig) -> Any:
    """起動時に current を読めないなら落とす（DC-02: 無音の欠損状態を作らない）。"""
    root = os.environ.get("NAROPS_MODEL_ROOT")
    if root:
        from .model.registry import ModelRegistry

        return ModelRegistry(root)
    bucket = os.environ.get("NAROPS_MODEL_BUCKET") or cfg.raw["gcp"]["resources"].get(
        "gcs_model")
    if not bucket:
        raise RuntimeError(
            "NAROPS_MODEL_BUCKET も NAROPS_MODEL_ROOT も未設定です。"
            "current リリースを読めないまま起動できません（DC-02）。")
    from .gcp import GcsModelRegistry

    return GcsModelRegistry(bucket=bucket, project=cfg.project)


def _build_services(wh: Any, registry: Any, cfg: OpsConfig, clock: Clock,
                    state: OperatingState) -> dict[str, Any]:
    return {
        "health": lambda: _health(wh, registry, clock, state),
        "race": lambda race_id: _race(wh, state, race_id),
        "races_today": lambda: _races_today(wh, clock),
        "oos": lambda: _oos(registry),
        "live": lambda: _live(wh),
        "models": lambda: _models(registry),
        "promote": lambda release_id, actor, confirmed: _promote(
            wh, registry, release_id, actor, confirmed),
    }


# ------------------------------------------------------------------- races_today
def _races_today(wh: Any, clock: Clock) -> list[dict]:
    """ダッシュボードのタイムライン。当日開催 + あれば推奨の最大 EV 補正値。

    `race_schedule` を基準に LEFT JOIN 相当（pandas 側で merge）する。
    予測が無いレースも「まだ計算されていない」として一覧には出す
    （黙って消さない — DC-02 と同じ考え方）。
    """
    today = business_date(clock.now())

    sched = wh.query(
        "SELECT race_id, race_no, baba_code, start_ts FROM race_schedule "
        "WHERE race_date = ? ORDER BY start_ts", [today], allow_full_scan=True)
    if sched.empty:
        return []

    pred = wh.query(
        "SELECT race_id, MAX(ev_adjusted) AS top_ev_adjusted, COUNT(*) AS n "
        "FROM prediction WHERE race_date = ? AND is_shadow = false GROUP BY race_id",
        [today], allow_full_scan=True)
    top_ev_by_race = (dict(zip(pred["race_id"].tolist(), pred["top_ev_adjusted"].tolist()))
                      if len(pred) else {})
    has_pred = set(pred["race_id"].tolist()) if len(pred) else set()

    now = clock.now()
    out = []
    for row in sched.itertuples():
        computed = row.race_id in has_pred
        if computed:
            status = "ok"
        elif to_utc(row.start_ts) < now:
            status = "expired"
        else:
            status = "not_computed"
        top_ev = top_ev_by_race.get(row.race_id)
        out.append({
            "race_id": row.race_id,
            "race_no": int(row.race_no),
            "track_name": _track_names().get(int(row.baba_code), f"場{row.baba_code}"),
            "start_ts": row.start_ts,
            "status": status,
            "top_ev_adjusted": None if _is_nan(top_ev) else float(top_ev),
        })
    return out


# ---------------------------------------------------------------------- health
def _health(wh: Any, registry: Any, clock: Clock, state: OperatingState) -> dict:
    from .db import freshness

    f = freshness.check(wh, clock)

    skew_rows = wh.query(
        "SELECT verdict FROM skew_check ORDER BY checked_at DESC LIMIT 1",
        allow_full_scan=True)
    skew_verdict = skew_rows["verdict"].iloc[0] if len(skew_rows) else "UNKNOWN"
    if skew_verdict not in ("PASS", "FAIL"):
        skew_verdict = "UNKNOWN"

    model_release = None
    model_trained_on = None
    current_id = registry.current_id()
    if current_id is not None:
        try:
            manifest = _manifest_of(registry, current_id)
            model_release = manifest.model_id
            model_trained_on = date.fromisoformat(manifest.train_period["end"])
        except Exception:  # noqa: BLE001 — health は current が壊れていても返す
            pass

    today = business_date(clock.now())
    stats = wh.query("SELECT COUNT(DISTINCT race_id) n FROM prediction WHERE race_date = ?",
                     [today], allow_full_scan=True)
    scheduled = wh.query(
        "SELECT COUNT(DISTINCT race_id) n FROM race_schedule WHERE race_date = ?",
        [today], allow_full_scan=True)
    # 今日レースが無い日は「未達成」ではなく「対象0件」— weekly_report_endpoint
    # と同じ約束（scheduled 側 service.py L486 と合わせる）で 1.0 を返す。
    coverage_today = (float(stats["n"].iloc[0]) / float(scheduled["n"].iloc[0])
                      if len(scheduled) and scheduled["n"].iloc[0] else 1.0)

    return {
        "db_watermark": f.watermark,
        "model_release": model_release,
        "model_trained_on": model_trained_on,
        "skew_verdict": skew_verdict,
        "coverage_today": round(coverage_today, 4),
        "delivery_blocked": state.delivery_blocked,
        "blocked_reason": state.blocked_reason or None,
    }


# ------------------------------------------------------------------------ race
def _race(wh: Any, state: OperatingState, race_id: str) -> dict:
    from .api import NotFound

    if len(race_id) != 12 or not race_id.isdigit():
        raise NotFound(f"race_id の形式が不正です: {race_id!r}")
    baba_code = int(race_id[0:2])
    race_date = date(int(race_id[2:6]), int(race_id[6:8]), int(race_id[8:10]))

    pred = wh.query(
        "SELECT horse_no, model_release, track_used, p_win, p_market, ev, ev_adjusted "
        "FROM prediction WHERE race_id = ? AND race_date = ? AND is_shadow = false",
        [race_id, race_date])
    if pred.empty:
        raise NotFound(f"レース {race_id} の推論結果がありません")

    sched = wh.query(
        "SELECT race_no, start_ts FROM race_schedule WHERE race_id = ? AND race_date = ?",
        [race_id, race_date], allow_full_scan=True)
    if sched.empty:
        raise NotFound(f"レース {race_id} の開催情報がありません")

    bets = wh.query(
        "SELECT horse_no, SUM(stake_yen) AS stake_yen FROM bet_candidate "
        "WHERE race_id = ? AND race_date = ? GROUP BY horse_no",
        [race_id, race_date])
    stake_by_horse = (dict(zip(bets["horse_no"].tolist(), bets["stake_yen"].tolist()))
                      if len(bets) else {})

    horses = []
    for row in pred.sort_values("p_win", ascending=False).itertuples():
        p_market = None if _is_nan(row.p_market) else float(row.p_market)
        ev = None if _is_nan(row.ev) else float(row.ev)
        ev_adjusted = None if _is_nan(row.ev_adjusted) else float(row.ev_adjusted)
        horses.append({
            "horse_no": int(row.horse_no),
            "horse_name": None,  # entry_history 経由の氏名解決は未実装（今回は未提供）
            "p_win": float(row.p_win),
            "p_market": p_market,
            "ev": ev,
            "ev_adjusted": ev_adjusted,
            "stake_yen": int(stake_by_horse.get(row.horse_no, 0)),
        })

    return {
        "race_id": race_id,
        "race_date": race_date,
        "start_ts": sched["start_ts"].iloc[0],
        "track_name": _track_names().get(baba_code, f"場{baba_code}"),
        "race_no": int(sched["race_no"].iloc[0]),
        "model_release": str(pred["model_release"].iloc[0]),
        "track_used": str(pred["track_used"].iloc[0]),
        "status": "ok",
        "horses": horses,
        "is_paper": state.mode.is_paper,
    }


def _is_nan(v: Any) -> bool:
    import math

    try:
        return v is None or math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------------- oos
def _oos(registry: Any) -> list[dict]:
    current_id = registry.current_id()
    if current_id is None:
        raise RuntimeError("current リリースが未設定です（/models で bootstrap してください）")
    manifest = _manifest_of(registry, current_id)
    metrics = manifest.oos_metrics
    evaluated_at = manifest.oos_evaluated_on or manifest.created_at
    return [{
        "model": "ensemble",
        "track": manifest.track,
        "race_nll": metrics.get("race_nll", float("nan")),
        "top1": metrics.get("top1", float("nan")),
        "top3": metrics.get("top3", float("nan")),
        "brier": metrics.get("brier", float("nan")),
        "ece": metrics.get("ece", float("nan")),
        "evaluated_at": datetime.fromisoformat(evaluated_at) if evaluated_at
                        else datetime.now(UTC),
    }]


# ------------------------------------------------------------------------ live
def _live(wh: Any) -> list[dict]:
    rows = wh.query(
        "SELECT business_date, n_races, n_bets, stake_yen, return_yen, roi, hit_rate, "
        "is_paper FROM pnl_daily ORDER BY business_date DESC", allow_full_scan=True)
    return rows.to_dict("records")


# ---------------------------------------------------------------------- models
def _models(registry: Any) -> list[dict]:
    current_id = registry.current_id()
    out = []
    for rid in registry.releases():
        try:
            m = _manifest_of(registry, rid)
        except Exception:  # noqa: BLE001 — 読めないリリースは一覧から除外するだけ
            continue
        out.append({
            "release_id": rid,
            "is_current": rid == current_id,
            "model_id": m.model_id,
            "dataset_version": m.dataset_version,
            "train_period_start": date.fromisoformat(m.train_period["start"]),
            "train_period_end": date.fromisoformat(m.train_period["end"]),
            "git_commit": m.git_commit,
            "track": m.track,
            "purpose": m.purpose,
            "oos_metrics": m.oos_metrics,
            "created_at": datetime.fromisoformat(m.created_at),
        })
    return out


# -------------------------------------------------------------------- promote
def _promote(wh: Any, registry: Any, release_id: str, actor: str, confirmed: bool) -> None:
    """WB-06 の実ゲート。シャドー実績は `prediction.is_shadow` から実測する。

    CLI (`narops promote`) は shadow/production の NLL・ECE・top1 を人間が
    フラグで渡す運用だが、Web からの昇格はそれをやらせたくない
    （手入力できる=改ざんできる、が本質）。ここでは
    `prediction(model_release=release_id, is_shadow=true)` を実測データとして
    `nar.eval.metrics.summary()` にかけ、`release.assert_promotion_allowed` に
    そのまま渡す。0件なら日数0で確実にゲートに落ちる（機械的に拒否、握り潰さない）。
    """
    from .release import SHADOW_DAYS, ShadowMetrics
    from .release import promote as release_promote
    from .shared import summary_metrics

    shadow_df = wh.query(
        "SELECT p.race_id, p.horse_no, p.p_win, p.race_date, e.is_win, e.finish_pos "
        "FROM prediction p JOIN entry_history e "
        "  ON p.race_id = e.race_id AND p.horse_no = e.horse_no "
        "WHERE p.model_release = ? AND p.is_shadow = true AND p.race_date <= CURRENT_DATE",
        [release_id], allow_full_scan=True)

    if shadow_df.empty:
        shadow = ShadowMetrics(days=0, nll=float("inf"), ece=float("inf"), top1=0.0)
    else:
        summary = summary_metrics(
            shadow_df["p_win"].to_numpy(),
            shadow_df["is_win"].to_numpy(),
            shadow_df["finish_pos"].to_numpy(),
            shadow_df["race_id"].to_numpy(),
        )
        shadow = ShadowMetrics(days=int(shadow_df["race_date"].nunique()),
                               nll=float(summary["race_nll"]), ece=float(summary["ece"]),
                               top1=float(summary["top1"]))

    current_id = registry.current_id()
    if current_id is None:
        raise RuntimeError("current リリースが未設定です。先に bootstrap してください。")
    prod_metrics = _manifest_of(registry, current_id).oos_metrics
    production = ShadowMetrics(days=SHADOW_DAYS,
                               nll=float(prod_metrics.get("race_nll", float("inf"))),
                               ece=float(prod_metrics.get("ece", 0.0)),
                               top1=float(prod_metrics.get("top1", 0.0)))

    release_promote(registry, release_id, shadow, production, actor=actor,
                    confirmed=confirmed)
