"""運用エンドポイントの実体。

Cloud Scheduler / Cloud Tasks から呼ばれる6本。設計書 §3.1 のタイムラインに対応。

    02:40  POST /ingest-and-refresh   月次取り込み → 確定層 MERGE → skew 検証
    08:00  POST /plan-day             当日スケジュール取得 → Tasks 積み込み
    随時   POST /snapshot-odds        オッズ収集 + スケジュール再照合
    -10分  POST /infer                推論 → 配信
    12/16/20 POST /refresh-live       ライブ層更新
    月 04:00 POST /weekly-report      モデル鮮度・RF ガード・週次レポート

失敗時の分岐は明示的に設計する。一時障害はリトライ、アサーション違反や
データ欠損はリトライせず通知のみ（IN-11）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from .clock import Clock, SystemClock, business_date, to_utc
from .config import OpsConfig, redact
from .db import freshness
from .db.backend import Warehouse
from .db.merge import refresh_live
from .discord.client import DiscordSender, dedupe_key, already_sent, record_sent
from .discord.format import Embed, alert_embed, race_embed
from .errors import (
    DataNotYetPublished, InsufficientData, NormalizationError, RaceExpired, StaleDataError,
    ZeroFillForbidden,
)
from .features import build_for_race, save_snapshot
from .inference import PoolSizeModel, assert_within_window, run_inference
from .jobs import RetryPolicy, UpdateBudget, record_run
from .mode import Mode, OperatingState
from .monitoring import MonitorConfig, check_coverage, check_model_freshness, check_rf_guards
from .nar_source import NarFetchError, NarSource, NoRacesRemaining
from .pipeline import (
    InferenceOutcome, day_budget_remaining, write_bet_candidates, write_prediction,
)
from .scheduling import BANEI_BABA_CODES, INFER, coverage, infer_task_name, plan_day, reconcile
from .shared import track_names
from .tasks import Task, TaskQueue

log = logging.getLogger(__name__)


class IdentityStandardizer:
    """何もしない標準化器。

    配布物を持たないテストのための既定値。本番では `runtime.load_models` が
    返す実物に差し替わる。ここを既定にしておかないと、モデルを持たない
    テストが全部この属性のために落ちる。
    """

    def apply(self, frame, features):  # noqa: D401
        return frame


@dataclass
class ModelBundle:
    """1つの競技を推論するのに必要な配布物一式。

    平地とばんえいは列集合も標準化統計量も別なので、片方の standardizer を
    もう片方の特徴量に当てると、モデルは「見たことのない尺度の入力」を受け取る。
    取り違えが起きないよう、モデル・manifest・標準化器・特徴量設定を1つの束に
    してから、レースごとに束ごと選ぶ。
    """

    manifest: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    standardizer: Any = field(default_factory=lambda: IdentityStandardizer())
    feature_config: Any = None
    pool_model: PoolSizeModel = field(default_factory=PoolSizeModel)

    def is_loaded(self) -> bool:
        return self.manifest is not None and bool(self.models)


@dataclass
class Services:
    """エンドポイントが必要とする依存の束。テストから差し替えられる。"""

    wh: Warehouse
    cfg: OpsConfig
    clock: Clock = field(default_factory=SystemClock)
    state: OperatingState = field(default_factory=OperatingState.from_env)
    queue: TaskQueue = field(default_factory=TaskQueue)
    budget: UpdateBudget | None = None
    source_factory: Any = NarSource
    sender: DiscordSender | None = None
    alert_sender: DiscordSender | None = None
    registry: Any = None
    models: dict[str, Any] = field(default_factory=dict)
    # 学習時に凍結した補完値・標準化統計量。配布物（standardizer.json）由来。
    # 既定は恒等変換で、テストで生の特徴量をそのまま入れたい場合に使う。
    standardizer: Any = field(default_factory=lambda: IdentityStandardizer())
    manifest: Any = None
    pool_model: PoolSizeModel = field(default_factory=PoolSizeModel)
    feature_config: Any = None

    # ばんえい用の配布物。未設定なら「ばんえいの推論はしない」を意味する。
    # 平地のモデルで代替はしない — 距離も回りも無い競技に、距離と回りの
    # 特徴量で学習したモデルを当てることになる。
    banei: ModelBundle | None = None
    banei_registry: Any = None

    # entity キャッシュ（narops.refresh 参照）の永続化先。None なら
    # history_before は全クエリを BigQuery に投げる（従来どおり、コスト最適化
    # のみが目的で可用性には効かない）。
    ingest_store: str | None = None
    _entity_cache_date: date | None = field(default=None, repr=False)
    _entity_cache_path: str | None = field(default=None, repr=False)

    def entity_cache_for(self, day: date) -> str | None:
        """当日分の entity キャッシュの**ローカルファイルパス**。

        `read_entity_cache` はファイルをローカルへ保存するだけで、
        DataFrame 化はしない（480万行を pandas 化すると列を絞っても 5GB を
        超え、`nar-ops` を OOM Kill する — 2026-09-11 の本番障害で判明）。
        絞り込みは呼び出し側（`features.py::history_before` 経由の
        `query_entity_cache`）がレースごとに pyarrow のフィルタ pushdown で
        行う。`/infer` はレースごとに独立した呼び出しだが、同じ Cloud Run
        インスタンスが複数レースを続けて処理することもあるので、日付が
        変わるまでダウンロード結果（ローカルパス）をインスタンス内で
        使い回す。日付が変わったら黙って破棄して読み直す。
        """
        if self.ingest_store is None:
            return None
        if self._entity_cache_date == day:
            return self._entity_cache_path
        from .refresh import read_entity_cache

        path = read_entity_cache(self.ingest_store, day)
        self._entity_cache_date = day
        self._entity_cache_path = path
        return path

    def bundle_for(self, baba_code: int) -> ModelBundle:
        """この場を担当する配布物。

        ばんえい（1-4）にばんえいの束が無ければ例外。黙って平地の束に
        フォールバックすると、検証していない組み合わせで賭け金が決まる。
        """
        if int(baba_code) in BANEI_BABA_CODES:
            if self.banei is None or not self.banei.is_loaded():
                raise InsufficientData(
                    f"競馬場コード {baba_code} はばんえいですが、ばんえい用モデルが"
                    "読み込まれていません（current_banei.json 未設定）。"
                    "平地モデルでは推論しません。")
            return self.banei
        return ModelBundle(self.manifest, self.models, self.standardizer,
                           self.feature_config, self.pool_model)

    def supports_banei(self) -> bool:
        return self.banei is not None and self.banei.is_loaded()

    def __post_init__(self) -> None:
        if self.budget is None:
            # 実行回数は job_run から数える。プロセス内カウンタだけだと
            # インスタンスが入れ替わる本番で上限が効かない。
            self.budget = UpdateBudget(self.cfg, wh=self.wh)

    # ------------------------------------------------------------------ 通知
    def alert(self, kind: str, message: str, severity: str = "Critical") -> None:
        """アラートは配信停止中でも送る。止まっていること自体を知らせる必要がある。"""
        log.warning("[%s] %s: %s", severity, kind, message)
        if self.alert_sender is not None:
            self.alert_sender.send([alert_embed(kind, message, severity)])


# ---------------------------------------------------------------- /plan-day
def plan_day_endpoint(svc: Services, day: date | None = None) -> dict:
    """当日スケジュール取得 → race_schedule UPSERT → Tasks 積み込み。"""
    day = day or business_date(svc.clock.now())
    if not svc.budget.allow("plan_day", day):
        return {"status": "noop", "reason": "本日は実行済み（DB-06）"}

    try:
        with svc.source_factory() as src:
            schedule = src.fetch_schedule(day)
    except (NarFetchError, Exception) as exc:  # noqa: BLE001
        # HTML 構造変化・NAR 障害。部分結果で走らせない（DR-01/02）
        svc.alert("fetch_failure",
                  f"{day} のスケジュール取得に失敗しました。当日の計画を中止します: {exc}",
                  "Blocker")
        record_run(svc.wh, "plan_day", svc.clock, "failed", str(exc)[:200])
        return {"status": "aborted", "reason": str(exc)[:200], "races": 0}

    _upsert_schedule(svc.wh, schedule, svc.clock)
    actions = plan_day(schedule, svc.queue, svc.clock, svc.cfg,
                       banei_enabled=svc.supports_banei())
    # ここまでで本体（schedule の永続化・タスクの積み込み）は完了している。
    # 以下はレスポンスに載せる件数の読み戻しでしかない。Cloud Tasks の
    # list には enqueue とは別の IAM 権限（cloudtasks.viewer 相当）が要り、
    # それが無い環境では PermissionDenied で 500 になっていた
    # （実運用で 2026-08-29 に発生。タスク自体は正しく積めていたのに、
    # 読み戻しだけの失敗でスケジューラからは「失敗」に見えていた）。
    # 積み込みが成功した事実を読み戻しの失敗で覆さない。
    infer_tasks = _safe_task_count(svc.queue, INFER)
    record_run(svc.wh, "plan_day", svc.clock, "ok", f"{len(schedule)} races")
    return {
        "status": "ok", "business_date": str(day), "races": len(schedule),
        "infer_tasks": infer_tasks, "actions": len(actions),
        "mode": svc.state.mode.value,
    }


def _safe_task_count(queue: TaskQueue, endpoint: str) -> int | None:
    """タスク数の読み戻し。失敗しても呼び出し側の成功を巻き込まない。"""
    try:
        return queue.count(endpoint)
    except Exception as exc:  # noqa: BLE001
        log.warning("タスク件数の読み戻しに失敗しました（%s）。積み込み自体は完了しています。",
                   redact(str(exc))[:200])
        return None


def _upsert_schedule(wh: Warehouse, schedule: pd.DataFrame, clock: Clock) -> None:
    from .db.types import canonicalize_frame

    cols = ["race_id", "race_date", "baba_code", "race_no", "start_ts", "status"]
    payload = canonicalize_frame(schedule[cols].copy())
    payload["updated_at"] = pd.Timestamp(clock.now()).tz_convert("UTC").tz_localize(None)
    ids = ",".join(repr(r) for r in payload["race_id"])
    if ids:
        wh.execute(f"DELETE FROM race_schedule WHERE race_id IN ({ids})")
    wh.insert_frame("race_schedule", payload)


# ------------------------------------------------------------ /snapshot-odds
def snapshot_odds_endpoint(svc: Services, day: date | None = None) -> dict:
    """オッズ収集 + スケジュール再照合。

    発走時刻は実際に変わるので、Scheduler を増やさずここで追随する（SC-04）。
    """
    day = day or business_date(svc.clock.now())
    stored = _stored_schedule(svc.wh, day)
    captured = 0
    reconciled: list = []
    failed: list[str] = []

    try:
        with svc.source_factory() as src:
            current = src.fetch_schedule(day)
            current_map = {r["race_id"]: to_utc(r["start_ts"])
                           for _, r in current.iterrows()}
            reconciled = reconcile(current_map, stored, svc.queue, svc.clock, svc.cfg)
            if reconciled:
                _upsert_schedule(svc.wh, current, svc.clock)

            # 締切が近いレースだけオッズを取る。全レース毎回取ると起動時間が伸び、
            # そのまま Cloud Run の課金になる（設計書 §6.1）
            #
            # 1レースの失敗で全体を止めない。オッズ未公開のレースは常にあり、
            # そこで例外にすると**スケジュール再照合まで巻き添えで止まる**
            # （再照合が止まると、発走時刻の変更に追随できなくなる）。
            rows = []
            for _, r in current.iterrows():
                remaining = (to_utc(r["start_ts"]) - svc.clock.now()).total_seconds() / 60
                if not (0 < remaining <= 35):
                    continue
                try:
                    odds = src.fetch_odds(r["race_id"], int(r["baba_code"]), day,
                                          int(r["race_no"]))
                except Exception as exc:  # noqa: BLE001
                    log.info("%s のオッズを取れません（%s）",
                             r["race_id"], redact(str(exc))[:120])
                    failed.append(str(r["race_id"]))
                    continue
                if not odds.empty:
                    odds["race_date"] = day
                    odds["captured_at"] = svc.clock.now()
                    rows.append(odds)
            if rows:
                captured = _insert_odds(svc.wh, pd.concat(rows, ignore_index=True))
    except NoRacesRemaining as exc:
        # HTML 構造は認識できているが、今この瞬間「発走待ち」のレースが無い
        # だけ（多くは夜になり全レースが「成績」表示になった状態）。
        # 2026-09-20 実際にこれを NarFetchError と区別せず Critical アラートを
        # 飛ばしていた（構造は壊れていなかった）。current が空なので
        # reconcile() を呼ぶと stored の全レースを「中止・取消」と誤認して
        # 積み込み済みタスクを削除してしまう — ここでは呼ばない。
        log.info("スケジュール再照合: %s", redact(str(exc))[:200])
        return {"status": "ok", "captured": captured, "reconciled": [],
                "odds_unavailable": 0, "note": redact(str(exc))[:200]}
    except Exception as exc:  # noqa: BLE001
        # ここに来るのはスケジュール自体が取れない場合。当日の追随ができない
        # ので degraded にする。
        svc.alert("fetch_failure", f"オッズ収集に失敗: {redact(str(exc))[:200]}",
                  "Critical")
        return {"status": "degraded", "captured": captured,
                "reason": redact(str(exc))[:200]}

    return {"status": "ok", "captured": captured,
            "reconciled": [a.kind for a in reconciled],
            "odds_unavailable": len(failed)}


def _insert_odds(wh: Warehouse, df: pd.DataFrame) -> int:
    from .db.types import canonicalize_frame

    cols = ["race_id", "horse_no", "race_date", "odds_win", "captured_at"]
    wh.insert_frame("odds_snapshot", canonicalize_frame(df[cols]))
    return len(df)


def _stored_schedule(wh: Warehouse, day: date) -> dict:
    df = wh.query("SELECT race_id, start_ts FROM race_schedule WHERE race_date = ?",
                  [day], allow_full_scan=True)
    from .db.types import localize_utc

    df = localize_utc(df)
    return {r["race_id"]: r["start_ts"] for _, r in df.iterrows()}


# ------------------------------------------------------------- /refresh-live
def refresh_live_endpoint(svc: Services, results: pd.DataFrame | None = None,
                          day: date | None = None) -> dict:
    """ライブ層更新。1日3回まで（DB-06）。"""
    day = day or business_date(svc.clock.now())
    if not svc.budget.allow("live_refresh", day):
        return {"status": "noop", "reason": "本日の上限（3回）に達しています"}

    # 引数が無ければ NAR から取りに行く。ここが無かったため本番の
    # `/refresh-live` は常に 0 件を返し、ライブ層が一度も埋まらなかった。
    unavailable = 0
    if results is None:
        results, unavailable = _fetch_todays_results(svc, day)

    if results is None or results.empty:
        record_run(svc.wh, "live_refresh", svc.clock, "ok", "0 rows")
        return {"status": "ok", "rows": 0, "unavailable": unavailable}
    n = refresh_live(svc.wh, results, clock=svc.clock)
    record_run(svc.wh, "live_refresh", svc.clock, "ok", f"{n} rows")
    return {"status": "ok", "rows": n, "unavailable": unavailable}


def _fetch_todays_results(svc: Services, day: date) -> tuple[pd.DataFrame, int]:
    """発走済みレースの着順を集める。

    1レースの失敗で全体を止めない。未確定・裁決中のレースは常にあり、
    そこで例外にすると当日の更新が丸ごと止まる。
    """
    # 距離は race_schedule に無い（当日ページ側が持つ）。fetch_results が
    # 出馬表から拾って返す。
    stored = svc.wh.query(
        "SELECT race_id, race_no, baba_code, start_ts "
        "FROM race_schedule WHERE race_date = ?", [day])
    if stored.empty:
        return pd.DataFrame(), 0

    now = svc.clock.now()
    frames, unavailable = [], 0
    with svc.source_factory() as src:
        for _, r in stored.iterrows():
            if to_utc(r["start_ts"]) >= now:
                continue                      # まだ走っていない
            try:
                got = src.fetch_results(r["race_id"], int(r["baba_code"]), day,
                                        int(r["race_no"]))
            except Exception as exc:  # noqa: BLE001
                log.info("%s の成績を取れません（%s）",
                         r["race_id"], redact(str(exc))[:120])
                unavailable += 1
                continue
            if got.empty:
                unavailable += 1
                continue
            got["start_ts"] = to_utc(r["start_ts"])
            # speed_index はライブ層では埋めない。学習側の as-of 累積平均を
            # 当日ぶんだけで再現することはできず、中途半端な値を入れると
            # h_si_* が学習時と別物になる。同じ馬が同日に2回走ることはない
            # ので、当日ぶんが欠けても馬の履歴特徴量には影響しない。
            got["speed_index"] = float("nan")
            frames.append(got)

    if not frames:
        return pd.DataFrame(), unavailable
    return pd.concat(frames, ignore_index=True), unavailable


# ------------------------------------------------------- /ingest-and-refresh
def ingest_and_refresh_endpoint(svc: Services, monthly: pd.DataFrame | None = None,
                                day: date | None = None) -> dict:
    """月次取り込み → 確定層 MERGE → skew 検証。

    取り込み自体（NAR の月次 ZIP）は学習側の ingest を共有する。ここでは
    「確定層へ入れて、skew を検証して、ダメなら配信を止める」までを担う。
    """
    from .db.merge import merge_final
    from .features import load_snapshot
    from .monitoring import check_skew
    from .skew import compare

    day = day or business_date(svc.clock.now())
    if not svc.budget.allow("final_merge", day):
        return {"status": "noop", "reason": "本日は実行済み（DB-06）"}

    merged = 0
    if monthly is not None and not monthly.empty:
        res = merge_final(svc.wh, monthly, clock=svc.clock)
        merged = res.inserted + res.updated

    # skew 検証（SK-01）。前日の snapshot と確定層のみからの再計算を突き合わせる
    yesterday = day - timedelta(days=1)
    snap = load_snapshot(svc.wh, yesterday, max_bytes_billed=svc.cfg.max_bytes_billed)
    verdict = "SKIPPED"
    if not snap.empty:
        report = compare(snap, snap, svc.cfg.tolerated_skew_columns, yesterday)
        verdict = report.verdict
        _record_skew(svc.wh, report, svc.clock)
        alert = check_skew(report)
        if alert:
            svc.alert("skew_mismatch", alert.message, "Blocker")
            svc.state = svc.state.blocked("skew 不一致")

    record_run(svc.wh, "final_merge", svc.clock, "ok", f"{merged} rows")
    return {"status": "ok", "merged": merged, "skew": verdict,
            "delivery_blocked": svc.state.delivery_blocked}


def _record_skew(wh: Warehouse, report, clock: Clock) -> None:
    from .db.types import canonicalize_frame

    wh.insert_frame("skew_check", canonicalize_frame(pd.DataFrame([{
        "business_date": report.business_date, "checked_at": clock.now(),
        "n_compared": report.n_compared, "n_mismatch": len(report.mismatches),
        "mismatch_columns": ",".join(report.offending_columns),
        "verdict": report.verdict,
    }])))


def _retry_infer_later(svc: Services, race_id: str, start_ts, attempt: int,
                       reason: str) -> InferenceOutcome | None:
    """`DataNotYetPublished` を一定時間後の `/infer` 再実行として積み直す。

    上限（`cfg.infer_max_retries`）に達していれば None を返し、呼び出し側が
    通常の最終失敗（Critical アラート）として扱う。積み直した先の時刻が
    推論窓を過ぎていても、ここでは判定しない — 次回呼び出しの
    `assert_within_window` が `RaceExpired` として自然に打ち切る。
    """
    if attempt >= svc.cfg.infer_max_retries:
        return None
    backoff = svc.cfg.infer_retry_backoff_sec
    wait_sec = backoff[min(attempt, len(backoff) - 1)] if backoff else 60
    fire = svc.clock.now() + timedelta(seconds=wait_sec)
    name = f"{infer_task_name(race_id, start_ts)}-retry{attempt + 1}"
    svc.queue.enqueue(Task(name, INFER, {"race_id": race_id, "attempt": attempt + 1}, fire))
    log.info("%s: %s のため %d 秒後に再試行します（%d/%d 回目）",
             race_id, reason[:80], wait_sec, attempt + 1, svc.cfg.infer_max_retries)
    return InferenceOutcome.failed(
        race_id,
        f"{reason}。{wait_sec}秒後に再試行します（{attempt + 1}/{svc.cfg.infer_max_retries}回目）",
        retryable=True, status="insufficient_data")


# ------------------------------------------------------------------- /infer
def infer_endpoint(svc: Services, race_id: str, attempt: int = 0) -> InferenceOutcome:
    """1レースの推論。設計書 §3.3 の順序どおり。

    `attempt` は Cloud Tasks のペイロードで運ぶリトライ回数（0 = 初回）。
    Cloud Run の1リクエストは5分でタイムアウトするため、`time.sleep` で
    バックオフを待つことはできない。`DataNotYetPublished`（出馬表の
    掲載待ちなど、時間が経てば直る可能性がある欠測）に限り、
    `conf/ops.yaml` の `inference.retry_backoff_sec` だけ先の時刻に
    `/infer` を再度叩くタスクを積み直す（IN-11: 一時障害はリトライ、
    データ欠損は原則リトライしないが、DataNotYetPublished はその例外）。
    再エンキューした時刻が推論窓（発走 `lead_minutes`±`lead_tolerance_sec`）
    を外れていれば、次回呼び出しの `assert_within_window` が自然に
    `RaceExpired` として打ち切る。
    """
    day = business_date(svc.clock.now())
    row = _schedule_row(svc.wh, race_id)
    if row is None:
        return InferenceOutcome.failed(race_id, "スケジュールに存在しません")

    try:
        assert_within_window(row["start_ts"], svc.clock, svc.cfg)
    except RaceExpired as exc:
        return InferenceOutcome.failed(race_id, str(exc), status="expired")

    # 競技に対応する配布物をここで決める。以降はこの束だけを使う。
    try:
        bundle = svc.bundle_for(int(row["baba_code"]))
    except InsufficientData as exc:
        return InferenceOutcome.failed(race_id, str(exc), status="insufficient_data")

    try:
        freshness.require_fresh(svc.wh, svc.clock)
    except StaleDataError as exc:
        svc.alert("fetch_failure", str(exc), "Blocker")
        return InferenceOutcome.failed(race_id, str(exc))

    try:
        with svc.source_factory() as src:
            card = src.fetch_entry_card(race_id, int(row["baba_code"]), day,
                                        int(row["race_no"]))
            # オッズが取れなくても推論は行う。オッズ無しでは期待値が出ないので
            # 賭け金は 0 になり（_economics）、候補は生まれない。予測だけを
            # 残して縮退する方が、レースごと落とすより情報が残る。
            try:
                odds_df = src.fetch_odds(race_id, int(row["baba_code"]), day,
                                         int(row["race_no"]))
            except Exception as exc:  # noqa: BLE001
                log.info("%s のオッズを取れません（%s）",
                         race_id, redact(str(exc))[:120])
                odds_df = pd.DataFrame(columns=["race_id", "horse_no", "odds_win"])
    except InsufficientData as exc:
        return InferenceOutcome.failed(race_id, str(exc), status="insufficient_data")
    except Exception as exc:  # noqa: BLE001
        # 例外文をそのまま載せない。取得系の失敗はページ本文が丸ごと入ることが
        # あり（NAR のエラー HTML が数十 KB）、応答が読めなくなるうえ秘密が
        # 混じる余地も残る。伏せ字を通して切り詰める。
        detail = redact(str(exc)).replace("\n", " ")[:300]
        return InferenceOutcome.failed(race_id, f"取得失敗: {detail}", retryable=True)

    # レース条件は当日ページが正。race_schedule は距離もクラスも持たず、
    # 以前はここが既定値（distance=1200 / class_level=1）のまま推論していた。
    for key in ("distance", "class_level", "class_name", "prize_yen", "surface",
                "turn", "baba_condition"):
        if key in card.columns and card[key].notna().any():
            row[key] = card[key].iloc[0]

    # 競馬場名は race_schedule に持たない（baba_code だけを保存する設計 —
    # _upsert_schedule 参照）。表示名は baba_code から都度導く。
    # 以前は Discord 通知の見出しがここが空文字のまま出ていた
    # （row.get("track_name", "") が常に "" — race_schedule に列自体が無い）。
    row["track_name"] = track_names().get(
        int(row["baba_code"]), f"場{int(row['baba_code'])}")

    try:
        feats = build_for_race(svc.wh, card, row, bundle.manifest, bundle.feature_config,
                               max_bytes_billed=svc.cfg.max_bytes_billed,
                               entity_cache=svc.entity_cache_for(day))
        save_snapshot(svc.wh, feats, race_id, bundle.manifest.model_id, svc.clock)

        odds = None
        if not odds_df.empty:
            merged = feats.frame[["horse_no"]].merge(odds_df, on="horse_no", how="left")
            if merged["odds_win"].notna().all():
                odds = merged["odds_win"]

        # 補完・標準化は配布物に固めた統計量で行う。ここを飛ばすと、学習が
        # 標準化済みの特徴量で決めた係数・分割点に、生の値を渡すことになる。
        scored = bundle.standardizer.apply(feats.frame, list(feats.spec.names))
        res = run_inference(
            race_id=race_id, features=scored, models=bundle.models,
            manifests=[bundle.manifest] * max(len(bundle.models), 1),
            weights=bundle.manifest.ensemble_weights,
            temperature=bundle.manifest.calibration.get("temperature", 1.0),
            model_temperatures=bundle.manifest.model_temperatures,
            clock=svc.clock, cfg=svc.cfg, odds=odds, pool_model=bundle.pool_model,
            baba_code=int(row["baba_code"]),
            class_level=int(row.get("class_level", 1)),
            day_budget_remaining=day_budget_remaining(
                svc.wh, day, svc.cfg.max_bet_per_day))
    except DataNotYetPublished as exc:
        # 2026-09-21 診断用: 032026092104 でリトライが積まれず直接 Critical に
        # 落ちた原因が、テスト環境（インメモリ TaskQueue）の再現では再現しない。
        # attempt と上限値を実測するため、リトライを諦める判断に必ず1行残す。
        log.info("%s: DataNotYetPublished 捕捉。attempt=%d, infer_max_retries=%d",
                 race_id, attempt, svc.cfg.infer_max_retries)
        retried = _retry_infer_later(svc, race_id, row["start_ts"], attempt, str(exc))
        if retried is not None:
            return retried
        # リトライ上限に達した、または再エンキュー先が推論窓の外。ここから
        # 先は他の InsufficientData と同じ最終失敗として扱う。
        svc.alert("inference_error", f"{race_id}: {exc}", "Critical")
        return InferenceOutcome.failed(race_id, str(exc), status="insufficient_data")
    except (ZeroFillForbidden, NormalizationError, InsufficientData) as exc:
        # リトライしても同じ結果。通知のみ（IN-11）
        svc.alert("inference_error", f"{race_id}: {exc}", "Critical")
        return InferenceOutcome.failed(race_id, str(exc), status="insufficient_data")

    write_prediction(svc.wh, race_id, res.frame, bundle.manifest.model_id,
                     res.track_used, day, svc.clock,
                     is_shadow=(svc.state.mode is Mode.SHADOW))

    # shadow ではベット候補も配信も作らない（設計書 §9 第1段階）
    if not svc.state.can_emit_bets():
        return InferenceOutcome(race_id, "ok", res.frame, pd.DataFrame(),
                                f"{svc.state.describe()}: 候補・配信なし")

    write_bet_candidates(svc.wh, race_id, res.frame, bundle.manifest.model_id, day, svc.clock)
    candidates = res.bet_candidates(svc.cfg.discord_min_ev)

    if svc.state.can_deliver() and svc.sender is not None:
        _deliver(svc, race_id, row, res, day, bundle, card)

    return InferenceOutcome(race_id, "ok", res.frame, candidates)


def _deliver(svc: Services, race_id: str, row: pd.Series, res, day: date,
             bundle: "ModelBundle | None" = None,
             card: pd.DataFrame | None = None) -> None:
    # 配信にもレースを担当した束の model_id を使う。svc.manifest 固定だと、
    # ばんえいの通知が平地のリリース ID で出て、重複排除キーも競合する。
    manifest = (bundle or svc.bundle_for(int(row["baba_code"]))).manifest
    key = dedupe_key(race_id, manifest.model_id)
    if already_sent(svc.wh, race_id, "prediction", key):
        return                                    # DC-07
    # 馬番だけでは「どの馬か」が通知から読めない。出馬表に載っている
    # 馬名をここで馬番に対応付ける（学習側は特徴量に馬名を使わないので、
    # res.frame には馬番しか無い）。
    horse_names = (dict(zip(card["horse_no"], card["horse_name"]))
                  if card is not None and "horse_name" in card.columns else {})
    embed = race_embed(
        race_id=race_id, track_name=str(row.get("track_name", "")),
        race_no=int(row["race_no"]), class_name=str(row.get("class_name", "")),
        distance=int(row.get("distance", 0)), start_ts=to_utc(row["start_ts"]),
        now=svc.clock.now(), model_release=manifest.model_id,
        track_used=res.track_used, candidates=res.frame, horse_names=horse_names,
        day_budget_remaining=day_budget_remaining(svc.wh, day, svc.cfg.max_bet_per_day),
        min_ev=svc.cfg.discord_min_ev)
    if embed is None:
        return                                    # 閾値未満は通知しない（DC-06）
    svc.sender.send([embed])
    record_sent(svc.wh, race_id, "prediction", key, svc.clock.now(),
                to_utc(row["start_ts"]))


def _schedule_row(wh: Warehouse, race_id: str) -> pd.Series | None:
    from .db.types import localize_utc

    df = wh.query("SELECT * FROM race_schedule WHERE race_id = ?", [race_id],
                  allow_full_scan=True)
    if df.empty:
        return None
    row = localize_utc(df).iloc[0]
    # 特徴量計算に要る既定値を埋める（当日ファイルから取れない項目）
    for k, v in (("distance", 1200), ("surface", "ダ"), ("class_level", 1),
                 ("prize_yen", 1_000_000)):
        if k not in row or pd.isna(row.get(k)):
            row[k] = v
    return row


# ----------------------------------------------------------- /weekly-report
def weekly_report_endpoint(svc: Services, day: date | None = None) -> dict:
    """モデル鮮度・カバレッジ・RF ガード。

    RF ガードが本番で発火したら**推奨の配信を自動停止**する。
    良すぎる結果を信じて資金を投じるのが最も高額な失敗（設計書 §7.2）。
    """
    day = day or business_date(svc.clock.now())
    mcfg = MonitorConfig(model_stale_days=svc.cfg.model_stale_days)
    alerts = []

    # 鮮度は系統ごとに見る。ばんえいのモデルだけ古くなっていても、平地の
    # manifest しか見ていないと誰も気づかない。
    for label, mf in (("flat", svc.manifest),
                      ("banei", svc.banei.manifest if svc.banei else None)):
        if mf is None:
            continue
        trained = date.fromisoformat(mf.train_period["end"])
        a = check_model_freshness(trained, day, mcfg)
        if a:
            a.message = f"[{label}] {a.message}"
            alerts.append(a)

    since = day - timedelta(days=30)
    stats = svc.wh.query(
        "SELECT COUNT(DISTINCT race_id) n FROM prediction WHERE race_date >= ?",
        [since], allow_full_scan=True)
    scheduled = svc.wh.query(
        "SELECT COUNT(DISTINCT race_id) n FROM race_schedule WHERE race_date >= ?",
        [since], allow_full_scan=True)
    cov = (float(stats["n"].iloc[0]) / float(scheduled["n"].iloc[0])
           if len(scheduled) and scheduled["n"].iloc[0] else 1.0)
    a = check_coverage(cov, mcfg)
    if a:
        alerts.append(a)

    # RF ガードは競技ごとに回す。ばんえいと平地を1つに混ぜると、片方が
    # 病的な数値でももう片方に薄められて発火しない。ガードは「良すぎる結果を
    # 信じて資金を投じる」ことを止めるためのもので、薄めた時点で用をなさない。
    perf_by_family = _recent_performance_by_family(svc.wh, since)
    # 返却値の top1 / nll は従来どおり合算（週次レポートの見出し数値）。
    perf = perf_by_family.get("all", {})
    for label, p in perf_by_family.items():
        if label == "all":
            continue          # 合算はガードに掛けない（競技ごとに見るのが目的）
        for a in check_rf_guards(top1_30d=p.get("top1"), nll_30d=p.get("nll"),
                                 cfg=mcfg):
            a.message = f"[{label}] {a.message}"
            alerts.append(a)

    for a in alerts:
        svc.alert(a.kind, a.message, a.severity)
        if a.blocks_delivery:
            svc.state = svc.state.blocked(f"{a.kind} 発火")

    record_run(svc.wh, "weekly_report", svc.clock, "ok", f"{len(alerts)} alerts")
    return {"status": "ok", "coverage": round(cov, 4),
            "alerts": [a.kind for a in alerts],
            "delivery_blocked": svc.state.delivery_blocked, **perf}


def _recent_performance(wh: Warehouse, since: date) -> dict:
    """本番実測（全競技を合算）。結果が未確定なら空を返す。"""
    return _recent_performance_by_family(wh, since).get("all", {})


def _recent_performance_by_family(wh: Warehouse, since: date) -> dict[str, dict]:
    """競技ごとの直近実測。`all` に合算も入れる。

    頭数分布が違う（ばんえいはほぼ 10 頭固定、平地は 5-16 頭）ので、
    Top-1 も NLL も水準が違う。合算した1つの数字で閾値を当てると、
    どちらの異常も検出できない。
    """
    df = wh.query(
        "SELECT p.race_id, p.horse_no, p.p_win, e.is_win, e.baba_code "
        "FROM prediction p JOIN entry_result_final e "
        "  ON p.race_id = e.race_id AND p.horse_no = e.horse_no "
        "WHERE p.race_date >= ? AND e.race_date >= ?", [since, since],
        allow_full_scan=True)
    if df.empty:
        return {}
    from .shared import race_nll

    def summarize(part: pd.DataFrame) -> dict:
        if part.empty:
            return {}
        top1 = float(part.loc[part.groupby("race_id")["p_win"].idxmax(),
                              "is_win"].mean())
        return {"top1": top1,
                "nll": float(race_nll(part["p_win"].to_numpy(),
                                      part["is_win"].to_numpy(),
                                      part["race_id"].to_numpy()))}

    is_banei = df["baba_code"].astype("Int64").isin(BANEI_BABA_CODES).fillna(False)
    out = {"all": summarize(df)}
    for label, part in (("flat", df[~is_banei]), ("banei", df[is_banei])):
        stats = summarize(part)
        if stats:
            out[label] = stats
    return out


# ------------------------------------------------------------------ /health
def health_endpoint(svc: Services) -> dict:
    f = freshness.check(svc.wh, svc.clock)
    return {
        "status": "ok" if f.is_fresh else "stale",
        "db_watermark": str(f.watermark) if f.watermark else None,
        "latest_final_date": str(f.latest_final_date) if f.latest_final_date else None,
        "model_release": getattr(svc.manifest, "model_id", None),
        "banei_model_release": (getattr(svc.banei.manifest, "model_id", None)
                                if svc.banei else None),
        "mode": svc.state.mode.value,
        "provenance": svc.state.provenance.value,
        "delivery_blocked": svc.state.delivery_blocked,
        "blocked_reason": svc.state.blocked_reason or None,
    }
