"""RL-01..06, DR-01..06, WB/AU: リリース運用・障害注入・Web API 契約。"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pandas as pd
import pytest

from narops.api import (
    ADMIN_FEATURES, BillingProcessor, DailyPnL, Forbidden, HorsePrediction, NotFound,
    OOSMetrics, PAID_FEATURES, Principal, RacePrediction, betting_ui_enabled,
    require_feature, require_owner,
)
from narops.clock import FixedClock, jst_datetime, to_utc
from narops.errors import (
    ArtifactIntegrityError, PromotionRejected, PublishGateFailed, StaleDataError,
)
from narops.release import (
    REQUIRED_GREEN, ShadowMetrics, ShadowRun, assert_promotion_allowed,
    assert_publish_gate, promote,
)

pytestmark = pytest.mark.component

START = jst_datetime(2026, 8, 25, 20, 35)


# ------------------------------------------------------------ RL-01（Blocker）
def test_rl01_publish_requires_all_blockers_green():
    assert_publish_gate({t: "GREEN" for t in REQUIRED_GREEN})


def test_rl01_publish_fails_on_red_blocker():
    results = {t: "GREEN" for t in REQUIRED_GREEN}
    results["LK-05"] = "RED"
    with pytest.raises(PublishGateFailed, match="LK-05"):
        assert_publish_gate(results)


def test_rl01_publish_fails_when_a_blocker_was_not_run():
    results = {t: "GREEN" for t in REQUIRED_GREEN if t != "IG-14"}
    with pytest.raises(PublishGateFailed, match="IG-14"):
        assert_publish_gate(results)


def test_rl01_required_set_covers_leak_and_market_gates():
    assert {"IG-14", "LK-05", "EV-03", "RF-01"} <= set(REQUIRED_GREEN)


# ------------------------------------------------------------ RL-02（Blocker）
def test_rl02_shadow_run_cannot_emit_bets_or_notifications():
    s = ShadowRun("v2026.09.01-A")
    s.record("R1", 1, 0.3)
    assert s.predictions and s.predictions[0]["is_shadow"] is True
    with pytest.raises(PromotionRejected, match="ベット候補"):
        s.emit_bet()
    with pytest.raises(PromotionRejected, match="Discord"):
        s.notify()
    assert s.bet_candidates == [] and s.notifications == []


def test_rl02_shadow_produces_comparison_report_only():
    s = ShadowRun("v2")
    s.record("R1", 1, 0.30)
    s.record("R1", 2, 0.20)
    prod = pd.DataFrame({"race_id": ["R1", "R1"], "horse_no": [1, 2],
                         "p_win": [0.25, 0.25]})
    rep = s.comparison_report(prod)
    assert len(rep) == 2
    assert rep["delta"].tolist() == pytest.approx([0.05, -0.05])


# ------------------------------------------------------------------ RL-03
def test_rl03_promotion_requires_two_weeks_of_shadow():
    shadow = ShadowMetrics(days=7, nll=1.80, ece=0.002, top1=0.35)
    prod = ShadowMetrics(days=30, nll=1.90, ece=0.003, top1=0.34)
    with pytest.raises(PromotionRejected, match="シャドー期間"):
        assert_promotion_allowed(shadow, prod)


def test_rl03_promotion_requires_metrics_at_least_as_good():
    prod = ShadowMetrics(days=30, nll=1.80, ece=0.002, top1=0.35)
    worse = ShadowMetrics(days=14, nll=1.95, ece=0.002, top1=0.33)
    with pytest.raises(PromotionRejected, match="NLL"):
        assert_promotion_allowed(worse, prod)

    worse_ece = ShadowMetrics(days=14, nll=1.70, ece=0.010, top1=0.36)
    with pytest.raises(PromotionRejected, match="ECE"):
        assert_promotion_allowed(worse_ece, prod)


def test_rl03_better_shadow_is_allowed():
    prod = ShadowMetrics(days=30, nll=1.90, ece=0.003, top1=0.34)
    better = ShadowMetrics(days=14, nll=1.80, ece=0.002, top1=0.36)
    assert_promotion_allowed(better, prod)


# ------------------------------------------------------------------ WB-06
def test_wb06_promotion_requires_two_step_confirmation(registry, release_dir):
    registry.publish("v2026.09.20-A", release_dir)
    prod = ShadowMetrics(days=30, nll=1.90, ece=0.003, top1=0.34)
    better = ShadowMetrics(days=14, nll=1.80, ece=0.002, top1=0.36)

    with pytest.raises(PromotionRejected, match="二段確認"):
        promote(registry, "v2026.09.20-A", better, prod, actor="alice")

    promote(registry, "v2026.09.20-A", better, prod, actor="alice", confirmed=True)
    assert registry.current_id() == "v2026.09.20-A"
    last = registry.audit_log()[-1]
    assert last["actor"] == "alice" and last["from"] == "v2026.08.24-A"


def test_wb06_promote_is_admin_only_feature():
    assert "model_promote" in ADMIN_FEATURES
    require_feature(Principal("root", "admin"), "model_promote")
    with pytest.raises(Forbidden):
        require_feature(Principal("u", "paid"), "model_promote")


# ------------------------------------------------------------ RL-04（Blocker）
def test_rl04_tampered_artifact_is_not_loaded(registry, release_dir):
    registry.publish("v2026.09.21-A", release_dir)
    target = registry.root / "releases" / "v2026.09.21-A" / "lgbm_rank.txt"
    target.write_text("evil", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="ハッシュ不一致"):
        registry.load("v2026.09.21-A")


# ------------------------------------------------------------------ RL-06
def test_rl06_service_accounts_have_exactly_the_designed_roles():
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "infra" / "services.json")
                     .read_text(encoding="utf-8"))
    # bigquery.dataEditor / dataViewer は「データへの権限」で、クエリの実行
    # そのものには bigquery.jobUser が別途要る。無いと Cloud Run から
    # 403 jobs.create になる（実際にそうなった）。最小権限のまま1つ足す。
    expected = {
        "nar-ops@sample-335613.iam.gserviceaccount.com": {
            # objectCreator は書き込み専用。配布物を GCS から読むので
            # objectViewer も要る（無いと起動時に 403 で落ちる）
            "roles/storage.objectCreator", "roles/storage.objectViewer",
            "roles/bigquery.dataEditor", "roles/bigquery.jobUser",
            # 発走時刻の変更にはタスクの作り直しが要る（enqueuer だけでは消せない）
            "roles/cloudtasks.enqueuer", "roles/cloudtasks.taskDeleter",
            "roles/secretmanager.secretAccessor",
            # Scheduler は nar-ops の OIDC トークンで Cloud Run を叩く。
            # 呼ばれる側だけでなく呼ぶ側にも invoker が要る
            "roles/run.invoker",
            # 2026-08-29 追加: nar-refresh Job が gs://nar-raw-.../ingest-store の
            # manifest.duckdb を実行のたびに上書きする。既存オブジェクトへの
            # 上書きは storage.objects.delete を要求し、objectCreator だけでは
            # 403 になった（実測）。
            "roles/storage.objectAdmin"},
        "nar-api@sample-335613.iam.gserviceaccount.com": {
            "roles/bigquery.dataViewer", "roles/bigquery.jobUser",
            # 2026-08-29 追加: /health・/performance/oos・/models が gs://nar-model/
            # の manifest を読み、/models/{id}/promote が実測ゲート通過後に
            # current.json を書き換える。書き込み対象はこのバケットのみ
            # （nar-raw・BigQuery への書き込み権限は付与していない）。
            "roles/storage.objectViewer", "roles/storage.objectCreator"},
        "nar-web@sample-335613.iam.gserviceaccount.com": {"roles/run.invoker"},
    }
    for sa, roles in expected.items():
        actual = set(cfg["service_accounts"][sa])
        assert actual == roles, f"{sa} の権限が設計と一致しません（余剰: {actual - roles}）"


def test_rl06_no_service_allows_unauthenticated():
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "infra" / "services.json")
                     .read_text(encoding="utf-8"))
    assert all(not s["allow_unauthenticated"] for s in cfg["cloud_run"].values())


def test_co05_high_cost_services_are_listed_as_denied():
    """CO-05 の逸脱を明示的に固定する。

    設計は組織ポリシーで Vertex AI / Cloud SQL を作成不可にすることを求めるが、
    このプロジェクトは共用で、既存ワークロードが両 API を有効化済み。
    プロジェクト単位で禁止すると他システムが壊れるため、**運用規約としての
    deny 一覧**に留めている。実際の防御は予算アラートで代替する。
    この逸脱を忘れて「CO-05 は green」と誤認しないよう、状態ごとテストに残す。
    """
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "infra" / "services.json")
                     .read_text(encoding="utf-8"))
    denied = set(cfg["org_policies"]["denied_services"])
    assert "aiplatform.googleapis.com" in denied and "sqladmin.googleapis.com" in denied
    assert cfg["org_policies"]["_enforcement"] == "documented-only", (
        "組織ポリシーで実強制できるようになったら、この逸脱注記を外すこと")
    assert cfg["budget_alerts_usd"] == [3, 5], "代替防御である予算アラートが未設定です"


def test_resources_are_namespaced_to_avoid_collision():
    """共用プロジェクトの既存資産と衝突しないこと。

    このプロジェクトには別システムの `hourse_racing` データセットと
    `hourse-racing-inference-sample-335613` バケットが既にある。
    名前が被ると他システムのデータを壊す。
    """
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "infra" / "services.json")
                     .read_text(encoding="utf-8"))
    assert cfg["bigquery"]["dataset"] == "nar_ops"
    assert cfg["bigquery"]["dataset"] != "hourse_racing"
    for bucket in cfg["gcs"].values():
        assert bucket.startswith("nar-"), f"{bucket} が名前空間の外です"
        assert "hourse-racing-inference" not in bucket
    assert all(s.startswith("nar-") for s in cfg["cloud_run"])
    assert all(j.startswith("nar-") for j in cfg["scheduler_jobs"])


# ------------------------------------------------------------ DR-01（Blocker）
def test_dr01_nar_outage_fails_closed(wh, clock):
    """取得失敗時は fail-closed。古いデータでの推論を 0 件にする。"""
    from narops.db import freshness

    with pytest.raises(StaleDataError):
        freshness.require_fresh(wh, clock)      # 何も入っていない = 取得失敗相当


def test_dr01_backoff_then_give_up():
    from narops.jobs import RetryPolicy

    p = RetryPolicy()
    waits = [p.wait_for(i) for i in (1, 2, 3)]
    assert waits == [30, 90, 270], "指数バックオフになっていません"
    assert not p.should_retry(TimeoutError(), attempt=3)


# ------------------------------------------------------------------ DR-02
def test_dr02_schedule_extraction_failure_aborts_the_day(queue_factory=None):
    """HTML 構造変化でスケジュール抽出に失敗したら、部分結果で走らない。"""
    from narops.scheduling import plan_day
    from narops.tasks import TaskQueue

    q = TaskQueue()
    broken = pd.DataFrame(columns=["race_id", "race_date", "baba_code", "race_no",
                                   "start_ts", "status"])
    actions = plan_day(broken, q, FixedClock(START), __import__(
        "narops.config", fromlist=["OpsConfig"]).OpsConfig.load())
    assert actions == [] and q.count() == 0


# ------------------------------------------------------------------ DR-03
def test_dr03_schema_drift_stops_promotion_to_silver():
    from narops.monitoring import check_schema_drift, should_block_delivery

    alert = check_schema_drift(["entry: 37列（期待 36）"])
    assert alert.severity == "Blocker" and should_block_delivery([alert])


# ------------------------------------------------------------------ DR-04
def test_dr04_bq_failure_yields_no_bet_candidates_and_marks_uncomputed(wh, clock):
    from narops.pipeline import InferenceOutcome

    outcome = InferenceOutcome.failed("R1", "BigQuery 一時障害", retryable=True)
    assert outcome.bet_candidates.empty
    assert outcome.status == "not_computed"
    assert outcome.web_label == "未算出"


# ------------------------------------------------------------------ DR-06
def test_dr06_concurrent_inference_converges_to_single_version(wh, clock):
    """同一レースへの並行推論で書き込み競合が起きず、最終行が単一版に収束する。"""
    from narops.pipeline import write_prediction

    frame = pd.DataFrame({"horse_no": [1, 2], "p_win": [0.6, 0.4],
                          "p_market": [0.5, 0.5], "ev": [1.1, 0.9],
                          "ev_adjusted": [1.05, 0.9], "stake_yen": [100, 0]})
    for _ in range(3):          # 同一タスクの並行/再実行を模す
        write_prediction(wh, "202026082511", frame, "v1", "A+B",
                         pd.Timestamp("2026-08-25").date(), clock)

    df = wh.query("SELECT * FROM prediction WHERE race_date = '2026-08-25'",
                  max_bytes_billed=10**9)
    assert len(df) == 2, f"重複行が残っています（{len(df)} 行）"
    assert set(df["model_release"]) == {"v1"}


# ------------------------------------------------------------------ WB-01
def test_wb01_race_prediction_schema_enforces_probability_sum():
    horses = [HorsePrediction(horse_no=1, p_win=0.6, stake_yen=0),
              HorsePrediction(horse_no=2, p_win=0.4, stake_yen=0)]
    RacePrediction(race_id="202026082511", race_date="2026-08-25",
                   start_ts=to_utc(START), track_name="大井", race_no=11,
                   model_release="v1", track_used="A+B", status="ok", horses=horses)

    bad = [HorsePrediction(horse_no=1, p_win=0.6, stake_yen=0),
           HorsePrediction(horse_no=2, p_win=0.1, stake_yen=0)]
    with pytest.raises(ValueError, match="総和"):
        RacePrediction(race_id="202026082511", race_date="2026-08-25",
                       start_ts=to_utc(START), track_name="大井", race_no=11,
                       model_release="v1", track_used="A+B", status="ok", horses=bad)


def test_wb01_invalid_race_id_is_rejected():
    with pytest.raises(ValueError):
        RacePrediction(race_id="abc", race_date="2026-08-25", start_ts=to_utc(START),
                       track_name="大井", race_no=11, model_release="v1",
                       track_used="A", status="ok")


def test_wb01_track_used_is_an_enum():
    with pytest.raises(ValueError):
        RacePrediction(race_id="202026082511", race_date="2026-08-25",
                       start_ts=to_utc(START), track_name="大井", race_no=11,
                       model_release="v1", track_used="C", status="ok")


# ------------------------------------------------------------------ WB-03
def test_wb03_oos_metrics_are_marked_as_stored():
    m = OOSMetrics(model="tabm", track="A", race_nll=1.90, top1=0.34, top3=0.57,
                   brier=0.80, ece=0.0025, evaluated_at=to_utc(START))
    assert m.source == "stored", "オンザフライ再計算を許してはいけない"


# ------------------------------------------------------------ WB-04（Blocker）
def test_wb04_pnl_distinguishes_paper_from_live():
    paper = DailyPnL(business_date="2026-08-24", n_races=40, n_bets=10,
                     stake_yen=1000, return_yen=900, roi=0.9, hit_rate=0.1,
                     is_paper=True)
    assert paper.is_paper is True
    with pytest.raises(ValueError):
        DailyPnL(business_date="2026-08-24", n_races=40, n_bets=10,
                 stake_yen=1000, return_yen=900, roi=0.9, hit_rate=0.1)


# ------------------------------------------------------------------ WB-05
def test_wb05_betting_ui_disabled_after_start():
    assert betting_ui_enabled(to_utc(START), to_utc(START) - timedelta(minutes=1))
    assert not betting_ui_enabled(to_utc(START), to_utc(START) + timedelta(seconds=1))


# ------------------------------------------------------------ WB-07（Blocker）
def test_wb07_idor_is_blocked():
    require_owner(Principal("alice"), "alice")
    with pytest.raises(NotFound):
        require_owner(Principal("bob"), "alice")


def test_wb07_admin_can_access_any_resource():
    require_owner(Principal("root", "admin"), "alice")


# ------------------------------------------------------------ AU-01（Blocker）
def test_au01_paid_features_blocked_at_api_level():
    free = Principal("u", "free")
    for feature in PAID_FEATURES:
        with pytest.raises(Forbidden):
            require_feature(free, feature)
    paid = Principal("u", "paid")
    for feature in PAID_FEATURES:
        require_feature(paid, feature)


def test_au01_free_features_are_open():
    require_feature(Principal("u", "free"), "public_dashboard")


# ------------------------------------------------------------ AU-02（Blocker）
def test_au02_billing_webhook_verifies_signature():
    p = BillingProcessor(secret="s3cret")
    payload = b'{"id":"evt_1"}'
    sig = p.verify.__self__  # noqa: F841 - 署名生成はテスト側で行う
    import hmac

    good = hmac.new(b"s3cret", payload, hashlib.sha256).hexdigest()
    assert p.handle("evt_1", "u1", "subscription.created", payload, good) == "upgraded"
    assert p.plan_of("u1") == "paid"

    with pytest.raises(Forbidden, match="署名"):
        p.handle("evt_2", "u1", "subscription.created", payload, "deadbeef")


def test_au02_duplicate_event_is_idempotent():
    import hmac

    p = BillingProcessor(secret="s")
    payload = b'{"id":"evt_1"}'
    sig = hmac.new(b"s", payload, hashlib.sha256).hexdigest()
    assert p.handle("evt_1", "u1", "subscription.created", payload, sig) == "upgraded"
    assert p.handle("evt_1", "u1", "subscription.created", payload, sig) == "duplicate"
    assert p.plan_of("u1") == "paid"


def test_au02_cancellation_downgrades_immediately():
    import hmac

    p = BillingProcessor(secret="s")
    payload = b"{}"
    sig = hmac.new(b"s", payload, hashlib.sha256).hexdigest()
    p.handle("e1", "u1", "subscription.created", payload, sig)
    assert p.plan_of("u1") == "paid"
    p.handle("e2", "u1", "subscription.canceled", payload, sig)
    assert p.plan_of("u1") == "free", "失効時に即時ダウングレードされていません"


# ------------------------------------------------------------------ 初回導入
def test_bootstrap_only_works_when_there_is_no_current(tmp_path, release_dir):
    """初回導入の経路が、以後の入れ替えの抜け道にならないこと。

    `promote` は「現行モデルより良いこと」を条件にしている。比較対象が無い初回に
    その判定は使えないが、条件を緩めると以後の入れ替えでも同じ穴が開く。
    """
    from narops.model.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "empty-model-root")
    reg.publish("v-first", release_dir)
    assert reg.current_id() is None
    reg.set_current("v-first", actor="tester", reason="初回導入")
    assert reg.current_id() == "v-first"


def test_promotion_still_required_once_a_current_exists(registry, release_dir):
    """current がある状態では、シャドー実績なしに入れ替えられないこと。"""
    import pytest

    from narops.errors import PromotionRejected
    from narops.release import ShadowMetrics, promote

    before = registry.current_id()
    registry.publish("v-second", release_dir)

    with pytest.raises(PromotionRejected, match="シャドー期間"):
        promote(registry, "v-second",
                shadow=ShadowMetrics(days=3, nll=1.0, ece=0.001, top1=0.35),
                production=ShadowMetrics(days=30, nll=1.1, ece=0.002, top1=0.34),
                actor="tester", confirmed=True)
    assert registry.current_id() == before


def test_initial_pointer_can_be_corrected_only_before_any_inference(tmp_path, release_dir):
    """初回設定の訂正は「まだ使われていない」ときだけ。

    判定はポインタの有無ではなく推論の実績で行う。ポインタの有無で判定すると、
    運用中のモデルを黙って差し替える穴になる。
    """
    from narops.db import schema
    from narops.db.backend import Warehouse
    from narops.model.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "root")
    reg.publish("v-a", release_dir)
    reg.publish("v-b", release_dir)
    reg.set_current("v-a", actor="t", reason="初回導入")

    wh = Warehouse(":memory:", max_bytes_billed=10_000_000)
    schema.create_all(wh)
    from datetime import date

    n = wh.query("SELECT COUNT(*) AS n FROM prediction "
                 "WHERE race_date >= ? AND model_release = ?",
                 [date(2020, 1, 1), "v-a"], allow_full_scan=True)["n"].iloc[0]
    assert int(n) == 0, "テスト前提: まだ推論していない"

    reg.set_current("v-b", actor="t", reason="初回設定の訂正")
    assert reg.current_id() == "v-b"
    wh.close()
