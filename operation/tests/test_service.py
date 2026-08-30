"""運用エンドポイントと段階導入モード。

設計書 §9 の段階計画（shadow → paper → live）がコードで強制されていることを固定する。
「気づいたら本番で賭けていた」を作らないのが目的。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, business_date, jst_datetime, to_utc
from narops.mode import Mode, ModelProvenance, OperatingState
from narops.model.manifest import Manifest
from narops.service import (
    Services, health_endpoint, infer_endpoint, plan_day_endpoint,
    refresh_live_endpoint, weekly_report_endpoint,
)
from tests.conftest import make_results

pytestmark = pytest.mark.component

DAY = date(2026, 8, 25)
START = jst_datetime(2026, 8, 25, 20, 35)
RACE_ID = "202026082511"


class StubSource:
    """NAR の代替。録画応答を返す（テスト仕様 §1.3）。"""

    schedule_rows: list[dict] = []
    card: pd.DataFrame | None = None
    odds: pd.DataFrame | None = None
    fail: Exception | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def fetch_schedule(self, day):
        if self.fail:
            raise self.fail
        return pd.DataFrame(self.schedule_rows)

    def fetch_entry_card(self, race_id, baba_code, day, race_no):
        return self.card if self.card is not None else pd.DataFrame()

    def fetch_odds(self, race_id, baba_code, day, race_no):
        return self.odds if self.odds is not None else pd.DataFrame(
            columns=["race_id", "horse_no", "odds_win"])


class ConstScorer:
    def __init__(self, name, seed=0):
        self.name = name
        self.rng = np.random.default_rng(seed)

    def score(self, frame):
        return self.rng.normal(size=len(frame))


@pytest.fixture
def schedule_rows():
    return [{
        "race_id": RACE_ID, "race_date": DAY, "baba_code": 20, "race_no": 11,
        "track_name": "大井", "start_ts": to_utc(START), "status": "scheduled",
    }]


@pytest.fixture
def svc(wh, cfg, release_dir, schedule_rows):
    from nar.config import feature_config
    from narops.db.merge import merge_final

    clock = FixedClock(START - timedelta(minutes=13))
    # 鮮度ゲート（DB-04）は確定層の最新確定日が前日以上であることを要求する。
    # make_results は 4 レースで1日進むので、DAY-1 = 8/24 に届くには 28 レース要る。
    # 20 レースだと 8/22 止まりで、推論系のテストが全部ゲートで弾かれる。
    merge_final(wh, make_results(n_races=28, start_day=18, seed=2), clock=clock)

    src = StubSource()
    src.schedule_rows = schedule_rows
    src.card = pd.DataFrame([{
        "race_id": RACE_ID, "horse_no": i + 1, "horse_sk": f"H{i:04d}",
        "jockey_sk": f"J{i % 12:03d}", "trainer_sk": f"T{i % 9:03d}",
        "sire_sk": f"S{i % 6:03d}"} for i in range(8)])
    src.odds = pd.DataFrame({"race_id": RACE_ID, "horse_no": range(1, 9),
                             "odds_win": [3.2, 5.1, 8.0, 12.5, 4.4, 22.0, 60.0, 15.0]})

    manifest = Manifest.read(release_dir / "manifest.json")
    manifest.ensemble_weights = {"a": 0.5, "b": 0.5}
    return Services(
        wh=wh, cfg=cfg, clock=clock,
        state=OperatingState(Mode.SHADOW, ModelProvenance.SYNTHETIC),
        source_factory=lambda: src, manifest=manifest,
        models={"a": ConstScorer("a", 1), "b": ConstScorer("b", 2)},
        feature_config=feature_config())


# ------------------------------------------------------------------ モード
def test_shadow_is_the_default():
    assert OperatingState.from_env().mode is Mode.SHADOW


def test_shadow_emits_neither_bets_nor_delivery():
    s = OperatingState(Mode.SHADOW, ModelProvenance.SYNTHETIC)
    assert not s.can_emit_bets() and not s.can_deliver()


def test_paper_delivers_but_stays_virtual():
    s = OperatingState(Mode.PAPER, ModelProvenance.REAL)
    assert s.can_deliver() and s.can_emit_bets() and s.mode.is_paper


def test_live_requires_a_model_trained_on_real_data():
    """合成データ学習のモデルで live に上げようとしたら起動を止める。"""
    with pytest.raises(RuntimeError, match="合成データ"):
        OperatingState(Mode.LIVE, ModelProvenance.SYNTHETIC).validate()
    OperatingState(Mode.LIVE, ModelProvenance.REAL).validate()


def test_blocked_state_overrides_mode():
    s = OperatingState(Mode.PAPER, ModelProvenance.REAL).blocked("skew 不一致")
    assert not s.can_deliver() and not s.can_emit_bets()
    assert "skew" in s.describe()


# ------------------------------------------------------------------ /plan-day
def test_plan_day_stores_schedule_and_enqueues(svc, schedule_rows):
    out = plan_day_endpoint(svc, DAY)
    assert out["status"] == "ok" and out["races"] == 1
    assert out["infer_tasks"] == 1
    stored = svc.wh.query("SELECT * FROM race_schedule WHERE race_date = ?", [DAY],
                          allow_full_scan=True)
    assert len(stored) == 1 and stored["race_id"].iloc[0] == RACE_ID


def test_plan_day_succeeds_even_if_the_task_count_readback_fails(svc, schedule_rows):
    """タスク件数の読み戻し失敗が、積み込み成功そのものを 500 に変えないこと。

    実運用（2026-08-29）で、Cloud Tasks の list に必要な IAM 権限
    （enqueue とは別）が無く、スケジュール保存とタスク積み込みは成功して
    いるのに `/plan-day` が 500 を返していた。読み戻しは本体の成否と別に
    扱う。
    """
    class _BrokenCount:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def count(self, endpoint=None):
            raise PermissionError("cloudtasks.tasks.list が許可されていません")

    svc.queue = _BrokenCount(svc.queue)
    out = plan_day_endpoint(svc, DAY)
    assert out["status"] == "ok" and out["races"] == 1
    assert out["infer_tasks"] is None, "読み戻し失敗時は不明として返すべき"
    stored = svc.wh.query("SELECT * FROM race_schedule WHERE race_date = ?", [DAY],
                          allow_full_scan=True)
    assert len(stored) == 1, "読み戻し失敗の前に完了した積み込みは残っているはず"


def test_plan_day_is_once_per_day(svc):
    assert plan_day_endpoint(svc, DAY)["status"] == "ok"
    assert plan_day_endpoint(svc, DAY)["status"] == "noop"


def test_plan_day_aborts_on_fetch_failure(svc):
    """DR-02: HTML 構造変化を検知したら当日を中止し、部分結果で走らない。"""
    from narops.nar_source import NarFetchError

    src = svc.source_factory()
    src.fail = NarFetchError("スケジュールを抽出できませんでした")
    out = plan_day_endpoint(svc, DAY)
    assert out["status"] == "aborted" and out["races"] == 0
    assert svc.queue.count() == 0, "中止したのにタスクを積んでいます"


# -------------------------------------------------------------------- /infer
def test_infer_in_shadow_writes_predictions_but_no_bets(svc):
    plan_day_endpoint(svc, DAY)
    out = infer_endpoint(svc, RACE_ID)

    assert out.status == "ok" and len(out.predictions) == 8
    assert out.bet_candidates.empty, "shadow でベット候補を出しています"
    assert svc.wh.row_count("bet_candidate") == 0
    preds = svc.wh.query("SELECT * FROM prediction WHERE race_date = ?", [DAY],
                         allow_full_scan=True)
    assert len(preds) == 8 and bool(preds["is_shadow"].all())


def test_infer_saves_feature_snapshot(svc):
    """SK-06: 推論に使った特徴量が必ず残る。"""
    plan_day_endpoint(svc, DAY)
    infer_endpoint(svc, RACE_ID)
    assert svc.wh.row_count("feature_snapshot") == 8


def test_infer_probabilities_sum_to_one(svc):
    plan_day_endpoint(svc, DAY)
    out = infer_endpoint(svc, RACE_ID)
    assert out.predictions["p_win"].sum() == pytest.approx(1.0, abs=1e-9)


def test_infer_after_start_is_expired(svc):
    plan_day_endpoint(svc, DAY)
    svc.clock.tick(20 * 60)
    out = infer_endpoint(svc, RACE_ID)
    assert out.status == "expired" and out.bet_candidates.empty


def test_infer_unknown_race_is_not_computed(svc):
    out = infer_endpoint(svc, "209912319999")
    assert out.status == "not_computed" and out.web_label == "未算出"


def test_infer_degrades_to_track_a_without_odds(svc):
    plan_day_endpoint(svc, DAY)
    src = svc.source_factory()
    src.odds = pd.DataFrame(columns=["race_id", "horse_no", "odds_win"])
    out = infer_endpoint(svc, RACE_ID)
    assert out.status == "ok"
    preds = svc.wh.query("SELECT * FROM prediction WHERE race_date = ?", [DAY],
                         allow_full_scan=True)
    assert set(preds["track_used"]) == {"A"}


def test_infer_stops_when_final_layer_is_stale(wh, cfg, release_dir, schedule_rows):
    """DB-04: 鮮度ゲートで落ちたら推論しない。"""
    from nar.config import feature_config

    clock = FixedClock(START - timedelta(minutes=13))
    src = StubSource()
    src.schedule_rows = schedule_rows
    svc = Services(wh=wh, cfg=cfg, clock=clock, source_factory=lambda: src,
                   manifest=Manifest.read(release_dir / "manifest.json"),
                   feature_config=feature_config())
    plan_day_endpoint(svc, DAY)
    out = infer_endpoint(svc, RACE_ID)      # 確定層が空
    assert out.status == "not_computed"
    assert wh.row_count("bet_candidate") == 0


def test_paper_mode_emits_bet_candidates(svc):
    svc.state = OperatingState(Mode.PAPER, ModelProvenance.SYNTHETIC)
    plan_day_endpoint(svc, DAY)
    infer_endpoint(svc, RACE_ID)
    preds = svc.wh.query("SELECT * FROM prediction WHERE race_date = ?", [DAY],
                         allow_full_scan=True)
    assert not bool(preds["is_shadow"].any()), "paper なのに shadow 扱いです"


def test_blocked_delivery_suppresses_bets_even_in_paper(svc):
    svc.state = OperatingState(Mode.PAPER, ModelProvenance.SYNTHETIC).blocked("skew")
    plan_day_endpoint(svc, DAY)
    infer_endpoint(svc, RACE_ID)
    assert svc.wh.row_count("bet_candidate") == 0


# ------------------------------------------------------------- /refresh-live
def test_refresh_live_is_capped_at_three_per_day(svc):
    rows = make_results(n_races=2, start_day=25, seed=9)
    for _ in range(3):
        assert refresh_live_endpoint(svc, rows, DAY)["status"] == "ok"
    assert refresh_live_endpoint(svc, rows, DAY)["status"] == "noop"


def test_refresh_live_does_not_touch_final_layer(svc):
    from narops.db.merge import table_hash

    before = table_hash(svc.wh, "entry_result_final")
    refresh_live_endpoint(svc, make_results(n_races=2, start_day=25, seed=9), DAY)
    assert table_hash(svc.wh, "entry_result_final") == before


# ----------------------------------------------------------- /weekly-report
def test_weekly_report_flags_stale_model(svc):
    svc.manifest.train_period = {"start": "2010-01-01", "end": "2020-01-01"}
    out = weekly_report_endpoint(svc, DAY)
    assert "model_stale" in out["alerts"]


def test_weekly_report_is_quiet_on_fresh_model(svc):
    svc.manifest.train_period = {"start": "2010-01-01", "end": str(DAY)}
    plan_day_endpoint(svc, DAY)
    infer_endpoint(svc, RACE_ID)
    out = weekly_report_endpoint(svc, DAY)
    assert "model_stale" not in out["alerts"]


# ------------------------------------------------------------------ /health
def test_health_reports_mode_and_freshness(svc):
    out = health_endpoint(svc)
    assert out["mode"] == "shadow" and out["provenance"] == "synthetic"
    assert out["delivery_blocked"] is False
    assert out["model_release"] == svc.manifest.model_id


def test_health_marks_stale_db(wh, cfg, release_dir):
    svc = Services(wh=wh, cfg=cfg, clock=FixedClock(START))
    assert health_endpoint(svc)["status"] == "stale"


def test_plan_day_accepts_an_explicit_date(svc):
    """日付を渡せないと、翌日ぶんの事前確認も取りこぼしのやり直しもできない。

    1日1回の制約は日付ごとに効くので、指定しても二重には走らない。
    """
    from datetime import timedelta

    from narops.service import plan_day_endpoint

    other = DAY + timedelta(days=1)
    assert plan_day_endpoint(svc, DAY)["status"] == "ok"
    assert plan_day_endpoint(svc, DAY)["status"] == "noop"
    # 別の日は別枠として実行できる
    assert plan_day_endpoint(svc, other)["status"] in ("ok", "aborted")


def test_one_unavailable_odds_page_does_not_stop_reconciliation(svc, schedule_rows):
    """オッズ未公開のレースは常にある。

    そこで例外にすると**スケジュール再照合まで巻き添えで止まり**、
    発走時刻の変更に追随できなくなる。
    """
    from narops.nar_source import NarFetchError
    from narops.service import snapshot_odds_endpoint

    src = svc.source_factory()
    src.schedule_rows = schedule_rows

    def boom(*a, **k):
        raise NarFetchError("NAR がエラーページを返しました")

    src.fetch_odds = boom
    out = snapshot_odds_endpoint(svc, DAY)
    assert out["status"] == "ok", f"1レースの失敗で全体が止まりました: {out}"
    assert out["odds_unavailable"] >= 0


def test_refresh_live_actually_fetches_results(svc, schedule_rows, monkeypatch):
    """`/refresh-live` は本番で常に 0 件を返していた。

    HTTP 経路が結果を渡さず、エンドポイント側にも取得処理が無かった。
    ライブ層が一度も埋まらず、当日の先行レースが as-of 履歴から抜けていた。
    """
    from narops.service import _upsert_schedule, refresh_live_endpoint

    src = svc.source_factory()
    src.schedule_rows = schedule_rows
    _upsert_schedule(svc.wh, pd.DataFrame(schedule_rows), svc.clock)

    calls = []

    def fake_results(race_id, baba_code, day, race_no):
        calls.append(race_id)
        return pd.DataFrame({
            "race_id": [race_id], "horse_no": [1], "horse_sk": ["h1"],
            "jockey_sk": ["j1"], "trainer_sk": ["t1"], "sire_sk": ["s1"],
            "finish_pos": [1], "is_win": [1], "time_sec": [77.9],
            "race_date": [DAY], "baba_code": [baba_code], "distance": [1500],
        })

    src.fetch_results = fake_results
    svc.clock.current = to_utc(jst_datetime(DAY.year, DAY.month, DAY.day, 23, 0))

    out = refresh_live_endpoint(svc)
    assert calls, "成績を一度も取りに行っていません"
    assert out["rows"] > 0, out
