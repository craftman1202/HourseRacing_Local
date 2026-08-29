"""nar-api の実データ接続（narops.api_app）のテスト。

`narops.api` 自体の契約テスト（WB-01 等）は test_release_dr.py にある。
ここでは `api_app.create_app()` が本物の Warehouse/ModelRegistry から正しい
値を組み立てて返すこと、および `/models/{id}/promote` の実ゲートが機能する
ことを確認する。
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from narops.api_app import create_app
from narops.clock import jst_datetime, to_utc

RACE_ID = "202026082501"
RACE_DATE = date(2026, 8, 25)


def _client(wh, registry, cfg, clock) -> TestClient:
    return TestClient(create_app(wh=wh, registry=registry, cfg=cfg, clock=clock))


def _seed_schedule(wh) -> None:
    wh.insert_frame("race_schedule", pd.DataFrame([{
        "race_id": RACE_ID, "race_date": RACE_DATE, "baba_code": 20, "race_no": 1,
        "start_ts": to_utc(jst_datetime(2026, 8, 25, 15, 30)), "status": "scheduled",
        "updated_at": to_utc(jst_datetime(2026, 8, 25, 8, 0)),
    }]))


def _seed_prediction(wh, *, model_release="v2026.08.24-A", is_shadow=False) -> None:
    wh.insert_frame("prediction", pd.DataFrame([
        {"race_id": RACE_ID, "horse_no": 1, "race_date": RACE_DATE,
         "model_release": model_release, "track_used": "A",
         "p_win": 0.6, "p_market": 0.5, "ev": 1.2, "ev_adjusted": 1.1,
         "computed_at": to_utc(jst_datetime(2026, 8, 25, 15, 0)), "is_shadow": is_shadow},
        {"race_id": RACE_ID, "horse_no": 2, "race_date": RACE_DATE,
         "model_release": model_release, "track_used": "A",
         "p_win": 0.4, "p_market": 0.5, "ev": None, "ev_adjusted": None,
         "computed_at": to_utc(jst_datetime(2026, 8, 25, 15, 0)), "is_shadow": is_shadow},
    ]))


def _seed_bet_candidate(wh) -> None:
    wh.insert_frame("bet_candidate", pd.DataFrame([{
        "race_id": RACE_ID, "horse_no": 1, "race_date": RACE_DATE,
        "model_release": "v2026.08.24-A", "bet_type": "単勝",
        "stake_yen": 300, "ev": 1.2, "ev_adjusted": 1.1, "kelly": 0.05,
        "computed_at": to_utc(jst_datetime(2026, 8, 25, 15, 0)),
    }]))


# --------------------------------------------------------------------- health
def test_health_reports_current_release_and_skew(wh, registry, cfg, clock):
    wh.insert_frame("skew_check", pd.DataFrame([{
        "business_date": RACE_DATE, "checked_at": to_utc(jst_datetime(2026, 8, 25, 4, 0)),
        "n_compared": 100, "n_mismatch": 0, "mismatch_columns": "", "verdict": "PASS",
    }]))
    r = _client(wh, registry, cfg, clock).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_release"] == "v2026.08.24-A"
    assert body["model_trained_on"] == "2023-12-31"
    assert body["skew_verdict"] == "PASS"
    assert body["delivery_blocked"] is False


def test_health_defaults_skew_to_unknown_without_rows(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).get("/health")
    assert r.json()["skew_verdict"] == "UNKNOWN"


# ----------------------------------------------------------------------- race
def test_race_not_found_returns_404(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).get(f"/races/{RACE_ID}")
    assert r.status_code == 404


def test_race_returns_prediction_sorted_by_p_win_with_stake(wh, registry, cfg, clock):
    _seed_schedule(wh)
    _seed_prediction(wh)
    _seed_bet_candidate(wh)
    r = _client(wh, registry, cfg, clock).get(f"/races/{RACE_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["race_id"] == RACE_ID
    assert body["race_no"] == 1
    assert body["model_release"] == "v2026.08.24-A"
    horses = body["horses"]
    assert [h["horse_no"] for h in horses] == [1, 2]        # p_win 降順
    assert horses[0]["stake_yen"] == 300
    assert horses[1]["stake_yen"] == 0                       # bet_candidate 無し
    assert horses[1]["ev"] is None                           # NaN -> null


def test_race_excludes_shadow_predictions(wh, registry, cfg, clock):
    _seed_schedule(wh)
    _seed_prediction(wh, is_shadow=True)
    r = _client(wh, registry, cfg, clock).get(f"/races/{RACE_ID}")
    assert r.status_code == 404


# ------------------------------------------------------------------------ oos
def test_oos_reflects_manifest_with_missing_metrics_as_null(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).get("/performance/oos")
    assert r.status_code == 200
    [row] = r.json()
    assert row["model"] == "ensemble"
    assert row["track"] == "A"
    assert row["race_nll"] == pytest.approx(1.90)
    assert row["top1"] == pytest.approx(0.34)
    assert row["ece"] == pytest.approx(0.003)
    assert row["top3"] is None    # manifest には無い -> NaN -> JSON null
    assert row["brier"] is None
    assert row["source"] == "stored"


# ----------------------------------------------------------------------- live
def test_live_reflects_pnl_daily_rows_desc(wh, registry, cfg, clock):
    wh.insert_frame("pnl_daily", pd.DataFrame([
        {"business_date": date(2026, 8, 24), "n_races": 10, "n_bets": 3,
         "stake_yen": 900, "return_yen": 1200, "roi": 1.33, "hit_rate": 0.33,
         "brier": 0.2, "coverage": 1.0, "is_paper": True,
         "recomputed_at": to_utc(jst_datetime(2026, 8, 25, 4, 0))},
        {"business_date": date(2026, 8, 23), "n_races": 8, "n_bets": 2,
         "stake_yen": 600, "return_yen": 0, "roi": 0.0, "hit_rate": 0.0,
         "brier": 0.3, "coverage": 1.0, "is_paper": True,
         "recomputed_at": to_utc(jst_datetime(2026, 8, 24, 4, 0))},
    ]))
    r = _client(wh, registry, cfg, clock).get("/performance/live")
    assert r.status_code == 200
    body = r.json()
    assert [row["business_date"] for row in body] == ["2026-08-24", "2026-08-23"]
    assert body[0]["is_paper"] is True


# --------------------------------------------------------------------- models
def test_models_lists_current_release(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).get("/models")
    assert r.status_code == 200
    [row] = r.json()
    assert row["release_id"] == "v2026.08.24-A"
    assert row["is_current"] is True
    assert row["dataset_version"] == "ds-" + "0" * 8


# ------------------------------------------------------------------- promote
def test_promote_rejects_when_no_shadow_data_exists(wh, registry, cfg, clock):
    """本物のゲート（release.assert_promotion_allowed）が実際に評価されること。

    実測ゼロ件 -> 日数0 < SHADOW_DAYS で確実に拒否される。フラグを人間が
    手入力する CLI 経路とは違い、Web からは実測でしか通せない。
    """
    r = _client(wh, registry, cfg, clock).post(
        "/models/does-not-exist/promote", json={"confirmed": True})
    assert r.status_code == 409
    assert "シャドー期間" in r.json()["detail"]


def test_promote_requires_confirmed_flag(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).post(
        "/models/does-not-exist/promote", json={"confirmed": False})
    assert r.status_code == 409
    assert "二段確認" in r.json()["detail"]


# ------------------------------------------------------------------ races_today
def test_races_today_empty_when_nothing_scheduled(wh, registry, cfg, clock):
    r = _client(wh, registry, cfg, clock).get("/races/today")
    assert r.status_code == 200
    assert r.json() == []


def test_races_today_marks_computed_vs_not_computed(wh, registry, cfg, clock):
    _seed_schedule(wh)                 # 2026-08-25 15:30 JST 発走、clock は同日 10:00 JST
    r = _client(wh, registry, cfg, clock).get("/races/today")
    assert r.status_code == 200
    [row] = r.json()
    assert row["race_id"] == RACE_ID
    assert row["status"] == "not_computed"     # まだ発走前で推論も無い
    assert row["top_ev_adjusted"] is None

    _seed_prediction(wh)
    r = _client(wh, registry, cfg, clock).get("/races/today")
    [row] = r.json()
    assert row["status"] == "ok"
    assert row["top_ev_adjusted"] == pytest.approx(1.1)


def test_races_today_route_is_not_shadowed_by_race_id_route(wh, registry, cfg, clock):
    """`/races/today` が `/races/{race_id}` に食われず正しくマッチすること。"""
    r = _client(wh, registry, cfg, clock).get("/races/today")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
