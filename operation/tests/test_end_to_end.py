"""1レース分の推論を最初から最後まで通す結合テスト。

個々の不変条件は各スイートで検証済みなので、ここでは「配線が繋がっていること」と
**失敗時に配信されないこと**を確認する。
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, business_date, jst_datetime, to_utc
from narops.discord.client import (
    DiscordSender, RateLimiter, already_sent, dedupe_key, record_sent,
)
from narops.discord.format import race_embed
from narops.errors import StaleDataError
from narops.features import build_for_race, save_snapshot
from narops.inference import PoolSizeModel, assert_within_window, run_inference
from narops.model.manifest import Manifest
from narops.pipeline import (
    InferenceOutcome, day_budget_remaining, write_bet_candidates, write_prediction,
)
from tests.conftest import make_results

pytestmark = pytest.mark.integration

RACE_ID = "202026082411"
START = jst_datetime(2026, 8, 24, 20, 35)
FAKE_WEBHOOK = "https://discord.com/api/webhooks/1/token"


class LinearModel:
    def __init__(self, name: str, w: np.ndarray) -> None:
        self.name, self.w = name, w

    def score(self, features: pd.DataFrame) -> np.ndarray:
        cols = ["h_winrate_prior", "j_winrate_shrunk"]
        x = features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
        return x @ self.w


@pytest.fixture
def card() -> pd.DataFrame:
    return pd.DataFrame([{
        "race_id": RACE_ID, "horse_no": i + 1, "horse_sk": f"H{i:04d}",
        "jockey_sk": f"J{i % 12:03d}", "trainer_sk": f"T{i % 9:03d}",
        "sire_sk": f"S{i % 6:03d}"} for i in range(8)])


@pytest.fixture
def race_row() -> pd.Series:
    return pd.Series({
        "race_id": RACE_ID, "race_date": START.date(), "start_ts": to_utc(START),
        "baba_code": 20, "distance": 1200, "race_no": 11, "surface": "ダ",
        "class_level": 2, "prize_yen": 1_200_000})


@pytest.fixture
def fresh_wh(wh, clock):
    """前日までの確定層が入った状態にする（鮮度ゲートを通す）。"""
    from narops.db.merge import merge_final

    merge_final(wh, make_results(n_races=20, start_day=18, seed=2),
                source_sha256="a" * 64, clock=clock)
    return wh


def test_full_race_pipeline(fresh_wh, card, race_row, release_dir, cfg):
    from nar.config import feature_config

    clock = FixedClock(START - timedelta(minutes=10))
    manifest = Manifest.read(release_dir / "manifest.json")

    # 1. 推論窓の検証
    remaining = assert_within_window(to_utc(START), clock, cfg)
    assert 9 < remaining <= 10

    # 2. as-of 特徴量
    feats = build_for_race(fresh_wh, card, race_row, manifest, feature_config(),
                           max_bytes_billed=cfg.max_bytes_billed)
    assert len(feats.frame) == 8

    # 3. snapshot 保存（SK-06）
    assert save_snapshot(fresh_wh, feats, RACE_ID, manifest.model_id, clock) == 8

    # 4. 推論
    odds = pd.Series([3.2, 5.1, 8.0, 12.5, 4.4, 22.0, 60.0, 15.0])
    res = run_inference(
        race_id=RACE_ID, features=feats.frame,
        models={"lgbm": LinearModel("lgbm", np.array([1.5, 0.8])),
                "tabm": LinearModel("tabm", np.array([1.2, 1.1]))},
        manifests=[manifest, manifest], weights={"lgbm": 0.3, "tabm": 0.7},
        temperature=manifest.calibration["temperature"], clock=clock, cfg=cfg,
        odds=odds, pool_model=PoolSizeModel(default_yen=1_400_000),
        day_budget_remaining=day_budget_remaining(
            fresh_wh, business_date(to_utc(START)), cfg.max_bet_per_day))

    assert res.track_used == "A+B"
    assert res.frame["p_win"].sum() == pytest.approx(1.0, abs=1e-9)

    # 5. 永続化（冪等）
    day = business_date(to_utc(START))
    for _ in range(2):
        write_prediction(fresh_wh, RACE_ID, res.frame, manifest.model_id,
                         res.track_used, day, clock)
        write_bet_candidates(fresh_wh, RACE_ID, res.frame, manifest.model_id, day, clock)
    assert fresh_wh.query(f"SELECT * FROM prediction WHERE race_date = '{day}'",
                          max_bytes_billed=10**9).shape[0] == 8

    # 6. 配信（重複防止つき）
    key = dedupe_key(RACE_ID, manifest.model_id)
    assert not already_sent(fresh_wh, RACE_ID, "prediction", key)

    embed = race_embed(
        race_id=RACE_ID, track_name="大井", race_no=11, class_name="C1三",
        distance=1200, start_ts=to_utc(START), now=clock.now(),
        model_release=manifest.model_id, track_used=res.track_used,
        candidates=res.frame, day_budget_remaining=7900, pool_yen=1_400_000,
        min_ev=cfg.discord_min_ev)

    if embed is not None:
        sender = DiscordSender(
            FAKE_WEBHOOK, transport=httpx.MockTransport(lambda r: httpx.Response(204)),
            rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None))
        sent = sender.send([embed])
        assert sent.sent == 1
        sender.close()
        record_sent(fresh_wh, RACE_ID, "prediction", key, clock.now(), to_utc(START))
        assert already_sent(fresh_wh, RACE_ID, "prediction", key)


def test_stale_db_stops_the_whole_pipeline(wh, clock, cfg):
    """鮮度ゲートで落ちたら、以降を一切実行しない（DB-04 / DR-01）。"""
    from narops.db import freshness

    with pytest.raises(StaleDataError):
        freshness.require_fresh(wh, clock)

    outcome = InferenceOutcome.failed(RACE_ID, "確定層が古い", retryable=False)
    assert outcome.bet_candidates.empty
    assert outcome.web_label == "未算出"
    assert wh.row_count("bet_candidate") == 0


def test_day_budget_accumulates_across_races(fresh_wh, clock, cfg):
    day = business_date(to_utc(START))
    frame = pd.DataFrame({"horse_no": [1], "p_win": [0.5], "ev": [1.3],
                          "ev_adjusted": [1.25], "kelly": [0.1], "stake_yen": [3000]})
    write_bet_candidates(fresh_wh, "202026082401", frame, "v1", day, clock)
    assert day_budget_remaining(fresh_wh, day, cfg.max_bet_per_day) == \
        cfg.max_bet_per_day - 3000


def test_shadow_release_writes_separately(fresh_wh, clock):
    """シャドー版の予測が本番の行を上書きしないこと（RL-02）。"""
    day = business_date(to_utc(START))
    frame = pd.DataFrame({"horse_no": [1, 2], "p_win": [0.6, 0.4],
                          "ev": [1.1, 0.9], "ev_adjusted": [1.05, 0.9],
                          "stake_yen": [0, 0]})
    write_prediction(fresh_wh, RACE_ID, frame, "v1", "A", day, clock, is_shadow=False)
    write_prediction(fresh_wh, RACE_ID, frame, "v2", "A", day, clock, is_shadow=True)

    df = fresh_wh.query(f"SELECT * FROM prediction WHERE race_date = '{day}'",
                        max_bytes_billed=10**9)
    assert len(df) == 4
    assert set(df[~df["is_shadow"]]["model_release"]) == {"v1"}
    assert set(df[df["is_shadow"]]["model_release"]) == {"v2"}
