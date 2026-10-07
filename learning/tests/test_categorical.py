"""カテゴリ変数の特徴量（nar.features.categorical、2026-10-07）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nar.config import feature_config
from nar.features.builder import ASOF_FEATURES, POPULARITY_FEATURES, asof_features
from nar.features.categorical import CATEGORICAL_FEATURES, _visiting, build, race_level

ROOT = Path(__file__).resolve().parents[1]


def _race(**kw):
    base = {"race_id": "R1", "race_date": "2024-05-01", "race_name": "Ｃ２一 ４歳以上",
            "condition_name": "サラブレッド系　一般　別定", "baba_condition": "稍重",
            "weather": "小雨", "surface": "ダート", "track_name": "大井"}
    base.update(kw)
    return pd.DataFrame([base])


def test_race_level_parses_class_age_weight_rule_going_and_weather():
    r = race_level(_race()).iloc[0]
    assert (r.c_tier, r.c_tier_sub) == (2.0, 2.0)
    assert r.c_age_upper == 1 and r.c_age_2yo == 0 and r.c_age_3yo == 0
    assert r.c_wr_betteitei == 1 and r.c_wr_teiryo == 0
    assert r.c_going == 1 and r.c_rain == 1 and r.c_snow == 0 and r.c_turf == 0


def test_newcomer_maiden_and_jra_exchange_flags():
    r = race_level(_race(race_name="JRA認定フレッシュチャレンジ2歳新馬",
                         condition_name="サラブレッド系 2歳 馬齢")).iloc[0]
    assert r.c_is_newcomer == 1 and r.c_is_jra_exchange == 1 and r.c_age_2yo == 1
    assert np.isnan(r.c_tier), "JRA の A を A クラスと読まない"
    assert race_level(_race(race_name="2歳 未勝利")).iloc[0].c_is_maiden == 1


def test_ages_before_2001_are_converted_from_kazoe():
    # 2000 年の「4才」= 今の 3 歳
    r = race_level(_race(race_date="2000-06-01", race_name="4才", condition_name="サラブレッド系 4才")).iloc[0]
    assert r.c_age_3yo == 1


def test_visiting_is_counted_by_circuit_not_by_track():
    aff = pd.Series(["船橋", "兵庫", "愛知", "JRA", "高知"])
    trk = pd.Series(["大井", "園田", "笠松", "大井", "大井"])
    assert list(_visiting(aff, trk)) == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_horse_level_columns_and_gelding_codes():
    e = pd.DataFrame({
        "race_id": "R1", "race_date": "2024-05-01", "baba_code": 20, "horse_no": [1, 2, 3],
        "sex": ["牡", "セン", "去"], "birth_ymd": ["20200401", "20190401", "20210401"],
        "weight_carried": ["56", "☆53", "◇54"], "weight_kg": [480, 460, 1200],
        "weight_diff": ["+2", "-4", ""],
        "jockey_affil_sk": ["a@大井", "b@高知", "c@JRA"], "trainer_affil_sk": ["x@大井"] * 3})
    x = build(e, _race(), {20: "大井"})
    assert list(x.c_is_gelding) == [0, 1, 1]
    assert list(x.c_age) == [4, 5, 3] and abs(x.c_age_rel.sum()) < 1e-9
    assert list(x.c_weight_carried) == [56, 53, 54]
    assert list(x.c_jockey_apprentice) == [0, 1, 0] and list(x.c_jockey_female) == [0, 0, 1]
    assert np.isnan(x.c_body_weight.iloc[2]), "1200kg は平地の馬体重として解析ミス扱い"
    assert list(x.c_jockey_visiting) == [0, 1, 0] and list(x.c_jockey_jra) == [0, 0, 1]
    assert list(x.columns[2:]) == list(CATEGORICAL_FEATURES)


def test_variant_adds_exactly_the_categorical_columns_and_leaves_others_alone():
    cat = feature_config(str(ROOT / "conf_pop_cat"))
    assert asof_features(cat) == ASOF_FEATURES + POPULARITY_FEATURES + CATEGORICAL_FEATURES
    for d in ("conf", "conf_pop", "conf_banei"):
        assert not set(CATEGORICAL_FEATURES) & set(asof_features(feature_config(str(ROOT / d))))
