"""人気順を足した派生系統（conf_pop / conf_banei_pop、2026-10-02）。

既存のトラックA（市場と独立）を一切変えないこと、人気由来の列が通常のトラックA に
紛れ込めば FE-10 の検査で落ちること、を固定する。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from nar.cli import _gold_subdir
from nar.config import feature_config
from nar.errors import LeakageError
from nar.features.builder import ASOF_FEATURES, POPULARITY_FEATURES, asof_features
from nar.transform.prerace import assert_no_market_info

ROOT = Path(__file__).resolve().parents[1]


def test_existing_configs_do_not_use_popularity():
    for d in ("conf", "conf_banei"):
        cfg = feature_config(str(ROOT / d))
        assert cfg.with_popularity is False
        assert not set(POPULARITY_FEATURES) & set(asof_features(cfg))


@pytest.mark.parametrize("d,variant", [("conf_pop", "flat"), ("conf_banei_pop", "banei")])
def test_pop_configs_add_exactly_the_popularity_columns(d, variant):
    cfg = feature_config(str(ROOT / d))
    base = feature_config(str(ROOT / d.replace("_pop", "")))
    assert cfg.with_popularity and cfg.variant == variant
    assert asof_features(cfg) == asof_features(base) + POPULARITY_FEATURES


def test_popularity_columns_are_caught_by_the_track_a_blocklist():
    cfg = feature_config(str(ROOT / "conf"))
    df = pd.DataFrame({c: [1.0] for c in POPULARITY_FEATURES})
    with pytest.raises(LeakageError):
        assert_no_market_info(df, cfg.market_term_blocklist)


def test_pop_variant_writes_to_its_own_gold():
    flat = feature_config(str(ROOT / "conf"))
    assert _gold_subdir(flat) == "features_noodds"
    assert _gold_subdir(replace(flat, with_popularity=True)) == "features_noodds_pop"
    banei = feature_config(str(ROOT / "conf_banei"))
    assert _gold_subdir(replace(banei, with_popularity=True)) == "features_noodds_banei_pop"


def test_flat_feature_list_is_unchanged():
    assert asof_features(feature_config(str(ROOT / "conf"))) == ASOF_FEATURES
