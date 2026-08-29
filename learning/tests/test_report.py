"""性能レポート生成と、線形化した指標関数の境界挙動。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import report as rpt
from nar.eval.metrics import normalize_within_race, race_softmax


# ------------------------------------------------------------------- 指標の境界
def test_race_softmax_handles_single_horse_race():
    p = race_softmax(np.array([3.0]), np.array(["R1"]))
    assert p[0] == pytest.approx(1.0)


def test_race_softmax_is_shift_invariant():
    """スコア全体に定数を足しても確率が変わらないこと（数値安定化の確認）。"""
    rid = np.array(["R1"] * 4 + ["R2"] * 3)
    s = np.array([1.0, 2.0, 3.0, 0.5, -1.0, 0.0, 2.0])
    assert np.allclose(race_softmax(s, rid), race_softmax(s + 1000.0, rid))


def test_race_softmax_does_not_overflow_on_large_scores():
    rid = np.array(["R1"] * 3)
    p = race_softmax(np.array([800.0, 799.0, 0.0]), rid)
    assert np.isfinite(p).all() and p.sum() == pytest.approx(1.0)


def test_normalize_within_race_falls_back_to_uniform_on_zero_mass():
    """総和 0 のレースで NaN を返さない。NaN は下流の NLL を黙って壊す。"""
    rid = np.array(["R1"] * 3 + ["R2"] * 2)
    p = normalize_within_race(np.array([0.0, 0.0, 0.0, 1.0, 3.0]), rid)
    assert np.allclose(p[:3], 1 / 3)
    assert np.allclose(p[3:], [0.25, 0.75])
    assert np.isfinite(p).all()


def test_normalize_preserves_group_boundaries_with_repeated_ids():
    """非連続に現れる race_id でも取り違えないこと。"""
    rid = np.array(["A", "B", "A", "B"])
    p = normalize_within_race(np.array([1.0, 1.0, 3.0, 7.0]), rid)
    assert p[0] == pytest.approx(0.25) and p[2] == pytest.approx(0.75)
    assert p[1] == pytest.approx(0.125) and p[3] == pytest.approx(0.875)


# ------------------------------------------------------------------- レポート
@pytest.fixture
def artifacts(tmp_path):
    pd.DataFrame([
        {"fold": 1, "model": "clogit", "race_nll": 2.10, "brier": 0.8, "top1": 0.27,
         "top3": 0.49, "ndcg@3": 0.5, "spearman": 0.2, "ece": 0.01},
        {"fold": 2, "model": "clogit", "race_nll": 2.06, "brier": 0.8, "top1": 0.28,
         "top3": 0.50, "ndcg@3": 0.5, "spearman": 0.2, "ece": 0.01},
        {"fold": 1, "model": "baseline_uniform", "race_nll": 2.30, "brier": 0.9,
         "top1": 0.11, "top3": 0.33, "ndcg@3": 0.4, "spearman": 0.0, "ece": 0.0},
        {"fold": 2, "model": "baseline_uniform", "race_nll": 2.30, "brier": 0.9,
         "top1": 0.11, "top3": 0.33, "ndcg@3": 0.4, "spearman": 0.0, "ece": 0.0},
    ]).to_csv(tmp_path / "fold_metrics.csv", index=False)
    pd.DataFrame([{"model": "ensemble_stacked", "race_nll": 2.04, "top1": 0.28,
                   "ece": 0.006}]).to_csv(tmp_path / "ensemble_metrics.csv", index=False)
    pd.DataFrame([{"model": "clogit", "weight": 1.0}]).to_csv(
        tmp_path / "ensemble_weights.csv", index=False)
    pd.DataFrame([{"model": "clogit", "race_nll": 2.02, "brier": 0.8, "top1": 0.29,
                   "top3": 0.5, "ndcg@3": 0.5, "spearman": 0.2, "ece": 0.01}]).to_csv(
        tmp_path / "oos_metrics.csv", index=False)
    pd.DataFrame([{"id": "RF-01", "fired": False, "severity": "Blocker",
                   "observed": 0.29, "message": "OOS Top-1 が 60% 超"}]).to_csv(
        tmp_path / "oos_guards.csv", index=False)
    return tmp_path


@pytest.fixture
def meta():
    return {
        "data_source": "合成データ", "n_races": 1000, "n_entries": 12000,
        "period": "2010-01-01 〜 2025-12-31", "dataset_version": "abc123",
        "models": ["clogit"], "embargo_days": 180,
        "oos_period": "2024-02-01 〜 データ末尾", "oos_access_count": 1,
        "conclusions": ["結論A"], "limitations": ["限界A"],
    }


def test_report_includes_every_required_section(artifacts, meta):
    md = rpt.build(artifacts, meta)
    for section in ("walk-forward 交差検証", "アンサンブル", "OOS 最終評価",
                    "Too-Good-To-Be-True", "結論", "この結果の限界"):
        assert section in md, f"{section} が欠けています"


def test_report_states_synthetic_data_caveat_up_front(artifacts, meta):
    """合成データであることを冒頭で明示する。隠さない。"""
    md = rpt.build(artifacts, meta)
    head = md[:md.index("## 1.")]
    assert "合成データ" in head
    assert "実データの性能を意味しません" in head


def test_report_ranks_models_by_nll(artifacts, meta):
    md = rpt.build(artifacts, meta)
    body = md[md.index("## 1."):md.index("## 2.")]
    assert body.index("条件付きロジット") < body.index("レース内一様分布"), (
        "NLL が小さい順に並んでいません")


def test_report_uses_readable_model_labels(artifacts, meta):
    md = rpt.build(artifacts, meta)
    assert "条件付きロジット" in md and "ベースライン: レース内一様分布" in md


def test_report_reports_oos_access_count(artifacts, meta):
    assert "1 回" in rpt.build(artifacts, meta)


def test_report_writes_file(artifacts, meta, tmp_path):
    out = rpt.write(artifacts, meta, tmp_path / "perf.md")
    assert out.exists() and len(out.read_text(encoding="utf-8")) > 500


def test_report_survives_missing_artifacts(tmp_path, meta):
    """OOS 未開封など、成果物が揃っていなくても落ちずに書けること。"""
    md = rpt.build(tmp_path, meta)
    assert "OOS は未開封です" in md
