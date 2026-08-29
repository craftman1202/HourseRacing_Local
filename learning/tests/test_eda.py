"""EDA。SKILLS の手順どおりに構成され、答えが数値で返ることを検証する。

- programmatic-eda: 手順1〜5 + 40項目チェックリスト + 2成果物
- data-quality-audit: 6次元スコアカード
- 設計書 §7: Q1〜Q6
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.config import eda_config
from nar.eda import profile, questions, quality, runner
from nar.eda.assumptions import AssumptionLog, seed_project_assumptions


@pytest.fixture(scope="module")
def conf():
    return eda_config()


@pytest.fixture(scope="module")
def th(conf):
    return profile.Thresholds.from_conf(conf)


@pytest.fixture(scope="module")
def result(synth_tables, conf, tmp_path_factory):
    out = tmp_path_factory.mktemp("eda")
    return runner.run(synth_tables["entry"], synth_tables["race"],
                      synth_tables["payout"], conf, out_dir=out)


# ------------------------------------------------------- programmatic-eda 手順1〜5
def test_plan_is_written_before_analysis():
    """analysis-planning: 何に答えるかを先に書き出す。"""
    p = runner.plan()
    assert p["id"].tolist() == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    assert p["why_it_matters"].str.len().min() > 10


def test_step1_overview_requires_explicit_grain(entry):
    ov = profile.overview(entry, "1行 = 1レース1頭")
    assert ov["grain"] == "1行 = 1レース1頭"
    assert ov["n_rows"] == len(entry)
    assert ov["unnamed_columns"] == []


def test_step1_grain_check_detects_unexpected_duplicate_keys(entry):
    ok = profile.grain_check(entry, ["race_id", "horse_no"])
    assert ok["is_unique"] is True

    doubled = pd.concat([entry.head(50), entry.head(50)])
    bad = profile.grain_check(doubled, ["race_id", "horse_no"])
    assert bad["is_unique"] is False and bad["n_duplicate_rows"] == 100


def test_step2_null_profile_applies_per_column_overrides(entry, th):
    """conf/eda.yaml の overrides が効くこと。既定 5% では馬体重が常に WARN になる。"""
    n = profile.null_profile(entry, th)
    row = n[n["column"] == "weight_kg"].iloc[0]
    assert row["warn_pct"] == 40.0, "列別上書きが効いていません"
    assert set(n["status"]) <= {"PASS", "WARN", "FAIL"}


def test_step2_warn_above_fail_is_a_config_error(th):
    """WARN だけ緩めて FAIL を据え置くと WARN を飛ばす区間ができる。設定ミスを通さない。"""
    broken = profile.Thresholds(per_column={"x": {"null_warn_pct": 50.0}})
    with pytest.raises(ValueError, match="null_warn_pct"):
        broken.null_limits("x")


def test_step2_missingness_is_evaluated_by_year(entry):
    """欠損を暗黙に drop しないため、パターン（MCAR らしさ）を年別に見る。"""
    e = entry.assign(year=pd.to_datetime(entry["race_date"]).dt.year)
    pat = profile.missingness_pattern(e, by="year")
    assert "weight_kg" in pat.columns
    early = pat[pat["year"] <= 2012]["weight_kg"].mean()
    late = pat[pat["year"] >= 2018]["weight_kg"].mean()
    assert early > late + 5, "年による欠損率の差が検出できていません（MCAR ではない）"


def test_step3_outliers_flag_but_do_not_remove(entry, th):
    o = profile.outliers(entry, th)
    assert len(o) > 0
    assert (o["classification"] == "未分類").all(), (
        "外れ値は自動分類しない。実データ／誤り／構造的の判断は人が行う。")


def test_step4_distribution_summary_flags_skew(entry, th):
    d = profile.distributions(entry, th)
    assert {"skew", "skew_status", "suggested_transform"} <= set(d.columns)
    odds = d[d["column"] == "odds_win"].iloc[0]
    assert odds["skew"] > 1.0, "単勝オッズは強く右に歪むはず"
    assert odds["suggested_transform"] == "log"


def test_step5_correlations_flag_strong_pairs(th):
    df = pd.DataFrame({"a": np.arange(500.0), "b": np.arange(500.0) * 2 + 1, "c": np.random.default_rng(0).normal(size=500)})
    c = profile.correlations(df, th)
    pair = c.iloc[0]
    assert {pair["col_a"], pair["col_b"]} == {"a", "b"}
    assert pair["flag"] == "near_perfect"


# ------------------------------------------------------ data-quality-audit 6次元
def test_quality_scorecard_passes_on_clean_data(synth_tables):
    sc = quality.audit(synth_tables["entry"], synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    assert sc.verdict() in {"PASS", "CONDITIONAL"}
    assert set(sc.scores) == set(quality.DIMENSIONS)


def test_quality_scorecard_catches_injected_defects(synth_tables):
    e = synth_tables["entry"].copy()
    e = pd.concat([e, e.head(20)])              # Uniqueness: キー重複
    e.loc[e.index[:5], "distance"] = -100       # Validity: 値域外
    sc = quality.audit(e, synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    dims = {f.dimension for f in sc.findings}
    assert "Uniqueness" in dims and "Validity" in dims
    assert sc.score() < 10.0


def test_quality_detects_orphan_foreign_keys(synth_tables):
    e = synth_tables["entry"].copy()
    e.loc[e.index[:10], "race_id"] = "999999999999"
    sc = quality.audit(e, synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    orphan = [f for f in sc.findings if f.check.startswith("entry → race")]
    assert len(orphan) == 1
    assert orphan[0].rows_affected == 10, "孤児件数が実際の件数と一致しません"


def test_quality_reports_no_orphans_on_consistent_tables(synth_tables):
    sc = quality.audit(synth_tables["entry"], synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    assert not any("孤児" in f.check for f in sc.findings)


def test_quality_detects_future_dated_records_with_results(synth_tables):
    """未来の日付に確定した着順があるのは時点の取り違え。これは CRITICAL。"""
    e = synth_tables["entry"].copy()
    e.loc[e.index[:3], "race_date"] = pd.Timestamp("2099-01-01")
    sc = quality.audit(e, synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    assert any(f.check == "未来日付の確定結果" and f.severity == quality.CRITICAL
               for f in sc.findings)


def test_quality_does_not_flag_scheduled_races_as_defects(synth_tables):
    """月次ファイルには当月の未実施レースも入る。着順が無ければ予定であって欠陥ではない。

    これを CRITICAL にすると、毎月の取り込みが常に品質 FAIL になる。
    """
    e = synth_tables["entry"].copy()
    e.loc[e.index[:3], "race_date"] = pd.Timestamp("2099-01-01")
    e.loc[e.index[:3], ["finish_pos", "is_win", "time_sec"]] = np.nan
    sc = quality.audit(e, synth_tables["race"], synth_tables["payout"],
                       today=pd.Timestamp("2026-01-01"))
    assert not any(f.check == "未来日付の確定結果" for f in sc.findings)
    scheduled = [f for f in sc.findings if f.check == "未実施レース"]
    assert scheduled and scheduled[0].severity == quality.INFO


# ------------------------------------------------------------------------ Q1
def test_q1_coverage_map_and_start_year_proposal(result):
    cov = result["coverage"]
    assert {"year", "baba_code", "n_races"} <= set(cov.columns)
    assert result["usable"]["proposed_train_from"] is not None


def test_q1_early_years_are_excluded_when_missingness_is_high(synth_tables):
    cov = questions.coverage_map(synth_tables["entry"], synth_tables["race"], synth.TRACKS)
    u = questions.usable_from_year(cov, max_null_pct=10.0, min_races_per_year=1)
    # 合成データは 2010〜2012 に馬体重欠損を仕込んである
    assert u["proposed_train_from"] >= 2013


# ------------------------------------------------------------------------ Q3
def test_q3_identity_audit_reports_pass_on_clean_keys(result):
    assert result["identity"]["verdict"] == "PASS"
    assert result["identity"]["n_same_day_duplicate_starts"] == 0


# ------------------------------------------------------------------------ Q4
def test_q4_favourite_longshot_bias_is_measured(result):
    flb = result["flb"]
    assert {"q_mean", "win_rate", "bias", "roi_flat"} <= set(flb.columns)
    assert flb["q_mean"].is_monotonic_increasing


def test_q4_longshot_bias_is_detected_when_injected():
    """人気薄を過剰に買う市場を仕込むと、低確率帯の bias が負になること。"""
    t = synth.generate(synth.SynthConfig(n_races=2500, seed=4, longshot_tilt=1.20))
    flb = questions.favourite_longshot_bias(t["entry"], n_bins=6)
    assert flb.iloc[0]["bias"] < 0, flb[["q_mean", "win_rate", "bias"]]


def test_q4_popularity_winrate_is_monotone(result):
    pw = result["target"]["popularity_winrate"].sort_values("popularity")
    top = pw[pw["popularity"] <= 5]["win_rate"]
    assert top.is_monotonic_decreasing, "人気順で勝率が単調に下がらないのは異常"


# ------------------------------------------------------------------------ Q5
def test_q5_psi_is_zero_for_identical_distributions():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert questions.psi(s, s.copy()) == pytest.approx(0.0, abs=1e-9)


def test_q5_psi_detects_a_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(2, 1, 5000))
    assert questions.psi(a, b) > 0.25


def test_q5_change_points_require_multiple_columns_to_shift():
    drift = pd.DataFrame([
        {"year": 2022, "column": "a", "psi": 0.4, "status": "FAIL"},
        {"year": 2022, "column": "b", "psi": 0.3, "status": "FAIL"},
        {"year": 2023, "column": "a", "psi": 0.5, "status": "FAIL"},
    ])
    cp = questions.change_points(drift, min_columns=2)
    assert cp["year"].tolist() == [2022], "1列だけの変動を変化点にしてはいけない"


# ------------------------------------------------------------------------ Q6
def test_q6_measured_takeout_is_used_instead_of_nominal(result):
    t = result["takeout"]
    assert {"takeout_measured", "takeout_nominal", "diff_pt", "status"} <= set(t.columns)
    assert (t["diff_pt"].abs() <= 2.0).all()


# --------------------------------------------------------- チェックリストと成果物
def test_checklist_has_40_items_and_keeps_manual_ones_manual(result):
    c = result["checklist"]
    assert len(c) >= 40, f"{len(c)} 項目しかありません"
    assert (c["status"] == "MANUAL").sum() >= 10, (
        "自動判定できない項目まで機械が PASS にしています")


def test_checklist_signoff_blocks_on_failures(result):
    s = result["signoff"]
    assert set(s["counts"]) <= {"PASS", "WARN", "FAIL", "MANUAL"}
    assert s["can_proceed"] == (s["counts"].get("FAIL", 0) == 0)


def test_both_deliverables_are_written(result):
    report = result["report_path"].read_text(encoding="utf-8")
    findings = result["findings_path"].read_text(encoding="utf-8")
    for section in ("Q1.", "Q2.", "Q3.", "Q4.", "Q5.", "Q6.", "チェックリスト"):
        assert section in report, f"レポートに {section} がありません"
    assert "次のアクション" in findings and "残る不確実性" in findings


def test_report_states_leak_conclusion_up_front(result):
    """リーク判定は結果と同じ場所に書く。隠さない。

    合否そのものは問わない。問うのは (1) 判定不能な列を残していないこと、
    (2) 使用可とした列に as-of-race の証拠が揃っていること、
    (3) 結論が所見サマリに出ていること。
    """
    leak = result["leak"]
    assert leak["undetermined"] == [], (
        f"判定不能の列を残したまま先へ進んでいます: {leak['undetermined']}")
    assert set(leak["whitelist"]) & set(leak["discard"]) == set(), (
        "同じ列が使用可と破棄の両方に入っています")
    detail = result["leak_detail"]
    for col in leak["whitelist"]:
        tests = {r["test"]: r["verdict"] for _, r in detail[detail["column"] == col].iterrows()}
        assert tests.get("LK-09") == "as_of_race", f"{col}: 初回出走行の検証が通っていません"
        assert "as_of_race" in (tests.get("LK-10"), tests.get("LK-11")), (
            f"{col}: 自分の結果を含まないことの検証が通っていません")
    assert leak["conclusion"] in result["findings_path"].read_text(encoding="utf-8")


# ------------------------------------------------------------ assumptions log
def test_assumptions_are_seeded_and_persisted(tmp_path):
    log = seed_project_assumptions(AssumptionLog(tmp_path / "a.json"))
    log.save()
    reloaded = AssumptionLog(tmp_path / "a.json")
    assert len(reloaded.items) == len(log.items) >= 6

    # 再実行しても積み上がらない。記録は前提の一覧であって実行履歴ではない。
    seed_project_assumptions(reloaded).save()
    assert len(AssumptionLog(tmp_path / "a.json").items) == len(log.items)
    cats = {a.category for a in reloaded.items}
    assert {"exclusion", "scope", "method", "threshold", "limitation"} <= cats


def test_assumptions_record_why_and_impact():
    log = seed_project_assumptions(AssumptionLog())
    for a in log.items:
        assert a.rationale and a.impact_if_wrong, f"{a.id} に理由か影響が書かれていません"
