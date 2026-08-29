"""EDA オーケストレーション。

analysis-planning（何に答えるかを先に書き出す）→ programmatic-eda の手順1〜5 →
data-quality-audit のスコアカード → 設計書 §7 の6問 → 40項目チェックリスト →
報告、の順で回す。順序に意味があるので個別に呼ぶより run() を使う。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..synth import TRACKS
from . import checklist as cl
from . import leakage, profile, questions, quality
from .assumptions import AssumptionLog, seed_project_assumptions
from .report import write_findings_summary, write_report

PLAN = [
    ("Q1", "時系列カバレッジと欠損の地図", "どの年から学習に使えるかを決める"),
    ("Q2", "累積成績8列の時点性判定", "特徴量設計を左右する。先に進む前に必ず結論を出す"),
    ("Q3", "名寄せの検証", "過去成績特徴量の精度を直接決める"),
    ("Q4", "ターゲット基礎統計と favourite-longshot bias", "市場効率性の水準を測る"),
    ("Q5", "年次PSIによる構造変化点の検出", "walk-forward 分割の設計に直結する"),
    ("Q6", "券種別・場別の控除率実測", "期待値計算に使う控除率を公称値から実測値に置き換える"),
]

DRIFT_COLUMNS = ("time_sec", "odds_win", "weight_kg", "distance")


def plan() -> pd.DataFrame:
    """analysis-planning: 答えるべき質問を先に書き出す。"""
    return pd.DataFrame(PLAN, columns=["id", "question", "why_it_matters"])


def run(
    entry: pd.DataFrame,
    race: pd.DataFrame,
    payout: pd.DataFrame,
    conf: dict,
    out_dir: str | Path | None = None,
    daily_snapshot: pd.DataFrame | None = None,
) -> dict:
    """EDA 一式を実行して結果 dict を返す。out_dir を渡すとレポートも書く。

    daily_snapshot（当日ファイル由来の出馬表）を渡すと LK-01（世代間差分）まで
    実行できる。渡さない場合は LK-09/10/11 で判定し、判定不能は破棄側に倒す。
    """
    th = profile.Thresholds.from_conf(conf)
    log = seed_project_assumptions(AssumptionLog(
        Path(out_dir) / "assumptions.json" if out_dir else None))

    # --- programmatic-eda 手順 1〜5 -------------------------------------------
    grain_desc = "1行 = 1レースにおける1頭の出走（race_id × horse_no）"
    ov = profile.overview(entry, grain_desc)
    grain = profile.grain_check(entry, ["race_id", "horse_no"])
    nulls = profile.null_profile(entry, th)
    e_year = entry.assign(year=pd.to_datetime(entry["race_date"]).dt.year)
    miss_year = profile.missingness_pattern(e_year, by="year")
    outl = profile.outliers(entry, th)
    dists = profile.distributions(entry, th)
    corr = profile.correlations(entry, th)

    # --- data-quality-audit --------------------------------------------------
    sc = quality.audit(entry, race, payout)
    scorecard = {
        "score": sc.score(), "verdict": sc.verdict(), "scores": sc.scores,
        "weights": quality.DIMENSIONS, "findings": sc.to_frame(),
    }

    # --- Q1 カバレッジ --------------------------------------------------------
    coverage = questions.coverage_map(entry, race, TRACKS)
    usable = questions.usable_from_year(coverage)

    # --- Q2 リーク検証（最優先） ----------------------------------------------
    verdicts: list[leakage.ColumnVerdict] = []
    if daily_snapshot is not None:
        verdicts += leakage.lk01_generation_diff(daily_snapshot, entry)
    present = [c for c in leakage.CUMULATIVE_RECORD_COLS if c in entry.columns]
    if present:
        # 列ごとの実体（集計範囲・条件・種別）に合わせた LK-09/10/11。
        # LK-02/03 だけでは「自分の結果を含むか」を切り分けられない。
        # 回り・馬場状態はレース側の属性なので、判定前に結合しておく。
        # entry 側に同名列があると join が turn_x / turn_y に化けるので、
        # 無い列だけを持ってくる。
        attrs = [c for c in ("turn", "baba_condition")
                 if c in race.columns and c not in entry.columns]
        target = (entry.merge(race[["race_id", *attrs]], on="race_id", how="left")
                  if attrs else entry)
        verdicts += leakage.run_all(target, tuple(present))
    else:
        verdicts += [
            leakage.ColumnVerdict(c, "LK-09", "undetermined",
                                  "silver に該当列がありません（既に破棄済み）", False)
            for c in leakage.CUMULATIVE_RECORD_COLS
        ]
    leak = leakage.decide(verdicts)
    leak_detail = pd.DataFrame([v.__dict__ for v in verdicts])
    label_corr = leakage.label_correlation_screen(
        entry, exclude=("finish_pos", "time_sec", "popularity", "odds_win"))

    # --- Q3 名寄せ ------------------------------------------------------------
    identity = questions.identity_audit(entry)

    # --- Q4 ターゲット --------------------------------------------------------
    target = questions.target_basics(entry)
    flb = (questions.favourite_longshot_bias(entry)
           if "odds_win" in entry.columns else pd.DataFrame())

    # --- Q5 分布シフト --------------------------------------------------------
    psi_conf = conf.get("psi", {})
    drift_cols = [c for c in DRIFT_COLUMNS if c in entry.columns]
    drift = questions.yearly_drift(
        entry, drift_cols, bins=psi_conf.get("bins", 10),
        warn=psi_conf.get("warn", 0.10), fail=psi_conf.get("fail", 0.25))
    cps = questions.change_points(drift, fail=psi_conf.get("fail", 0.25))

    # --- Q6 控除率 ------------------------------------------------------------
    takeout = (questions.measured_takeout(entry)
               if "odds_win" in entry.columns else pd.DataFrame())

    # --- チェックリストとサインオフ -------------------------------------------
    dupes_pct = 100.0 * entry.duplicated().mean()
    check = cl.build(
        ov=ov, grain=grain, nulls=nulls, dupes_pct=dupes_pct, outliers=outl,
        dists=dists, corr=corr, coverage=coverage,
        quality_verdict=scorecard["verdict"], leak_conclusion=leak["conclusion"],
        leak_undetermined=list(leak["undetermined"]),
    )

    result = {
        "plan": plan(), "overview": ov, "grain": grain, "nulls": nulls,
        "missingness_by_year": miss_year, "outliers": outl, "distributions": dists,
        "correlations": corr, "scorecard": scorecard, "coverage": coverage,
        "usable": usable, "leak": leak, "leak_detail": leak_detail,
        "label_corr": label_corr, "identity": identity, "target": target, "flb": flb,
        "drift": drift, "change_points": cps, "takeout": takeout,
        "checklist": check, "signoff": cl.signoff(check),
        "assumptions": log, "assumptions_md": log.to_markdown(),
    }

    if out_dir:
        log.save()
        result["report_path"] = write_report(result, out_dir)
        result["findings_path"] = write_findings_summary(result, out_dir)
    return result
