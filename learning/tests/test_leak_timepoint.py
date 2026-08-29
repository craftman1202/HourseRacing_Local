"""累積成績8列の時点性判定（LK-09/10/11）。

実データで「8列すべてが as-of-race」という結論を出した以上、その判定ロジックが
3通りの時点性を確実に見分けられることを合成データで固定しておく必要がある。
判定が甘くなればリーク列がそのまま特徴量に入る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar.eda import leakage as L


def _career(n_starts: int, wins: list[int], mode: str, horse: str = "h1",
            age: int = 2, start: str = "2010-01-01") -> pd.DataFrame:
    """1頭ぶんの出走履歴を作る。

    mode="as_of_race"      … 当該レースを含まない通算（正しい pre-race 値）
    mode="includes_own"    … 当該レースの結果を含む通算（リーク）
    mode="as_of_download"  … 全行が最終値（ファイル生成時点）
    """
    rows = []
    cum_first = cum_other = 0
    ts = pd.Timestamp(start)
    for i in range(n_starts):
        won = wins[i]
        before = (cum_first, cum_other)
        cum_first += won
        cum_other += 1 - won
        after = (cum_first, cum_other)
        f, o = {"as_of_race": before, "includes_own": after}.get(mode, (None, None))
        rows.append({"horse_sk": horse, "jockey_sk": "j1", "baba_code": 20,
                     "distance": 1200.0, "turn": "左", "baba_condition": "良",
                     "age": age + i // 3, "is_win": won,
                     "start_ts": ts + pd.Timedelta(days=30 * i),
                     "time_sec": 78.0 - i * 0.6,
                     "n_first": f, "n_other": o})
    df = pd.DataFrame(rows)
    if mode == "as_of_download":
        df["全成績"] = f"{cum_first}-0-0-{cum_other}"
    else:
        df["全成績"] = [f"{a}-0-0-{b}"
                      for a, b in zip(df["n_first"], df["n_other"])]
    return df.drop(columns=["n_first", "n_other"])


def _population(mode: str, n_horses: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for h in range(n_horses):
        n = int(rng.integers(6, 12))
        wins = list(rng.integers(0, 2, size=n))
        frames.append(_career(n, wins, mode, horse=f"h{h}"))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ LK-09
@pytest.mark.parametrize("mode,expected", [
    ("as_of_race", "as_of_race"),
    ("includes_own", "as_of_download"),
])
def test_lk09_separates_asof_race_from_includes_own(mode, expected):
    """初回出走行が 0 か 1 か。この2つを分けられるのは LK-09 だけ。"""
    v = L.lk09_first_occurrence(_population(mode), "全成績")
    assert v.verdict == expected, v.evidence


def test_lk09_ignores_horses_whose_career_starts_before_the_data():
    """1998年以前から走っている馬を混ぜても判定は変わらない。

    左打ち切りの馬は初回行が初回でないので、除外できていないと閾値を割る。
    """
    clean = _population("as_of_race")
    censored = _career(8, [1] * 8, "as_of_race", horse="old", age=7,
                       start="1998-02-01")
    censored["全成績"] = "40-0-0-60"          # 1998年以前の実績を持ったまま登場
    mixed = pd.concat([clean, censored], ignore_index=True)
    assert L.lk09_first_occurrence(mixed, "全成績").verdict == "as_of_race"


def test_lk09_ignores_horses_transferred_in_with_prior_record():
    """中央からの転入馬（初出走が3歳以上）も同様に外す。"""
    clean = _population("as_of_race")
    transfers = []
    for i in range(40):
        t = _career(5, [0] * 5, "as_of_race", horse=f"t{i}", age=5,
                    start="2012-01-01")
        t["全成績"] = "3-2-1-9"               # 中央での実績を引き継いでいる
        transfers.append(t)
    mixed = pd.concat([clean, *transfers], ignore_index=True)
    assert L.lk09_first_occurrence(mixed, "全成績").verdict == "as_of_race"


# ------------------------------------------------------------------ LK-10
@pytest.mark.parametrize("mode,expected", [
    ("as_of_race", "as_of_race"),
    ("includes_own", "as_of_download"),
])
def test_lk10_checks_where_a_win_first_appears(mode, expected):
    v = L.lk10_own_result_excluded(_population(mode, n_horses=120), "全成績")
    assert v.verdict == expected, v.evidence


def test_lk10_takes_the_next_row_within_scope_not_the_next_row_overall():
    """騎手成績の集計範囲は (馬, 騎手)。馬の次走で見ると別カウンタと比べてしまう。"""
    a = _career(6, [1, 0, 1, 0, 1, 0], "as_of_race", horse="h1")
    b = a.copy()
    a["jockey_sk"], b["jockey_sk"] = "jA", "jB"
    b["start_ts"] = a["start_ts"] + pd.Timedelta(days=15)
    both = pd.concat([a, b], ignore_index=True)
    both = pd.concat([both] * 30, ignore_index=True)
    both["horse_sk"] = np.repeat([f"h{i}" for i in range(30)], 12)
    both["騎手成績"] = both["全成績"]
    assert L.lk10_own_result_excluded(both, "騎手成績").verdict == "as_of_race"


# ------------------------------------------------------------------ LK-02
def test_lk02_still_catches_a_frozen_download_time_value():
    v = L.lk02_monotonicity(_population("as_of_download"), "全成績")
    assert v.verdict == "as_of_download", v.evidence


def test_lk02_is_not_applied_to_conditional_counters():
    """条件付き列に単調性を掛けると「条件外で増えない」ことを不整合と誤判定する。"""
    assert L.UNCONDITIONAL_COUNTERS == frozenset({"全成績"})
    tests = {v.test for v in L.run_all(_population("as_of_race"), ("ダート左成績",))}
    assert "LK-02" not in tests


# ------------------------------------------------------------------ LK-11
def test_lk11_prefers_prior_only_cumulative_minimum():
    pop = _population("as_of_race", n_horses=40)
    pop = pop.sort_values(["horse_sk", "start_ts"]).copy()
    keys = ["horse_sk", "baba_code", "distance"]
    pop["_prev"] = pop.groupby(keys, sort=False)["time_sec"].shift(1)
    prior = pop.groupby(keys, sort=False)["_prev"].cummin()
    pop["最高タイム"] = [
        "" if pd.isna(x) else f"{int(x // 60)}:{x % 60:04.1f}" for x in prior]
    pop = pop.drop(columns=["_prev"])
    v = L.lk11_best_time_reconstruction(pop, "最高タイム")
    assert v.verdict == "as_of_race", v.evidence


# ------------------------------------------------------------------ decide
def test_decide_requires_both_gates_before_whitelisting():
    """LK-09 だけ通っても使用可にしない。両方揃って初めてホワイトリスト。"""
    only_first = [L.ColumnVerdict("X", "LK-09", "as_of_race", "", True),
                  L.ColumnVerdict("X", "LK-10", "undetermined", "", False)]
    assert L.decide(only_first)["whitelist"] == []

    both = [L.ColumnVerdict("X", "LK-09", "as_of_race", "", True),
            L.ColumnVerdict("X", "LK-10", "as_of_race", "", True)]
    assert L.decide(both)["whitelist"] == ["X"]


def test_decide_lets_a_single_download_verdict_veto_everything():
    vs = [L.ColumnVerdict("X", "LK-09", "as_of_race", "", True),
          L.ColumnVerdict("X", "LK-10", "as_of_race", "", True),
          L.ColumnVerdict("X", "LK-02", "as_of_download", "決定的な反証", False)]
    d = L.decide(vs)
    assert d["whitelist"] == [] and d["discard"] == ["X"]


# ------------------------------------------------------------------ ホワイトリスト
def test_whitelist_matches_the_recorded_verdict():
    """ASOF_RACE_WHITELIST は leak_verdict.json の結論と一致していなければならない。

    判定を回さずに列を足す（あるいは判定が覆ったのに列が残る）と、そのまま
    リーク列が特徴量に入る。証拠とコードが乖離していないことを固定する。
    """
    import json
    from pathlib import Path

    from nar.transform.prerace import ASOF_RACE_WHITELIST

    verdict = Path(__file__).resolve().parents[1] / "artifacts" / "leak_verdict.json"
    if not verdict.exists():
        pytest.skip("leak_verdict.json がありません（nar leak-check 未実行）")
    recorded = set(json.loads(verdict.read_text(encoding="utf-8"))["whitelist"])
    assert set(ASOF_RACE_WHITELIST) == recorded, (
        f"コード側 {sorted(ASOF_RACE_WHITELIST)} と判定結果 {sorted(recorded)} が違います")
