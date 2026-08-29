"""LK-01..08: リーク検証。

他のすべてのテストが通ってもこの章が通らなければ成果物は無価値、という位置づけ。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nar import synth
from nar.config import feature_config
from nar.eda import leakage
from nar.errors import LeakageError
from nar.features.builder import ASOF_FEATURES, build, content_hash
from nar.models.base import to_batch
from nar.models.clogit import ConditionalLogit
from nar.eval import metrics
from nar.transform.prerace import (
    CUMULATIVE_RECORD_COLS, POST_RACE_ALL, assert_prerace, to_prerace,
)

CUTOFF = "2020-01-01"


def _asof_frame(entry: pd.DataFrame, values: dict[str, list]) -> pd.DataFrame:
    """累積成績列を持つ小さな出馬表を作る。"""
    rows = []
    for horse, vals in values.items():
        for i, v in enumerate(vals):
            rows.append({
                "horse_sk": horse, "race_id": f"R{i}", "horse_no": 1,
                "start_ts": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                "全成績": v, "is_win": 0, "finish_pos": 5,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ LK-01
def test_lk01_generation_diff_detects_as_of_download():
    daily = pd.DataFrame({"race_id": ["R1"], "horse_no": [1], "全成績": ["3-1-0-2"]})
    monthly = pd.DataFrame({"race_id": ["R1"], "horse_no": [1], "全成績": ["9-4-2-11"]})
    v = [x for x in leakage.lk01_generation_diff(daily, monthly, ("全成績",))][0]
    assert v.verdict == "as_of_download"
    assert v.usable is False


def test_lk01_identical_generations_defer_to_lk02():
    same = pd.DataFrame({"race_id": ["R1"], "horse_no": [1], "全成績": ["3-1-0-2"]})
    v = leakage.lk01_generation_diff(same, same.copy(), ("全成績",))[0]
    assert v.verdict == "undetermined" and v.usable is False


# ------------------------------------------------------------------------ LK-02
def test_lk02_constant_value_is_as_of_download():
    """全行同一値 = ファイル生成時点の通算成績が全行にコピーされている。"""
    df = _asof_frame(None, {"H1": ["10-1-3-16"] * 6, "H2": ["4-0-1-9"] * 5})
    v = leakage.lk02_monotonicity(df)
    assert v.verdict == "as_of_download"


def test_lk02_incrementing_by_one_is_as_of_race():
    seq = ["0-0-0-0", "1-0-0-0", "1-1-0-0", "1-1-1-0", "1-1-1-1"]
    v = leakage.lk02_monotonicity(_asof_frame(None, {"H1": seq}))
    assert v.verdict == "as_of_race"


# ------------------------------------------------------------------------ LK-03
def test_lk03_terminal_matching_inclusive_total_is_discarded():
    """最終行が「最終レースを含む」通算と一致 = その行は自分の結果を知っている。"""
    seq = ["1-0-0-0", "2-0-0-0", "3-0-0-0"]   # 3走目の行が「3走ぶん」を持つ
    v = leakage.lk03_terminal_match(_asof_frame(None, {"H1": seq}))
    assert v.verdict == "as_of_download" and v.usable is False


def test_lk03_terminal_matching_exclusive_total_is_usable():
    seq = ["0-0-0-0", "1-0-0-0", "2-0-0-0"]   # 3走目の行が「2走ぶん」を持つ
    v = leakage.lk03_terminal_match(_asof_frame(None, {"H1": seq}))
    assert v.verdict == "as_of_race" and v.usable is True


def test_decide_puts_undetermined_on_the_discard_side():
    """判定不能を「使う」側に倒さない。非対称なリスクなので破棄側に倒す。"""
    vs = [leakage.ColumnVerdict(c, "LK-02", "undetermined", "", False)
          for c in CUMULATIVE_RECORD_COLS]
    d = leakage.decide(vs)
    assert d["whitelist"] == []
    assert set(d["discard"]) == set(CUMULATIVE_RECORD_COLS)
    assert set(d["undetermined"]) == set(CUMULATIVE_RECORD_COLS)


# ------------------------------------------------------------------------ LK-04
def test_lk04_to_prerace_removes_every_post_race_column():
    """ホワイトリスト外の post-race 列は1つも残らないこと。

    累積成績8列は LK-09/10/11 で as-of-race を確定させたのでホワイトリストに
    入っている。それ以外は落ちなければならない。
    """
    from nar.transform.prerace import ASOF_RACE_WHITELIST

    df = pd.DataFrame({c: [1] for c in POST_RACE_ALL} | {"距離": [1200]})
    out = to_prerace(df)
    residual = set(out.columns) & POST_RACE_ALL
    assert residual == set(ASOF_RACE_WHITELIST), (
        f"想定外の post-race 列が残っています: {sorted(residual - set(ASOF_RACE_WHITELIST))}")
    assert "距離" in out.columns
    assert_prerace(out)


def test_lk04_whitelist_is_ignored_when_caller_passes_an_empty_one():
    """判定前の既定（空ホワイトリスト）では8列も落ちること。"""
    df = pd.DataFrame({c: [1] for c in POST_RACE_ALL} | {"距離": [1200]})
    out = to_prerace(df, whitelist=frozenset())
    assert set(out.columns) & POST_RACE_ALL == set()


def test_lk04_assert_prerace_raises_when_bypassed():
    df = pd.DataFrame({"着順": [1], "距離": [1200]})
    with pytest.raises(LeakageError, match="着順"):
        assert_prerace(df)


# --------------------------------------------------------------- LK-05（最重要）
def test_lk05_future_poisoning_leaves_past_features_bit_identical(synth_tables):
    """T 以降を乱数で破壊しても、T 以前の特徴量行列が1ビットも変わらないこと。

    ウィンドウ関数の EXCLUDE CURRENT ROW 漏れ、ターゲットエンコーディングの
    全期間集計、収縮パラメータの全期間推定を、この1本でまとめて捕捉する。
    """
    cfg = feature_config()
    entry, race = synth_tables["entry"], synth_tables["race"]

    f1 = build(entry, race, cfg)
    f2 = build(synth.poison_future(entry, CUTOFF, seed=3), race, cfg)

    past = lambda f: f[f["start_ts"] < pd.Timestamp(CUTOFF)].sort_values(
        ["race_id", "horse_no"]).reset_index(drop=True)
    p1, p2 = past(f1), past(f2)

    diff = leakage.future_poisoning_diff(p1, p2)
    assert len(diff) == 0, f"未来を参照している列:\n{diff}"
    assert content_hash(p1) == content_hash(p2)


def test_lk05_diagnostic_names_the_offending_column(synth_tables):
    """診断機能自体が働くこと。故意にリークさせた列を名指しできる。"""
    cfg = feature_config()
    entry, race = synth_tables["entry"], synth_tables["race"]
    f1 = build(entry, race, cfg)

    # 全期間平均（未来込み）を混ぜた列を足す = 典型的な全期間集計リーク
    leaked = f1.copy()
    leaked["bad_global_mean"] = f1["h_winrate_prior"].mean()
    other = f1.copy()
    other["bad_global_mean"] = f1["h_winrate_prior"].mean() + 0.01

    diff = leakage.future_poisoning_diff(leaked, other)
    assert "bad_global_mean" in diff["column"].tolist()


# ------------------------------------------------------------------ LK-06, LK-07
@pytest.mark.slow
def test_lk06_label_shuffle_collapses_to_uniform_baseline(synth_tables):
    """レース内で着順を完全シャッフルすると、予測力が一様分布まで落ちること。

    少しでも予測できてしまう場合、シャッフルしていない情報経路
    （as-of 集計、エンコーディング、CV 分割）にリークがある。
    """
    cfg = feature_config()
    race = synth_tables["race"]
    shuffled = synth.shuffle_labels_within_race(synth_tables["entry"], seed=5)
    feat = build(shuffled, race, cfg).dropna(subset=["h_winrate_prior"])

    # 補完・標準化は学習と同じ prepare を通す。テスト側で中央値補完を書くと、
    # 全行 NaN の列で中央値そのものが NaN になり、損失が NaN になって
    # 「一様分布より良い」判定が意味を失う。
    from nar.train.pipeline import prepare

    cols = [c for c in ASOF_FEATURES if c in feat.columns]
    feat = prepare(feat, cols)

    split = int(len(feat) * 0.7)
    tr = feat.iloc[:split].sort_values(["race_id", "horse_no"])
    va = feat.iloc[split:].sort_values(["race_id", "horse_no"])

    model = ConditionalLogit(l2=1e-2).fit(to_batch(tr, cols))
    b = to_batch(va, cols)
    p = b.flat_predictions(model.predict_proba(b), len(va))

    nll = metrics.race_nll(p, va["is_win"].to_numpy(), va["race_id"].to_numpy())
    uni = metrics.uniform_nll(va["race_id"].to_numpy())
    assert nll >= uni - 0.05, f"シャッフル下で NLL {nll:.4f} < 一様 {uni:.4f}。リーク経路あり。"


# ------------------------------------------------------------------------ LK-08
def test_lk08_injected_leak_column_is_flagged(synth_tables):
    """FX-04: 着順の完全関数である列を混入させ、検出されることを確認する。"""
    poisoned = synth.inject_leak_column(synth_tables["entry"])
    screen = leakage.label_correlation_screen(
        poisoned, exclude=("finish_pos", "time_sec", "popularity", "odds_win"))
    top = screen.iloc[0]
    assert top["column"] == "x_leak" and bool(top["suspect"])


def test_lk08_prerace_gate_rejects_untransformed_frame(synth_tables):
    """to_prerace() 未通過のフレームは、特徴量ビルダーに渡る前に止まる。"""
    df = synth_tables["entry"].head(10).copy()
    df["着順"] = df["finish_pos"]
    with pytest.raises(LeakageError):
        assert_prerace(df)


# --------------------------------------------------------------- 同時刻の別レース
def test_same_instant_races_at_other_tracks_are_not_counted():
    """同じ分に発走する他場のレースは、対象レースの発走時点でまだ終わっていない。

    全順序で「手前」に来るからといって集計に入れると、未来の結果を見たことになる。
    運用側は `start_ts <` で厳密に切っているので、学習側が同時刻を含めると
    そこだけ推論と食い違う（実測で j_starts_prior が 1 ずれた）。
    """
    import numpy as np
    import pandas as pd

    from nar.config import feature_config
    from nar.features.builder import build

    ts = pd.Timestamp("2020-05-05 15:30:00")
    rows = []
    for i, (rid, baba) in enumerate((("A", 20), ("B", 21))):
        for k in range(4):
            rows.append({
                "race_id": rid, "horse_no": k + 1, "waku": k + 1,
                "horse_sk": f"H{i}{k}", "jockey_sk": "J0",
                "trainer_sk": f"T{k}", "sire_sk": f"S{k}",
                "race_date": ts.date(), "start_ts": ts,
                "baba_code": baba, "distance": 1200.0,
                "finish_pos": float(k + 1), "is_win": int(k == 0),
                "time_sec": 72.0 + k, "turn": "右", "baba_condition": "良",
            })
    entry = pd.DataFrame(rows)
    race = (entry[["race_id", "race_date", "start_ts", "baba_code", "distance"]]
            .drop_duplicates("race_id")
            .assign(race_no=1, surface="ダ", turn="右", class_level=1,
                    prize_yen=1_000_000, n_runners=4))
    feat = build(entry, race, feature_config())
    # 同じ騎手が両レースに乗っている想定。どちらの行も相手を数えてはいけない
    assert (feat["j_starts_prior"] == 0).all(), (
        f"同時刻の別レースを数えています: {feat['j_starts_prior'].tolist()}")
    assert (feat["h_starts_prior"] == 0).all()


def test_shrinkage_prior_depends_only_on_the_race(cfg=None):
    """収縮の事前確率はレース内で完結すること。

    データセット全体の累積勝率を事前確率にすると、推論時に渡せる部分集合からは
    再現できず、学習と推論で収縮勝率がずれる（実測で 4 桁目から食い違った）。
    履歴の一部だけを渡しても、同じレースの特徴量が変わらないことを固定する。
    """
    import pandas as pd

    from nar.config import feature_config
    from nar.features.builder import build

    from nar import synth

    t = synth.generate(synth.SynthConfig(n_races=1500, seed=4))
    entry, race = t["entry"], t["race"]
    target = entry["race_id"].value_counts().index[len(entry["race_id"].unique()) // 2]
    ts = entry.loc[entry["race_id"] == target, "start_ts"].iloc[0]

    full = build(entry, race, feature_config())
    # 対象レースの馬・騎手・調教師・種牡馬だけに絞った履歴（運用時に渡せる形）
    card = entry[entry["race_id"] == target]
    keys = {c: set(card[c]) for c in ("horse_sk", "jockey_sk", "trainer_sk", "sire_sk")}
    keep = entry["race_id"].eq(target)
    for col, vals in keys.items():
        keep |= entry[col].isin(vals) & (entry["start_ts"] <= ts)
    subset = entry[keep]
    partial = build(subset, race[race["race_id"].isin(set(subset["race_id"]))],
                    feature_config())

    a = full[full["race_id"] == target].set_index("horse_no").sort_index()
    b = partial[partial["race_id"] == target].set_index("horse_no").sort_index()
    for col in ("j_winrate_shrunk", "t_winrate_shrunk", "s_winrate_shrunk"):
        x, y = a[col].astype(float), b[col].astype(float)
        both = x.notna() & y.notna()
        assert (x[both] == y[both]).all(), (
            f"{col} が履歴の絞り込みで変わっています: {x[both].tolist()} vs {y[both].tolist()}")
