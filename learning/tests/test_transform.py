"""TR-01..11: 変換・キー設計層。"""

from __future__ import annotations

import pandas as pd
import pytest

from nar.errors import UnknownTrackError
from nar.transform import keys, payout
from nar.transform.keys import KNOWN_BABA_CODES, TrackMaster, horse_sk, make_race_id


# --------------------------------------------------------------------- TR-01/02
def test_tr01_race_id_format_and_example():
    rid = make_race_id(20, "2026-08-24", 11)
    assert rid == "202026082411"
    keys.assert_race_id_format(pd.Series([rid]))


def test_tr01_malformed_race_id_is_rejected():
    with pytest.raises(ValueError, match="12桁"):
        keys.assert_race_id_format(pd.Series(["2020260824"]))


def test_tr02_race_id_is_unique_per_race(entry):
    per_race = entry.groupby("race_id")[["baba_code", "race_date"]].nunique()
    assert (per_race == 1).all().all(), "同一 race_id に複数の場・日付が混ざっています"


# --------------------------------------------------------------------- TR-03/04
def test_tr03_unknown_track_raises_instead_of_null():
    m = TrackMaster()
    with pytest.raises(UnknownTrackError, match="未知の競馬場名"):
        m.code("存在しない場")


def test_tr04_master_covers_known_baba_codes():
    """現在開催中の14場。設計書の13コードに水沢(11)が加わる。

    当初 水沢=10（盛岡と同じ）と書いていたが、月別開催日程の実測値は 11。
    """
    known = {3, 10, 11, 18, 19, 20, 21, 22, 23, 24, 27, 28, 31, 32, 36}
    assert set(KNOWN_BABA_CODES.values()) == known


def test_tr04_each_track_has_its_own_code():
    """1コード1場。同一県の2場を同じコードにすると race_id が衝突する。

    盛岡/水沢（岩手）と 園田/姫路（兵庫）は同じ主催者だが別コード。
    ここを共有させると、両場が同日同レース番号で開催したときに
    別レースが同じ race_id を持つ。
    """
    m = TrackMaster()
    assert m.code("帯広ば") == 3        # ばんえい
    assert m.code("大井") == 20
    assert m.code("盛岡") != m.code("水沢")
    assert m.code("園田") != m.code("姫路")
    assert len(set(KNOWN_BABA_CODES.values())) == len(KNOWN_BABA_CODES)


# --------------------------------------------------------------------- TR-05/07
def test_tr05_horse_sk_is_stable_across_files():
    a = horse_sk("テスト馬", "2018-04-01", "父馬A")
    b = horse_sk(" テスト馬 ", "2018-04-01", "父馬A")
    assert a == b, "空白差でキーが変わってはいけない"


def test_tr07_same_name_different_pedigree_separates():
    a = horse_sk("同名馬", "2018-04-01", "父A")
    b = horse_sk("同名馬", "2019-04-01", "父B")
    assert a != b, "同名別馬が同一キーになっています"


def test_horse_alias_links_renamed_horse():
    df = pd.DataFrame({
        "馬名": ["旧名", "旧名", "新名"],
        "生年月日": ["2018-04-01"] * 3,
        "父馬名": ["父A"] * 3,
        "母馬名": ["母A"] * 3,
        "race_date": pd.to_datetime(["2021-01-01", "2021-06-01", "2022-01-01"]),
    })
    df = keys.add_horse_sk(df)
    alias = keys.build_horse_alias(df)
    renamed = alias[alias["is_rename"]]
    assert len(renamed) == 1
    assert renamed.iloc[0]["alias_name"] == "新名"
    assert renamed.iloc[0]["canonical_name"] == "旧名"


def test_tr06_identity_audit_flags_over_long_careers(entry):
    from nar.eda.questions import identity_audit

    audit = identity_audit(entry)
    assert audit["over_12y_pct"] <= 1.0, (
        f"現役期間12年超が {audit['over_12y_pct']}%。名寄せの結合条件が緩すぎます。")
    assert audit["n_same_day_duplicate_starts"] == 0


def test_person_keys_keep_both_granularities():
    df = pd.DataFrame({"騎手名": ["山田 太郎"], "所属": ["大井"]})
    out = keys.person_keys(df, "騎手名", "所属", "jockey")
    assert out["jockey_sk"].iloc[0] == "山田太郎"
    assert out["jockey_affil_sk"].iloc[0] == "山田太郎@大井"


# ------------------------------------------------------------------ TR-08/09/10
@pytest.fixture
def wide_payout():
    """同着で複勝が3行になるレース（FX-07 相当）。"""
    return pd.DataFrame([{
        "race_id": "202026082411",
        "単勝1_組番": "5", "単勝1_払戻金": "830", "単勝1_人気": "4",
        "複勝1_組番": "5", "複勝1_払戻金": "240", "複勝1_人気": "4",
        "複勝2_組番": "1", "複勝2_払戻金": "150", "複勝2_人気": "1",
        "複勝3_組番": "8", "複勝3_払戻金": "310", "複勝3_人気": "6",
        "馬連1_組番": "1-5", "馬連1_払戻金": "2340", "馬連1_人気": "9",
        "3連単1_組番": "5-1-8", "3連単1_払戻金": "45600", "3連単1_人気": "112",
        "ワイド1_組番": "", "ワイド1_払戻金": "", "ワイド1_人気": "",
    }])


def test_tr08_unpivot_preserves_row_count_and_total(wide_payout):
    long = payout.unpivot(wide_payout)
    assert len(long) == 6, "非NULL の券種セル数と一致しなければならない"
    payout.assert_roundtrip(wide_payout, long)


def test_tr09_dead_heat_sequence_numbered_without_duplicates(wide_payout):
    long = payout.unpivot(wide_payout)
    fuku = long[long["bet_type"] == "複勝"].sort_values("dead_heat_seq")
    assert fuku["dead_heat_seq"].tolist() == [1, 2, 3]
    payout.assert_no_duplicate_combination(long)


def test_tr09_combination_split_handles_multi_leg_bets(wide_payout):
    long = payout.unpivot(wide_payout)
    trifecta = long[long["bet_type"] == "3連単"].iloc[0]
    assert (trifecta["comb_1"], trifecta["comb_2"], trifecta["comb_3"]) == (5, 1, 8)
    win = long[long["bet_type"] == "単勝"].iloc[0]
    assert pd.isna(win["comb_2"]) and pd.isna(win["comb_3"])


def test_tr10_units_are_correct(entry, race):
    assert entry["time_sec"].dtype.kind == "f"
    assert race["distance"].dtype.kind in "iu"
    assert (race["distance"] > 0).all(), "ゼロ距離が存在します"
    assert (entry["time_sec"] > 0).all(), "負またはゼロのタイムが存在します"


def test_tr10_payout_is_positive_integer_yen(payout):
    assert payout["payout_yen"].dtype.kind in "iu"
    assert (payout["payout_yen"] >= 100).all(), "元返し未満の払戻があります"


# ------------------------------------------------------------------------ TR-11
def test_tr11_banei_excluded_from_training_set(race):
    from nar import synth
    from nar.config import feature_config
    from nar.features.builder import build

    with_banei = synth.generate(synth.SynthConfig(n_races=300, seed=3, include_banei=True))
    assert (with_banei["entry"]["baba_code"] == 3).any(), "テスト前提: ばんえいが含まれること"

    feat = build(with_banei["entry"], with_banei["race"], feature_config())
    assert (feat["baba_code"] == 3).sum() == 0, "学習データセットにばんえいが残っています"
    # 廃止ばんえい3場も同様に除く。実データには 北見ば/岩見ば/旭川ば が実在する
    assert feat["baba_code"].isin([1, 2, 4]).sum() == 0, "廃止ばんえい場が残っています"
