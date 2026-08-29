"""実データのパース検証。

フィクスチャは実 NAR 月次ファイル（2026-07）から作ったゴールデン ZIP（FX-01）。
合成データでは通ってしまう「実ファイル固有の落とし穴」を固定する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nar.ingest.unzip import decode, extract
from nar.transform import bronze, silver
from nar.transform.keys import TrackMaster
from nar.transform.prerace import CUMULATIVE_RECORD_COLS, assert_prerace, to_prerace
from nar.transform.schema_guard import EXPECTED_COLUMNS, schema_of

FIXTURE = Path(__file__).parent / "fixtures" / "golden_monthly_202607.zip"


@pytest.fixture(scope="module")
def raw() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def tables(raw) -> dict[str, pd.DataFrame]:
    return {t.kind: t.frame for t in bronze.build_from_zip(raw, "2026-07")}


@pytest.fixture(scope="module")
def silver_tables(tables) -> dict[str, pd.DataFrame]:
    m = TrackMaster()
    race = silver.build_race(tables["race"], m)
    entry = silver.attach_start_ts(silver.build_entry(tables["entry"], m), race)
    payout = silver.build_payout(tables["payout"], m)
    return {"race": race, "entry": entry, "payout": payout}


# ------------------------------------------------------------------ エンコーディング
def test_monthly_files_are_utf8_bom_not_cp932(raw):
    """実ファイルは UTF-8 BOM。設計書の想定（CP932 優先）とは違う。

    cp932 は UTF-8 バイト列を例外なく復号してしまい、文字化けしたまま
    気付けない。BOM を先に見る実装であることを固定する。
    """
    for csvfile in extract(raw):
        assert csvfile.codec == "utf-8-sig", f"{csvfile.name} が {csvfile.codec}"
        assert not csvfile.replaced


def test_cp932_would_silently_corrupt_utf8():
    """cp932 優先だと壊れることを明示的に記録する（回帰防止）。"""
    text = "競馬場,競走年月日"
    utf8_bytes = text.encode("utf-8")
    mojibake = utf8_bytes.decode("cp932")       # 例外にならない
    assert mojibake != text, "cp932 が UTF-8 を素通ししました"
    assert decode(b"\xef\xbb\xbf" + utf8_bytes)[0].lstrip("﻿") == text


def test_japanese_headers_are_readable(tables):
    assert "競馬場" in tables["race"].columns
    assert "馬名" in tables["entry"].columns
    assert "単勝組番" in tables["payout"].columns


# ------------------------------------------------------------------ SG-01
def test_column_counts_match_the_specification(tables):
    """レース一覧66・出馬表36・払戻54。公式説明書からピン留めした定数と一致。"""
    assert tables["race"].shape[1] == EXPECTED_COLUMNS["race"] == 66
    assert tables["entry"].shape[1] == EXPECTED_COLUMNS["entry"] == 36
    assert tables["payout"].shape[1] == EXPECTED_COLUMNS["payout"] == 54


def test_schema_hash_is_stable(raw):
    a = [schema_of(c.text, kind=bronze.classify(c.name)) for c in extract(raw)]
    b = [schema_of(c.text, kind=bronze.classify(c.name)) for c in extract(raw)]
    assert [x.hash() for x in a] == [y.hash() for y in b]


# ------------------------------------------------------------------ 走破タイム
def test_time_is_parsed_as_mssf_not_raw_number(silver_tables):
    """`1143` は 1143 秒ではなく 74.3 秒。

    素の数値として読むと速度指数が桁ごと壊れる。
    """
    e = silver_tables["entry"]
    finished = e[e["time_sec"].notna()]
    assert len(finished) > 0
    assert finished["time_sec"].max() < 400, "秒に変換されていません"
    assert finished["time_sec"].min() > 20


@pytest.mark.parametrize("raw_value,expected", [
    ("1143", 74.3), ("591", 59.1), ("2045", 124.5), ("1505", 110.5),
    ("1:14.3", 74.3), ("", None),
])
def test_time_parsing_cases(raw_value, expected):
    got = silver._parse_time(pd.Series([raw_value])).iloc[0]
    if expected is None:
        assert pd.isna(got)
    else:
        assert got == pytest.approx(expected, abs=1e-6)


# ------------------------------------------------------------------ 払戻
def test_all_bet_types_are_extracted(silver_tables):
    """組番が複数列に分かれている実構造を正しく読めていること。"""
    p = silver_tables["payout"]
    kinds = set(p["bet_type"])
    assert {"単勝", "複勝", "馬連", "馬単", "ワイド", "3連複", "3連単"} <= kinds


def test_multi_leg_bets_have_multiple_legs(silver_tables):
    p = silver_tables["payout"]
    for bet, n_legs in (("単勝", 1), ("馬連", 2), ("ワイド", 2), ("3連単", 3)):
        rows = p[p["bet_type"] == bet]
        assert len(rows) > 0, f"{bet} が1件も取れていません"
        legs = rows[["comb_1", "comb_2", "comb_3"]].notna().sum(axis=1)
        assert (legs == n_legs).all(), f"{bet} の脚数が {set(legs)}（期待 {n_legs}）"


def test_payouts_are_positive_yen(silver_tables):
    p = silver_tables["payout"]
    assert (p["payout_yen"] >= 100).all(), "元返し未満の払戻があります"


def test_fukusho_has_dead_heat_sequence(silver_tables):
    """複勝は1レース複数行。dead_heat_seq が 1..n で振られる。"""
    p = silver_tables["payout"]
    fuku = p[p["bet_type"] == "複勝"]
    per_race = fuku.groupby("race_id")["dead_heat_seq"].apply(list)
    for seq in per_race:
        assert seq == list(range(1, len(seq) + 1))


# ------------------------------------------------------------------ キー
def test_race_id_format(silver_tables):
    from nar.transform.keys import assert_race_id_format

    assert_race_id_format(silver_tables["race"]["race_id"])
    assert_race_id_format(silver_tables["entry"]["race_id"])


def test_entry_joins_to_race(silver_tables):
    e, r = silver_tables["entry"], silver_tables["race"]
    assert e["race_id"].isin(set(r["race_id"])).all(), "race に無い race_id があります"


def test_start_ts_is_attached(silver_tables):
    e = silver_tables["entry"]
    assert e["start_ts"].notna().all(), "発走時刻が付いていません（as-of 順序に必須）"


def test_horse_sk_is_stable_for_same_horse(silver_tables):
    """同じ馬名+生年月日+父なら同一キー。"""
    e = silver_tables["entry"]
    grouped = e.groupby(["horse_name", "birth_ymd", "sire_sk"])["horse_sk"].nunique()
    assert (grouped == 1).all()


def test_banei_is_present_in_silver_but_excluded_later(silver_tables):
    """silver にはばんえいを残す。除外は特徴量構築の責務（TR-11）。"""
    assert (silver_tables["race"]["baba_code"] == 3).any()


# ------------------------------------------------------------------ 結果列
def test_scratched_horses_have_null_finish_position(silver_tables):
    """取消・除外は着順が数値にならず NaN。0 埋めしない。"""
    e = silver_tables["entry"]
    assert e["finish_pos"].isna().any() or len(e) < 100
    assert not (e["finish_pos"] == 0).any(), "着順 0 は存在しません"


def test_each_race_has_exactly_one_winner(silver_tables):
    e = silver_tables["entry"]
    wins = e.groupby("race_id")["is_win"].sum()
    assert (wins == 1).all(), f"1着が1頭でないレース: {wins[wins != 1].to_dict()}"


# ------------------------------------------------------------------ リーク列
def test_cumulative_columns_are_present_in_silver(silver_tables):
    """8列は silver には残る。時点性の判定材料として必要。"""
    e = silver_tables["entry"]
    present = [c for c in CUMULATIVE_RECORD_COLS if c in e.columns]
    assert len(present) >= 6, f"累積成績列がほとんど残っていません: {present}"


def test_prerace_drops_post_race_columns_but_keeps_the_confirmed_ones(silver_tables):
    """着順系は消える。累積成績8列は as-of-race 確定後なので残る（LK-04）。"""
    from nar.transform.prerace import ASOF_RACE_WHITELIST

    e = silver_tables["entry"]
    pre = to_prerace(e)
    assert_prerace(pre)
    for c in ("着順", "タイム", "人気"):
        assert c not in pre.columns
    for c in CUMULATIVE_RECORD_COLS:
        present = c in e.columns
        if present and c in ASOF_RACE_WHITELIST:
            assert c in pre.columns, f"{c} は使用可のはずが落ちています"
        elif present:
            assert c not in pre.columns, f"{c} は未確定のはずが残っています"


def test_prerace_with_empty_whitelist_still_drops_all_eight(silver_tables):
    """判定前の既定に戻せば8列も落ちること。"""
    pre = to_prerace(silver_tables["entry"], whitelist=frozenset())
    for c in CUMULATIVE_RECORD_COLS:
        assert c not in pre.columns


def test_cumulative_record_format_is_parseable(silver_tables):
    """`9-22-14-85` 形式。LK-02/03 の判定に使える形であること。"""
    from nar.eda.leakage import parse_record

    e = silver_tables["entry"]
    if "全成績" not in e.columns:
        pytest.skip("全成績 列がありません")
    parsed = parse_record(e["全成績"])
    assert parsed["starts"].notna().any(), "総出走数を取り出せません"
    assert (parsed["starts"].dropna() >= 0).all()


# ------------------------------------------------------------------ bronze 分割
def test_odds_zip_has_three_parts_with_distinct_part_ids():
    """オッズは1か月あたり `_01_` 〜 `_03_` の3本。

    固定名 part.parquet で書くと相互に上書きされ 2/3 が消える（実際に消えた）。
    """
    assert bronze.part_of("202602_01_odds.csv") == "01"
    assert bronze.part_of("202602_03_odds.csv") == "03"
    assert bronze.part_of("202607_racelist.csv") == "00"


def test_bronze_write_read_roundtrips_multiple_parts(tmp_path):
    from nar.io.store import Store

    store = Store(f"file://{tmp_path}")
    for part, n in (("01", 2), ("02", 3), ("03", 1)):
        frame = pd.DataFrame({"a": [part] * n})
        bronze.write(store, bronze.BronzeTable("odds", "2026-02", frame, "utf-8-sig",
                                               "src", part))
    got = bronze.read(store, "odds", "2026-02")
    assert len(got) == 6, "分割が上書きされています"
    assert set(got["a"]) == {"01", "02", "03"}


def test_ym_is_taken_from_file_key_not_its_last_segment():
    """odds の file_key は monthly/odds/2026-02/01。末尾を取ると `01` になる。"""
    import re

    for key, expected in (("monthly/race/2026-07", "2026-07"),
                          ("monthly/odds/2026-02/01", "2026-02")):
        assert re.search(r"\d{4}-\d{2}", key).group(0) == expected


# ------------------------------------------------------------------ 競馬場マスタ
def test_track_master_parses_name_to_code_rows():
    from nar.ingest.schedule import parse_codes

    html = """
    <table>
      <tr><td>曜日</td><td>月</td></tr>
      <tr class="topOfArea"><td>帯広ば</td>
          <td><a href="/KeibaWeb/TodayRaceInfo/RaceList?k_raceDate=2026%2F06%2F01&amp;k_babaCode=3">☆</a></td>
          <td><a href="/x?k_babaCode=3">☆</a></td></tr>
      <tr><td>水沢</td><td><a href="/x?k_babaCode=11">●</a></td></tr>
    </table>"""
    assert parse_codes(html) == {"帯広ば": 3, "水沢": 11}


def test_track_master_rejects_one_name_with_two_codes():
    """名前とコードの多対多を黙って後勝ちにすると race_id が壊れる。"""
    from types import SimpleNamespace

    from nar.errors import UnknownTrackError
    from nar.ingest.schedule import build_track_master

    pages = ['<tr><td>大井</td><td><a href="?k_babaCode=20">o</a></td></tr>'.encode(),
             '<tr><td>大井</td><td><a href="?k_babaCode=21">o</a></td></tr>'.encode()]
    client = SimpleNamespace(
        fetch=lambda url, params: SimpleNamespace(content=pages[params["k_month"] - 1]))
    with pytest.raises(UnknownTrackError, match="複数コード"):
        build_track_master(client, [(2000, 1), (2000, 2)])


def test_known_codes_had_two_collisions_that_the_generated_master_fixes():
    """回帰防止: 水沢=10・姫路=27 と書いていた時期がある。

    盛岡と水沢、園田と姫路が同一コードになり、同日開催で race_id が衝突した。
    """
    from nar.transform.keys import KNOWN_BABA_CODES as K

    assert K["盛岡"] != K["水沢"], "盛岡と水沢が同一コードです"
    assert K["園田"] != K["姫路"], "園田と姫路が同一コードです"
    assert len(set(K.values())) == len(K), "コードが重複しています"


def test_all_four_banei_tracks_are_excluded():
    """ばんえいは帯広だけではない（北見ば/岩見ば/旭川ば が 1998-2006 に実在）。"""
    from nar.config import feature_config

    assert set(feature_config().exclude_baba_codes) >= {1, 2, 3, 4}


# ------------------------------------------------------------------ 最高タイム良馬場
def test_best_time_on_good_going_carries_a_going_label(silver_tables):
    """実データの `最高タイム良馬場` は `良1:33.2` 形式。

    ラベルを剥がさないと全行 NaN になり、列がまるごと黙って消える。
    """
    e = silver_tables["entry"]
    if "最高タイム良馬場" not in e.columns:
        pytest.skip("列がありません")
    filled = e["最高タイム良馬場"].astype(str).str.strip()
    filled = filled[~filled.isin(["", "nan", "None"])]
    if filled.empty:
        pytest.skip("値のある行がありません")
    assert filled.str.match(r"^(良|稍重|重|不良)").all()
    assert silver._parse_time(filled).notna().all(), "ラベル付きが秒に直せていません"


@pytest.mark.parametrize("raw_value,expected", [
    ("良1:33.2", 93.2), ("稍重1:27.7", 87.7), ("不良2:04.5", 124.5),
])
def test_time_parsing_strips_known_going_labels(raw_value, expected):
    assert silver._parse_time(pd.Series([raw_value])).iloc[0] == pytest.approx(expected)


def test_time_parsing_does_not_strip_unknown_prefixes():
    """何でも剥がすと壊れた値を通してしまう。既知ラベルだけ剥がす。"""
    assert pd.isna(silver._parse_time(pd.Series(["XX1:33.2"])).iloc[0])
