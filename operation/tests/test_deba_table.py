"""当日出馬表（DebaTable）の解析。

フィクスチャは 2026-08-26 園田5R の実 HTML。同じレースの月次ファイルと
全項目が一致することを確認済みで、その一致をここで固定する。
出馬表の解析が崩れると、推論だけが学習と違う値を見ることになる（SK-01）。
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from narops import deba_table
from narops.errors import InsufficientData

FIXTURE = Path(__file__).parent / "fixtures" / "deba_sonoda_20260826_5r.html.gz"
RACE_ID = "272026082605"
RACE_DATE = "2026-08-26"


@pytest.fixture(scope="module")
def html() -> str:
    return gzip.decompress(FIXTURE.read_bytes()).decode("utf-8")


@pytest.fixture(scope="module")
def card(html) -> pd.DataFrame:
    return deba_table.parse(html, RACE_ID, RACE_DATE)


def test_every_runner_is_extracted(card):
    assert len(card) == 10
    assert card["horse_no"].tolist() == list(range(1, 11))


def test_race_list_page_is_not_the_entry_card():
    """当初 RaceList を出馬表として叩いていた。あれは目次で出走馬が載っていない。"""
    from narops.nar_source import DEBA_TABLE, RACE_LIST

    assert DEBA_TABLE.endswith("DebaTable")
    assert RACE_LIST != DEBA_TABLE


def test_waku_is_carried_across_rowspan(card):
    """1枠に2頭入るとセルが束ねられ、2頭目の枠番セルが存在しない。

    取りこぼすと相対枠順が欠ける。
    """
    assert card["waku"].notna().all()
    assert card.loc[card["horse_no"] == 8, "waku"].iloc[0] == 7
    assert card.loc[card["horse_no"] == 10, "waku"].iloc[0] == 8


def test_birth_date_is_reconstructed_from_age_and_month_day(card):
    """ページには生年が無い。日本の競走馬の年齢は「開催年 − 生年」なので復元できる。

    ここが1年ずれると horse_sk が変わり、過去成績が全部欠けて初出走扱いになる。
    """
    row = card[card["horse_name"] == "イービジョンスター"].iloc[0]
    assert row["age"] == 3 and row["birth_ymd"] == "20230401"
    assert (card["birth_ymd"].str.len() == 8).all()


def test_horse_sk_matches_the_learning_side_definition(card):
    """学習側と同じキーが作れること。ここが崩れると履歴と結合できない。"""
    from nar.transform.keys import horse_sk

    row = card[card["horse_name"] == "イービジョンスター"].iloc[0]
    assert horse_sk(row["horse_name"], row["birth_ymd"], row["sire_name"]) == (
        "e8951a2d412e9439")


def test_all_eight_cumulative_columns_are_present(card):
    """学習で使う as-of-race の8列が推論時にも取れること（train-serving skew 防止）。"""
    from nar.transform.prerace import CUMULATIVE_RECORD_COLS

    for col in CUMULATIVE_RECORD_COLS:
        assert col in card.columns, f"{col} が出馬表から取れていません"
    assert (card["全成績"].str.match(r"^\d+-\d+-\d+-\d+$")).all()


def test_jockey_pair_record_comes_from_the_second_row(card):
    """`騎手成績` は「この馬とこの騎手」の成績で、負担重量と同じセルに入っている。"""
    row = card[card["horse_name"] == "モモロイヤル"].iloc[0]
    assert row["騎手成績"] == "4-1-0-0"
    assert row["weight_carried"] == "56.0"


def test_best_times_keep_their_going_label(card):
    row = card[card["horse_name"] == "クオンタムゲート"].iloc[0]
    assert row["最高タイム"] == "1:32.5"
    assert row["最高タイム良馬場"] == "良1:33.0"


def test_odds_and_popularity_are_extracted(card):
    assert card["odds_win"].notna().all()
    assert card["popularity"].tolist() == [8, 1, 6, 3, 10, 9, 7, 5, 4, 2]


def test_geldings_are_parsed(card):
    """性別表記は 牡/牝/セン。「セン」を1文字クラスで書くと「ン」が漏れる。"""
    row = card[card["horse_name"] == "インジェクション"].iloc[0]
    assert row["sex"] == "セン" and row["age"] == 6


def test_empty_page_raises_instead_of_returning_an_empty_card():
    """空を返すと「出走馬0頭のレース」として下流が動いてしまう。"""
    with pytest.raises(InsufficientData, match="出走馬の行がありません"):
        deba_table.parse("<html><body>準備中</body></html>", RACE_ID, RACE_DATE)


def test_post_race_columns_are_dropped_but_the_confirmed_eight_survive(card):
    from narops.shared import to_prerace
    from nar.transform.prerace import ASOF_RACE_WHITELIST, assert_prerace

    pre = to_prerace(card)
    assert_prerace(pre)
    for col in ASOF_RACE_WHITELIST:
        if col in card.columns:
            assert col in pre.columns


# ------------------------------------------------------------ NAR のエラー応答
def test_nar_error_page_is_detected_not_parsed():
    """NAR は存在しないページにも HTTP 200 でエラー HTML を返す。

    そのまま下流に渡すと「解析できない」という別の顔をした失敗になり、
    原因が読めない。取得の時点で止める。
    """
    from datetime import date
    from types import SimpleNamespace

    from narops.nar_source import NarFetchError, NarSource

    body = "<html><body><p>404 not Found<br />お探しのページは見つかりません。</p></body></html>"
    src = NarSource(user_agent="t")
    src._client = SimpleNamespace(
        fetch=lambda url, params: SimpleNamespace(content=body.encode("utf-8")))
    with pytest.raises(NarFetchError, match="エラーページ"):
        src.fetch_entry_card("232026082811", 23, date(2026, 8, 28), 11)


def test_read_html_is_given_a_buffer_not_a_raw_string():
    """生の文字列を渡すと、新しい pandas はファイルパスとして開こうとする。

    `FileNotFoundError` になり、しかも本文がそのままエラーに載る。
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "narops"
    for f in src.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "read_html(" in code:
                assert "StringIO" in code, f"{f.name}:{i} が生の文字列を渡しています"


REAL_HEADER = """<div class="raceTitle">馬い！淡路洲本農園玉ねぎ記念Ｃ３一
 ダート 1500ｍ（左） 天候：曇 馬場：良 サラブレッド系 一般
 ＊電話投票コード：32# 賞金 1着900,000円 2着351,000円</div>"""


def test_race_conditions_come_from_the_page_not_from_defaults():
    """推論は class_level=1 / distance=1200 の固定値で走っていた。

    race_schedule は距離もクラスも持たず、`_schedule_row` が既定値を
    埋めていた。**学習データに class_level=1 は1件も無い**
    （2 が 77.6%、4 が 19.8%、5 が 2.6%）ので、モデルが一度も見ていない
    値を毎回渡していたことになる。
    """
    from narops.deba_table import parse_race_header

    got = parse_race_header(REAL_HEADER)
    assert got["distance"] == 1500
    assert got["class_level"] == 2, "「一般」は水準2。既定値の1ではない"
    assert got["prize_yen"] == 900_000.0
    assert got["surface"] == "ダ"
    assert got["turn"] == "左"


def test_class_level_only_takes_values_the_model_was_trained_on():
    """学習の `class_name` は5値しか取らない（普通/一般/特別/重賞/準重賞）。

    水準は 2 / 4 / 5 のみ。1 と 3 は学習データに1件も無いので、
    当日ページの自由文からこの5値へ寄せる。
    """
    from narops.deba_table import parse_race_header

    cases = {
        # 船橋: 条件が距離表記より後ろ
        "馬い！記念Ｃ３一 ダート 1500ｍ（左） サラブレッド系 一般": ("一般", 2),
        # 園田: 「特別」がレース名側（距離表記より前）に出る
        "デュランタ賞Ｃ２ ３歳以上特別 ダート 1700ｍ（右） サラブレッド系 定量": ("特別", 4),
        # どれにも当たらない条件戦は「普通」。学習の最頻値と同じ水準
        "Ｃ３ ３歳以上 ダート 820ｍ（右） サラブレッド系 3歳以上 定量": ("普通", 2),
        "黒潮盃ＪpnIII ダート 1800ｍ（右） サラブレッド系": ("重賞", 5),
    }
    for text, (name, level) in cases.items():
        got = parse_race_header(f'<div class="raceTitle">{text}</div>')
        assert (got["class_name"], got["class_level"]) == (name, level), text
        assert got["class_level"] in (2, 4, 5), "学習に無い水準です"


def test_missing_header_returns_empty_so_the_caller_can_fail():
    """読めないときに固定値へ落とさない。呼び出し側で失敗させる。"""
    from narops.deba_table import parse_race_header

    assert parse_race_header("<div>出馬表</div>") == {}
