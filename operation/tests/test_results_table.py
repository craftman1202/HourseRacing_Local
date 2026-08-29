"""当日成績表の解析。ライブ層を埋める唯一の経路（`entry_result_live`）。"""

from __future__ import annotations

import pandas as pd

from narops.results_table import parse

# 実ページ（船橋 2026-08-28 1R）と同じ構造の抜粋
HTML = """<table>
  <tr><th>着順</th><th>枠</th><th>馬番</th><th>馬名</th><th>騎手（所属）</th>
      <th>タイム</th><th>単勝 オッズ</th></tr>
  <tr><td>1</td><td>2</td><td>2</td><td>レオセラフィム</td>
      <td>實川純 （船橋）</td><td>1:17.9</td><td>13.4</td></tr>
  <tr><td>2</td><td>3</td><td>3</td><td>ダイレンジャー</td>
      <td>ゴンサ （船橋）</td><td>1:18.8</td><td>14.8</td></tr>
  <tr><td>取</td><td>5</td><td>5</td><td>トリケシウマ</td>
      <td>山中悠 （船橋）</td><td></td><td></td></tr>
</table>"""


def test_time_is_seconds_not_the_printed_string():
    """`1:17.9` は 77.9 秒。分を落とすと速度指数が全部狂う。"""
    out = parse(HTML)
    assert out.loc[out["horse_no"] == 2, "time_sec"].iloc[0] == 77.9
    assert out.loc[out["horse_no"] == 3, "time_sec"].iloc[0] == 78.8


def test_scratched_horses_are_not_history():
    """取消・除外は走っていない。履歴に入れると出走数が水増しされる。"""
    out = parse(HTML)
    assert 5 not in set(out["horse_no"]), "取消馬が履歴に入りました"
    assert list(out["horse_no"]) == [2, 3]


def test_is_win_marks_only_the_winner():
    out = parse(HTML)
    assert list(out["is_win"]) == [1, 0]


def test_unfinished_race_returns_empty_not_an_error():
    """開催中は必ず未確定のレースがある。例外にすると当日更新が丸ごと止まる。"""
    out = parse("<table><tr><th>発走時刻</th></tr><tr><td>15:20</td></tr></table>")
    assert isinstance(out, pd.DataFrame) and out.empty


def test_payout_table_is_not_mistaken_for_results():
    """払戻表にも数字が並ぶ。着順・馬番の両方がある表だけを本体とみなす。"""
    out = parse("""<table><tr><td>単勝</td><td>2</td><td>1,340円</td></tr></table>""")
    assert out.empty
