

def test_odds_url_is_a_page_that_exists_on_nar():
    """OddsWin という名前のページは NAR に無い。

    存在しない名前を叩いていたため、全レースがエラーページを返し、
    オッズが一度も取れていなかった。出馬表ページ内のリンクが指す名前を使う。
    """
    from narops.nar_source import ODDS_URL

    assert ODDS_URL.endswith("OddsTanFuku"), ODDS_URL


def test_parse_win_odds_reads_the_tanfuku_table():
    """単勝・複勝が並ぶ表から単勝だけを取る。複勝を拾うと賭け金が狂う。"""
    from narops.nar_source import parse_win_odds

    html = """<table>
      <tr><th>枠</th><th>馬番</th><th>馬名</th><th>単勝 オッズ</th>
          <th>複勝オッズ</th></tr>
      <tr><td>1</td><td>1</td><td>パラダイスビート</td><td>19.5</td><td>3.8-</td></tr>
      <tr><td>2</td><td>2</td><td>フルーツパフェ</td><td>5.4</td><td>1.4-</td></tr>
    </table>"""
    out = parse_win_odds(html, "192026082806")
    assert list(out["horse_no"]) == [1, 2]
    assert list(out["odds_win"]) == [19.5, 5.4], "複勝を拾っています"
