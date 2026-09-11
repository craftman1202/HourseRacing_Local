

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


# ------------------------------------------------------------------ 場コード
def test_baba_code_matches_the_learning_side_single_source_of_truth():
    """学習側 `nar.transform.keys.KNOWN_BABA_CODES` と1文字も違わないこと。

    以前はここに同じ辞書を手書きで複製していた。学習側で 水沢=10→11、
    姫路=27→28 の誤りが修正されたとき、運用側だけ古い値のまま取り残され、
    水沢の当日レースが盛岡と同じ baba_code で処理される事故になった
    （§ 水沢推論停止、2026-09）。辞書を2つ持つ限りこの種の乖離は必ず起こる。
    """
    from nar.transform.keys import KNOWN_BABA_CODES

    from narops.nar_source import BABA_CODE

    assert BABA_CODE is KNOWN_BABA_CODES, (
        "narops.nar_source.BABA_CODE は学習側 KNOWN_BABA_CODES を直接指す"
        "べきです。独立した辞書に戻すと、学習側だけ直った修正が運用側に"
        "反映されないまま気付かれません。")


def test_no_two_tracks_share_a_baba_code():
    """場コードに重複がないこと（衝突すると race_id が衝突する）。"""
    from narops.nar_source import BABA_CODE

    codes = list(BABA_CODE.values())
    dups = {c for c in codes if codes.count(c) > 1}
    assert not dups, f"重複コード: {[(n, c) for n, c in BABA_CODE.items() if c in dups]}"


def test_mizusawa_and_morioka_both_appear_when_racing_the_same_day():
    """盛岡・水沢が同日開催でも、両方が別レースとしてスケジュールに残ること。

    場コードが衝突していた期間は、`race_id`（baba_code + 日付 + レース番号）も
    衝突し、`parse_schedule` 末尾の `seen.setdefault` による race_id 重複除去で
    後から処理した側のレースが黙って消えていた。これが「水沢の推論が回って
    いない」の実体だった（当日ログにエラーは出ない — 単に1レースも積まれない）。
    """
    from datetime import date

    from narops.nar_source import parse_schedule

    html = """
    <div>盛岡</div><div>1R</div><div>10:00</div>
    <div>水沢</div><div>1R</div><div>10:05</div>
    """
    rows = parse_schedule(html, date(2026, 9, 7))
    tracks = {r["track_name"]: r["baba_code"] for r in rows}
    assert tracks == {"盛岡": 10, "水沢": 11}, tracks
    assert len({r["race_id"] for r in rows}) == 2, "race_id が衝突しレースが消えています"
