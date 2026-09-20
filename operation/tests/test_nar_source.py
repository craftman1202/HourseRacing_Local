

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


# ---------------------------------------------------- 0件＝構造破壊、の誤検知
def test_structure_recognized_when_all_races_of_the_day_already_finished():
    """2026-09-20 の実例。夜になり全レースが「成績」表示に変わっただけで、
    競馬場名の直後に状態ラベルが正しく並んでいる＝ HTML 構造は壊れていない。
    """
    from datetime import date

    from narops.nar_source import parse_schedule, schedule_page_structure_recognized

    html = """
    <div>帯広ば</div><div>成績</div><div>成績</div><div>特別</div><div>成績</div>
    <div>高知</div><div>成績</div><div>特別</div><div>成績</div>
    """
    assert parse_schedule(html, date(2026, 9, 20)) == [], \
        "このケースの前提（発走待ちが0件）が崩れています"
    assert schedule_page_structure_recognized(html), \
        "競馬場名の直後に既知の状態ラベルがあるのに認識できていません"


def test_structure_not_recognized_for_a_genuinely_broken_page():
    """競馬場名すら見つからない・状態ラベルが続かないページは区別できない
    （本当に HTML が壊れたか、その日は開催が無いかのどちらか）ので
    fail-closed のまま False にする。
    """
    from narops.nar_source import schedule_page_structure_recognized

    assert not schedule_page_structure_recognized("<div>準備中です</div>")
    # 競馬場名は出るが、直後に来るのが未知のラベル（構造変化の疑い）
    assert not schedule_page_structure_recognized(
        "<div>帯広ば</div><div>新しい未知の表示</div>")


def test_fetch_schedule_distinguishes_no_races_remaining_from_broken_html():
    """`fetch_schedule` は0件のとき、構造を認識できたかで例外を使い分ける。"""
    from datetime import date
    from types import SimpleNamespace

    import pytest

    from narops.nar_source import NarFetchError, NarSource, NoRacesRemaining

    finished_html = "<div>帯広ば</div><div>成績</div><div>成績</div>"
    src = NarSource(user_agent="t")
    src._client = SimpleNamespace(
        fetch=lambda url, params: SimpleNamespace(content=finished_html.encode("utf-8")))
    with pytest.raises(NoRacesRemaining):
        src.fetch_schedule(date(2026, 9, 20))

    broken_html = "<div>意味不明な新しいレイアウト</div>"
    src._client = SimpleNamespace(
        fetch=lambda url, params: SimpleNamespace(content=broken_html.encode("utf-8")))
    with pytest.raises(NarFetchError) as exc_info:
        src.fetch_schedule(date(2026, 9, 20))
    assert not isinstance(exc_info.value, NoRacesRemaining), \
        "本当に構造が読めないケースは NoRacesRemaining にしてはいけません"
