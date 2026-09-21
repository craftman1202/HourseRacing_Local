"""DC-01..10: Discord 配信。

秘密情報の非露出（DC-01）と内容整合（DC-05）と重複防止（DC-07）が Blocker。
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime, to_utc
from narops.config import SecretResolver, contains_secret, redact
from narops.discord import betslip
from narops.discord.client import (
    DiscordSender, RateLimiter, already_sent, dedupe_key, record_sent,
)
from narops.discord.format import (
    COLOR_GREEN, COLOR_GREY, COLOR_YELLOW, Embed, batch, ev_color, race_embed,
)
from narops.errors import SecretLeak

pytestmark = pytest.mark.component

FAKE_WEBHOOK = "https://discord.com/api/webhooks/123456789/AbCdEf-secret-token_XYZ"
START = jst_datetime(2026, 8, 25, 20, 35)


@pytest.fixture
def candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "horse_no": [7, 3, 11],
        "p_win": [0.241, 0.187, 0.142],
        "p_market": [0.154, 0.210, 0.083],
        "ev": [1.30, 0.89, 1.45],
        "ev_adjusted": [1.28, 0.89, 1.41],
        "stake_yen": [1200, 0, 900],
    })


# ------------------------------------------------------------ DC-01（Blocker）
def test_dc01_repr_does_not_leak_webhook():
    s = DiscordSender(FAKE_WEBHOOK, transport=httpx.MockTransport(
        lambda r: httpx.Response(204)))
    assert FAKE_WEBHOOK not in repr(s)
    assert "REDACTED" in repr(s)
    s.close()


def test_dc01_redact_removes_webhook_urls():
    text = f"送信失敗: {FAKE_WEBHOOK} timeout"
    out = redact(text, [FAKE_WEBHOOK])
    assert FAKE_WEBHOOK not in out and "REDACTED" in out


def test_dc01_redact_catches_unknown_webhook_by_shape():
    """キャッシュしていない URL も形状で潰す。既知の値の置換だけでは漏れる。"""
    other = "https://discord.com/api/webhooks/999/zzz-token"
    assert "REDACTED" in redact(f"err {other}", [])
    assert contains_secret(f"err {other}", [])


def test_dc01_exception_text_is_redacted():
    def boom(request):
        raise httpx.ConnectError(f"failed to connect to {FAKE_WEBHOOK}")

    s = DiscordSender(FAKE_WEBHOOK, transport=httpx.MockTransport(boom),
                      rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None))
    with pytest.raises(RuntimeError) as ei:
        s.send([Embed("t", "d", COLOR_GREY)])
    assert FAKE_WEBHOOK not in str(ei.value)
    s.close()


def test_dc01_payload_scan_detects_leak():
    from narops.discord.client import assert_no_secret

    assert_no_secret({"embeds": []}, [FAKE_WEBHOOK])
    with pytest.raises(SecretLeak):
        assert_no_secret({"url": FAKE_WEBHOOK}, [FAKE_WEBHOOK])


# ------------------------------------------------------------------ DC-02
def test_dc02_env_is_resolved_from_dotenv(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"DISCORD_WEBHOOK_ALERT={FAKE_WEBHOOK}\n", encoding="utf-8")
    r = SecretResolver(p)
    assert r.get("DISCORD_WEBHOOK_ALERT") == FAKE_WEBHOOK


def test_dc02_missing_secret_fails_startup(tmp_path):
    """未設定時は起動失敗。無音の送信スキップにしない。"""
    r = SecretResolver(tmp_path / "absent.env")
    with pytest.raises(RuntimeError, match="未設定"):
        r.get("DISCORD_WEBHOOK_PREDICTION")


def test_dc02_secret_manager_fallback(tmp_path):
    class FakeSM:
        def access(self, name: str) -> str:
            return FAKE_WEBHOOK

    r = SecretResolver(tmp_path / "absent.env", secret_manager=FakeSM(), project="p")
    assert r.get("DISCORD_WEBHOOK_DAILY") == FAKE_WEBHOOK


def test_dc02_optional_secret_returns_empty(tmp_path):
    r = SecretResolver(tmp_path / "absent.env")
    assert r.get("DISCORD_WEBHOOK_DEADLETTER", required=False) == ""


# ------------------------------------------------------------------ DC-03
def test_dc03_batching_respects_embed_count():
    embeds = [Embed(f"t{i}", "d", COLOR_GREY) for i in range(23)]
    chunks = batch(embeds)
    assert all(len(c) <= 10 for c in chunks)
    assert sum(len(c) for c in chunks) == 23, "分割で欠落しています"


def test_dc03_batching_respects_char_limit():
    big = [Embed("t", "x" * 2000, COLOR_GREY) for _ in range(6)]
    chunks = batch(big)
    assert all(sum(e.char_count() for e in c) <= 5500 for c in chunks)
    assert sum(len(c) for c in chunks) == 6


def test_dc03_single_oversized_embed_is_not_dropped():
    huge = [Embed("t", "x" * 9000, COLOR_GREY)]
    chunks = batch(huge)
    assert sum(len(c) for c in chunks) == 1, "上限超えの単体 embed を捨てています"


# ------------------------------------------------------------------ DC-04
def test_dc04_rate_limiter_spaces_requests():
    waits: list[float] = []
    clock = FixedClock(START)

    def sleep(s: float) -> None:
        waits.append(s)
        clock.tick(s)

    rl = RateLimiter(rps=0.4, clock=clock, sleep=sleep)
    for _ in range(3):
        rl.acquire()
    assert len(waits) == 2
    assert all(w == pytest.approx(2.5, abs=1e-6) for w in waits), "30 req/min を超えます"


def test_dc04_429_honours_retry_after():
    calls = {"n": 0}
    slept: list[float] = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(204)

    s = DiscordSender(FAKE_WEBHOOK, transport=httpx.MockTransport(handler),
                      rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None),
                      sleep=slept.append)
    res = s.send([Embed("t", "d", COLOR_GREY)])
    assert res.sent == 1 and res.retries == 1
    assert slept and slept[0] == pytest.approx(3.5), "Retry-After を尊重していません"
    s.close()


def test_dc04_no_message_is_lost_under_rate_limiting():
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        if state["n"] % 3 == 0:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(204)

    s = DiscordSender(FAKE_WEBHOOK, transport=httpx.MockTransport(handler),
                      rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None),
                      sleep=lambda _: None)
    embeds = [Embed(f"t{i}", "d", COLOR_GREY) for i in range(25)]
    res = s.send(embeds)
    assert res.sent == 25 and not res.dead_lettered
    s.close()


# ------------------------------------------------------------ DC-05（Blocker）
def test_dc05_embed_values_match_bet_candidates(candidates):
    e = race_embed(race_id="202026082511", track_name="大井", race_no=11,
                   class_name="C1三", distance=1200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=12),
                   model_release="v2026.08.24-A", track_used="B",
                   candidates=candidates, day_budget_remaining=7900, pool_yen=1_400_000)
    assert e is not None
    for _, r in candidates[candidates["ev_adjusted"] >= 1.05].iterrows():
        assert f"{r['p_win'] * 100:.1f}%" in e.description
        assert f"{r['ev_adjusted']:.2f}" in e.description
        assert f"¥{int(r['stake_yen']):,}" in e.description
    assert "¥2,100" in e.description, "買い目合計が候補と一致していません"


def test_dc05_below_threshold_horse_is_not_listed(candidates):
    e = race_embed(race_id="R", track_name="大井", race_no=11, class_name="C1",
                   distance=1200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=12),
                   model_release="v1", track_used="A", candidates=candidates)
    assert "0.89" not in e.description, "EV 0.89 の馬が掲載されています"


# ------------------------------------------------------------------ DC-06
def test_dc06_colour_thresholds():
    assert ev_color(1.25) == COLOR_GREEN
    assert ev_color(1.10) == COLOR_YELLOW
    assert ev_color(1.00) == COLOR_GREY
    assert ev_color(1.20) == COLOR_GREEN and ev_color(1.05) == COLOR_YELLOW


def test_dc06_race_with_no_qualifying_bet_is_skipped():
    weak = pd.DataFrame({"horse_no": [1], "p_win": [0.2], "p_market": [0.3],
                         "ev": [0.8], "ev_adjusted": [0.8], "stake_yen": [0]})
    assert race_embed(race_id="R", track_name="大井", race_no=1, class_name="C1",
                      distance=1200, start_ts=to_utc(START), now=to_utc(START),
                      model_release="v1", track_used="A", candidates=weak) is None


# ------------------------------------------------------------ DC-07（Blocker）
def test_dc07_duplicate_send_is_prevented(wh, clock):
    key = dedupe_key("202026082511", "v2026.08.24-A")
    assert not already_sent(wh, "202026082511", "prediction", key)
    record_sent(wh, "202026082511", "prediction", key, clock.now(), to_utc(START))
    assert already_sent(wh, "202026082511", "prediction", key)


def test_dc07_model_version_change_allows_resend(wh, clock):
    rid = "202026082511"
    record_sent(wh, rid, "prediction", dedupe_key(rid, "v1"), clock.now(), to_utc(START))
    assert not already_sent(wh, rid, "prediction", dedupe_key(rid, "v2")), \
        "モデル版が変わったら再送できるべき"


# ------------------------------------------------------------------ DC-08
@pytest.mark.parametrize("site", ["rakuten", "spat4"])
def test_dc08_betslip_roundtrip(candidates, site):
    slips = betslip.to_slips(candidates, "202026082511")
    assert len(slips) == 2
    assert betslip.roundtrip(slips, site) == slips, "往復で元の候補が復元できません"


def test_dc08_rakuten_format_shape(candidates):
    slips = betslip.to_slips(candidates, "202026082511")
    text = betslip.render(slips, "rakuten")
    assert text.splitlines()[0] == "20,20260825,11,T,7,12"


def test_dc08_spat4_format_is_fixed_width(candidates):
    slips = betslip.to_slips(candidates, "202026082511")
    for line in betslip.render(slips, "spat4").splitlines():
        assert len(line) == 20, f"固定長でない行: {line!r}"


def test_dc08_unknown_site_is_rejected(candidates):
    slips = betslip.to_slips(candidates, "202026082511")
    with pytest.raises(ValueError, match="未知の投票サイト"):
        betslip.render(slips, "unknown")


def test_dc08_malformed_race_id_is_rejected():
    with pytest.raises(ValueError, match="12桁"):
        betslip.render([betslip.Slip("bad", "単勝", 1, 100)], "rakuten")


# ------------------------------------------------------------------ DC-09
def test_dc09_persistent_failure_goes_to_dead_letter():
    s = DiscordSender(FAKE_WEBHOOK,
                      transport=httpx.MockTransport(lambda r: httpx.Response(503)),
                      rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None),
                      sleep=lambda _: None, max_attempts=3)
    res = s.send([Embed("t", "d", COLOR_GREY)])
    assert res.sent == 0
    assert len(res.dead_lettered) == 1, "失敗データが保持されていません"
    s.close()


def test_dc09_client_error_raises_immediately():
    s = DiscordSender(FAKE_WEBHOOK,
                      transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")),
                      rate_limiter=RateLimiter(rps=1000, sleep=lambda _: None))
    with pytest.raises(RuntimeError, match="400"):
        s.send([Embed("t", "d", COLOR_GREY)])
    s.close()


# ------------------------------------------------------------------ DC-10
def test_dc10_daily_summary_matches_pnl(wh, clock):
    from narops.discord.format import daily_summary_embed

    row = pd.Series({"business_date": "2026-08-24", "n_races": 42, "n_bets": 11,
                     "stake_yen": 8800, "return_yen": 9460, "roi": 1.075,
                     "hit_rate": 0.2727, "brier": 0.8123, "is_paper": True})
    e = daily_summary_embed(row, skew_verdict="PASS", coverage=0.985)
    assert "107.5%" in e.description and "27.3%" in e.description
    assert "¥8,800" in e.description and "¥9,460" in e.description
    assert "PASS" in e.description
    assert "ペーパートレード" in e.description, "運用区分が明示されていません"


def test_dc10_live_mode_is_labelled():
    from narops.discord.format import daily_summary_embed

    row = pd.Series({"business_date": "2026-08-24", "n_races": 1, "n_bets": 1,
                     "stake_yen": 100, "return_yen": 0, "roi": 0.0,
                     "hit_rate": 0.0, "brier": 0.9, "is_paper": False})
    assert "実運用" in daily_summary_embed(row, "PASS", 1.0).description


# ------------------------------------------------------------ DC-01（伏せ字）
def test_redaction_is_not_limited_to_numeric_webhook_ids():
    """ID が数字とは限らない。形で潰さないと未知の値が漏れる。

    `\\d+` 限定だと、数字以外の ID を持つ URL がそのまま例外文に載る
    （実際にエラーハンドラのテストで漏れを検出した）。
    """
    from narops.config import contains_secret, redact

    leaky = [
        "https://discord.com/api/webhooks/xxx/yyy",
        "https://discord.com/api/webhooks/123456789/abc-DEF_1",
        "https://discordapp.com/api/webhooks/1/2",
        "https://example.com/webhook/abc",
    ]
    for url in leaky:
        assert contains_secret(url), url
        assert url not in redact(f"エラー: {url} で失敗"), url
        assert "***REDACTED***" in redact(url)


def test_redaction_leaves_ordinary_urls_alone():
    from narops.config import contains_secret, redact

    plain = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable"
    assert not contains_secret(plain)
    assert redact(plain) == plain


# --------------------------------------------------- stake_hint_yen（参考額）
def test_dc05_shows_a_hint_when_ev_passes_but_stake_rounds_to_zero():
    """EV は基準を満たすのに実額が0のとき、通知に理由が読めること。

    以前は「推奨」欄が空欄（—）になるだけで、EV が良いのになぜ賭けないのかが
    通知だけからは分からなかった。単位未満（stake_hint_yen>0）なのか、
    そもそも検討対象外なのかを区別できるようにする。
    """
    cand = pd.DataFrame({
        "horse_no": [5], "p_win": [0.145], "p_market": [0.041],
        "ev": [2.82], "ev_adjusted": [2.82],
        "stake_yen": [0], "stake_hint_yen": [74],
    })
    e = race_embed(race_id="R", track_name="帯広ば", race_no=1, class_name="C1",
                   distance=200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=10),
                   model_release="v-banei", track_used="A+B", candidates=cand)
    assert e is not None
    assert "(¥74)" in e.description
    assert "最低賭け金" in e.description


def test_dc05_no_hint_note_when_every_stake_is_actionable(candidates):
    """実額がちゃんと出ている通常時は、参考額の注記を出さない（冗長な文言を足さない）。"""
    e = race_embed(race_id="202026082511", track_name="大井", race_no=11,
                   class_name="C1三", distance=1200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=12),
                   model_release="v2026.08.24-A", track_used="B",
                   candidates=candidates)
    assert e is not None
    assert "最低賭け金" not in e.description


def test_dc05_hint_is_absent_when_ev_itself_fails():
    """EV 基準を割った銘柄は、参考額があっても出さない（stake_hint_yen が0のはず）。"""
    cand = pd.DataFrame({
        "horse_no": [9], "p_win": [0.02], "p_market": [0.05],
        "ev": [0.5], "ev_adjusted": [0.5],
        "stake_yen": [0], "stake_hint_yen": [0],
    })
    e = race_embed(race_id="R", track_name="帯広ば", race_no=1, class_name="C1",
                   distance=200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=10),
                   model_release="v-banei", track_used="A", candidates=cand)
    assert e is None, "EV 基準未満なので通知自体が出ないはず"


def test_fmt_stake_prefers_the_actionable_amount_over_the_hint():
    from narops.discord.format import fmt_stake

    assert fmt_stake(1200, 1200) == "¥1,200"
    assert fmt_stake(0, 74) == "(¥74)"
    assert fmt_stake(0, 0) == "—"
    assert fmt_stake(0, None) == "—"
    assert fmt_stake(float("nan"), 74) == "(¥74)"


# --------------------------------------------------- 単勝・複勝の併記
def test_dc05_shows_both_win_and_place_probability_when_available():
    """p_top3 が渡されたら単勝・複勝の両方を通知に載せること。"""
    cand = pd.DataFrame({
        "horse_no": [5], "p_win": [0.20], "p_top3": [0.55], "p_market": [0.10],
        "ev": [2.0], "ev_adjusted": [2.0], "stake_yen": [500],
    })
    e = race_embed(race_id="R", track_name="帯広ば", race_no=1, class_name="C1",
                   distance=200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=10), candidates=cand,
                   model_release="v-banei", track_used="A+B")
    assert e is not None
    assert "単勝" in e.description and "複勝" in e.description
    assert "20.0%" in e.description   # 単勝
    assert "55.0%" in e.description   # 複勝


def test_dc05_falls_back_gracefully_without_place_probability(candidates):
    """p_top3 を渡さない既存の呼び出し元でも、複勝欄なしで正しく組み立つこと。"""
    e = race_embed(race_id="202026082511", track_name="大井", race_no=11,
                   class_name="C1三", distance=1200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=12),
                   model_release="v2026.08.24-A", track_used="B",
                   candidates=candidates)
    assert e is not None
    assert "複勝" not in e.description
    assert "単勝" in e.description


# ------------------------------------------------------------ 並び順
def test_dc05_rows_are_sorted_by_win_probability_not_ev():
    """EV 順だと「万馬券候補が上、本命が下」という直感に反する並びになる。

    EV は的中率×払戻なので、同じ的中率でもオッズが高い（人気薄の）馬ほど
    伸びやすく、p_win の順序とは一致しない。実際に勝ちそうな順で読めるように
    p_win 降順にする。
    """
    cand = pd.DataFrame({
        "horse_no": [1, 2, 3],
        "p_win": [0.10, 0.35, 0.20],       # 本命は2番
        "p_market": [0.08, 0.30, 0.15],
        "ev": [3.5, 1.1, 1.3],             # EV では1番が最上位（人気薄の穴）
        "ev_adjusted": [3.5, 1.1, 1.3],
        "stake_yen": [0, 0, 0],
    })
    e = race_embed(race_id="R", track_name="大井", race_no=1, class_name="C1",
                   distance=1200, start_ts=to_utc(START),
                   now=to_utc(START) - timedelta(minutes=10),
                   model_release="v-test", track_used="A", candidates=cand,
                   min_ev=1.0)
    assert e is not None
    lines = e.description.splitlines()
    rows = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
    order = [int(r.split()[0]) for r in rows]
    assert order == [2, 3, 1], f"p_win 降順（2,3,1）になっていません: {order}"
