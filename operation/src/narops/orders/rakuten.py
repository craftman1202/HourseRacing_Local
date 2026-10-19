"""楽天競馬の発注ドライバ（2026-10-18、画面は実機で確認済み）。

流れ（bet.keiba.rakuten.co.jp の「通常投票」）:
  1. 当日のトップから場コード → レースIDの前半を引き、/bet/normal/RACEID/<ID><R> を開く
  2. 馬券カゴ（サーバー側に残る！）を全削除してから始める。残っていた買い目を買わないため
  3. 券種タブ → 表示された列（kaime1..3）ごとに馬番を1頭ずつ選び、金額を入れて「セット」。1点ずつ
  4. カゴの中身を読み戻して計画と照合（レース・券種・買い目・金額・件数・合計）
  5. 「投票内容を確認する」→ 確認画面の表を計画と照合 → 合計金額を入力 → 「投票する」を1回だけ押す
  どこかで1点でも違えば送信せず、カゴを空にして error を返す。送信後に完了を確認できなくても
  再送しない（二重購入を避ける。履歴を人が確認する）。

submit=False（ドライラン）は確認画面の照合まで行い、押さずにカゴを空にして終わる。
認証情報はここでは使わない（ログイン済みのプロファイルだけを使う。切れていたら error）。
枠連・枠単は列の構造が未確認なので発注しない（rejected）。
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .driver import OrderResult
from .plan import OrderLine, OrderPlan

PROFILE = Path(os.environ.get("NAR_ORDERS_PROFILE") or Path.home() / ".cache/nar-orders/chrome-profile")
RECEIPTS = Path.home() / ".cache/nar-orders/receipts"
TOP = "https://keiba.rakuten.co.jp/"
BET = "https://bet.keiba.rakuten.co.jp/bet/normal/RACEID/{id}"

TAB = {"単勝": "単勝", "複勝": "複勝", "馬連": "馬複", "ワイド": "ワイド", "馬単": "馬単",
       "3連複": "三連複", "3連単": "三連単"}
HEADS = {"単勝": ["1着"], "複勝": ["馬1"], "馬連": ["馬1", "馬2"], "ワイド": ["馬1", "馬2"],
         "馬単": ["1着", "2着"], "3連複": ["馬1", "馬2", "馬3"], "3連単": ["1着", "2着", "3着"]}
ORDERED = {"馬単", "3連単"}
UNSUPPORTED = {"枠連", "枠単"}
CLOSE_GUARD_SEC = 150          # 締切（発走2分前）の30秒手前を過ぎたら送らない


class LoginRequired(RuntimeError):
    pass


# ------------------------------------------------------------ 純粋関数（テスト対象）
def classify_completion(body: str) -> tuple[str, str, int | None]:
    """送信後の画面の文字から (status, detail, 精算可能額)。

    受付画面にも「エラー内容」という列見出しがあるので、エラー語の有無では判定しない
    （2026-10-18 に実際に買えた注文を rejected と誤判定した）。受付の印は「投票完了」と受付番号。
    """
    text = re.sub(r"\s+", " ", body or "")
    bal = re.search(r"精算可能額\s*([\d,]+)円", text)
    balance = int(bal.group(1).replace(",", "")) if bal else None
    m = re.search(r"受付番号\s*[:：]?\s*(\d+)", text)
    if "投票完了" in text or m:
        return "placed", f"受付番号 {m.group(1) if m else '?'}", balance
    if re.search(r"エラー|失敗|できません|受け付けられません|締め切り|締切", text[:1200]):
        return "rejected", "楽天がエラーを返しました: " + text[:160], balance
    return "error", "送信後の画面を判別できません。二重購入を避けるため再送しません。履歴を確認してください: " + text[:100], balance

def numbers(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text or ""))


def canon(bet_type: str, t) -> tuple[int, ...]:
    t = tuple(int(x) for x in t)
    return t if bet_type in ORDERED or len(t) == 1 else tuple(sorted(t))


def expected_rows(line: OrderLine) -> list[tuple[tuple[int, ...], int]]:
    return sorted((canon(line.bet_type, t), line.unit_yen) for t in line.tickets)


def verify(rows: list[dict], line: OrderLine, race_no: int, track: int | None = None) -> str | None:
    """読み戻した行（horse / amount_yen / race_no / track / error）が計画と同じなら None、違えば理由。"""
    if len(rows) != len(line.tickets):
        return f"件数が違います（画面 {len(rows)} / 計画 {len(line.tickets)}）"
    for r in rows:
        if r.get("race_no") not in (None, race_no):
            return f"レースが違います（画面 {r.get('race_no')}R / 計画 {race_no}R）"
        if track is not None and r.get("track") not in (None, track):
            return f"競馬場が違います（画面 {r.get('track')} / 計画 {track}）"
        if r.get("error") not in (None, "", "-"):
            return f"画面にエラー表示: {r.get('error')}"
    got = sorted((canon(line.bet_type, numbers(r["horse"])), int(r["amount_yen"])) for r in rows)
    want = expected_rows(line)
    if got != want:
        return f"買い目または金額が違います（画面 {got[:4]} / 計画 {want[:4]}）"
    total = sum(r["amount_yen"] for r in rows)
    if total != line.stake_yen:
        return f"合計が違います（画面 ¥{total:,} / 計画 ¥{line.stake_yen:,}）"
    return None


# ------------------------------------------------------------ ドライバ
class RakutenDriver:
    def __init__(self) -> None:
        self._pw = None
        self._ctx = None
        self._page = None
        self._prefix: dict[str, str] = {}

    # --- ブラウザ
    def _page_(self):
        if self._page is None:
            from playwright.sync_api import sync_playwright

            self._pw = sync_playwright().start()
            self._ctx = self._pw.chromium.launch_persistent_context(
                str(PROFILE), channel="chrome", headless=os.environ.get("NAR_ORDERS_HEADLESS") == "1",
                viewport={"width": 1280, "height": 1000}, locale="ja-JP")
            self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
            self._page.on("dialog", lambda d: d.accept())
        return self._page

    def close(self) -> None:
        try:
            if self._ctx is not None:
                self._ctx.close()
            if self._pw is not None:
                self._pw.stop()
        finally:
            self._ctx = self._pw = self._page = None

    # --- レースID
    def _bet_url(self, plan: OrderPlan) -> str:
        track, day, rno = plan.race_id[:2], plan.race_id[2:10], int(plan.race_id[10:12])
        key = day + track
        if key not in self._prefix:
            page = self._page_()
            page.goto(TOP, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            html = page.content()
            m = re.search(rf"RACEID/({day}{track}\d{{6}})00", html)
            if not m:
                raise RuntimeError(f"楽天のトップに {day} 場コード {track} のレースが見つかりません")
            self._prefix[key] = m.group(1)
        return BET.format(id=f"{self._prefix[key]}{rno:02d}")

    # --- カゴ
    def _cart(self, page) -> list[dict]:
        return page.evaluate("""() => [...document.querySelectorAll('#buyTable tr.buyData')].map(tr => ({
            kind: (tr.querySelectorAll('td.min')[1] || {innerText: ''}).innerText.replace(/\\s+/g, ''),
            horse: (tr.querySelector('td.horse') || {innerText: ''}).innerText.trim(),
            amount_yen: (parseInt((tr.querySelector('input[name=buyAmount]') || {value: '0'}).value) || 0) * 100,
            race_no: parseInt((tr.querySelector('input[name=buyRaceNo]') || {value: ''}).value) || null,
            track: parseInt((tr.querySelector('input[name=buyTrackNo]') || {value: ''}).value) || null}))""")

    def _clear_cart(self, page) -> None:
        if page.locator("#buyTable tr.buyData").count():
            page.locator("#btnBuyAmountAllDelTop").click()
            page.wait_for_timeout(1200)
        if page.locator("#buyTable tr.buyData").count():
            raise RuntimeError("馬券カゴを空にできません")

    def balance(self, plan: OrderPlan) -> int | None:
        """投票画面の見出しにある「精算可能額」。購入で減り、払戻で増える（損失の計算に使う）。"""
        page = self._page_()
        page.goto(self._bet_url(plan), wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        if "bet.keiba.rakuten.co.jp" not in page.url or page.locator("#baseAmount").count() == 0:
            raise LoginRequired("楽天にログインしていません（ログイン画面に戻されました）")
        m = re.search(r"精算可能額\s*([\d,]+)円", page.inner_text("body"))
        return int(m.group(1).replace(",", "")) if m else None

    # --- 発注
    def place(self, plan: OrderPlan, line: OrderLine, submit: bool) -> OrderResult:
        if line.bet_type in UNSUPPORTED or line.bet_type not in TAB:
            return OrderResult("rejected", f"{line.bet_type} は未対応（画面の構造を未確認）")
        left = (plan.start() - datetime.now(timezone.utc)).total_seconds()
        if left < CLOSE_GUARD_SEC:
            return OrderResult("rejected", f"締切間近のため送信しません（発走まで {left:.0f} 秒）")
        page = self._page_()
        try:
            url = self._bet_url(plan)
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(3500)
            if "bet.keiba.rakuten.co.jp" not in page.url or page.locator("#baseAmount").count() == 0:
                raise LoginRequired("楽天にログインしていません（ログイン画面に戻されました）")
            self._clear_cart(page)
            page.get_by_text(TAB[line.bet_type], exact=True).first.click()
            page.wait_for_timeout(900)
            heads = page.evaluate("() => [...document.querySelectorAll('thead.paddockList th.selecter')]"
                                  ".filter(e => e.offsetWidth || e.offsetHeight).map(e => e.innerText.trim())")
            if heads != HEADS[line.bet_type]:
                return OrderResult("rejected", f"{line.bet_type} は発売されていません（列 {heads}）")
            for t in line.tickets:
                cols = canon(line.bet_type, t)
                for i, h in enumerate(cols, start=1):
                    page.locator(f"tr#umaban_{h:02d} td.kaime{i} a").click(timeout=5000)
                page.locator("#baseAmount").fill(str(line.unit_yen // 100))
                before = len(self._cart(page))
                page.locator("a.confirmBtn.set").first.click()
                page.wait_for_timeout(500)
                if len(self._cart(page)) != before + 1:
                    raise RuntimeError(f"{'-'.join(map(str, cols))} をカゴに入れられません")
            err = verify(self._cart(page), line, int(plan.race_id[10:12]), int(plan.race_id[:2]))
            if err:
                raise RuntimeError("カゴの照合に失敗: " + err)
            page.locator("#btnBuyConfirmTop").click()
            page.wait_for_url(re.compile(r"/bet/confirm"), timeout=15000)
            page.wait_for_timeout(1500)
            rows = page.evaluate("""() => [...document.querySelectorAll('table.confirmTable tbody tr')]
                .filter(tr => tr.querySelector('td.groupShort')).map(tr => ({
                horse: tr.querySelector('td.groupShort').innerText.trim(),
                amount_yen: parseInt(tr.querySelector('td.amount').innerText.replace(/[^0-9]/g, '')) || 0,
                error: tr.querySelector('td.errorCause') ? tr.querySelector('td.errorCause').innerText.trim() : ''}))""")
            err = verify(rows, line, int(plan.race_id[10:12]))
            total = page.evaluate("() => parseInt(((document.querySelector('tr.totalAmount span') || {}).innerText || '0').replace(/[^0-9]/g, '')) || 0")
            if not err and total != line.stake_yen:
                err = f"確認画面の合計が違います（画面 ¥{total:,} / 計画 ¥{line.stake_yen:,}）"
            if err:
                raise RuntimeError("確認画面の照合に失敗: " + err)
            if not submit:
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2500)
                self._clear_cart(page)
                return OrderResult("dry_run", f"確認画面まで一致（{len(rows)}点 ¥{total:,}）。送信していません")
            left = (plan.start() - datetime.now(timezone.utc)).total_seconds()
            if left < CLOSE_GUARD_SEC:
                raise RuntimeError(f"締切間近のため送信しません（発走まで {left:.0f} 秒）")
            page.locator("input[name=verify]").fill(str(total))
            page.locator("a.voteBtn").click()                       # 1回だけ。再送しない
            try:
                page.wait_for_url(lambda u: "/bet/confirm" not in u, timeout=30000)
            except Exception:  # noqa: BLE001
                self._save_receipt(page, plan, line, "unconfirmed")
                return OrderResult("error", "送信後に完了画面を確認できません。二重購入を避けるため再送しません。"
                                            "楽天の「履歴」で購入済みか確認してください")
            page.wait_for_timeout(1500)
            body = page.inner_text("body")
            self._save_receipt(page, plan, line, "done")
            status, detail, balance = classify_completion(body)
            if status == "placed":
                detail = f"{detail}、{len(rows)}点 ¥{total:,}"
            return OrderResult(status, detail, balance)
        except LoginRequired as exc:
            return OrderResult("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            try:
                self._clear_cart(page)                          # 中途半端なカゴを残さない
            except Exception:  # noqa: BLE001
                pass
            return OrderResult("error", f"{type(exc).__name__}: {str(exc)[:200]}")

    def _save_receipt(self, page, plan: OrderPlan, line: OrderLine, tag: str) -> None:
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        stem = RECEIPTS / f"{time.strftime('%Y%m%d-%H%M%S')}_{plan.race_id}_{line.bet_type}_{tag}"
        try:
            page.screenshot(path=str(stem) + ".png", full_page=True)
            Path(str(stem) + ".html").write_text(page.content(), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
