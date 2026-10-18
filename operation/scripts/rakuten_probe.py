"""楽天競馬の投票画面の構成を記録する調査スクリプト（購入はしない）。

使い方（あなたのPCのターミナルで。Claude Code からなら `! ` を付けて）:
    cd operation && ../learning/.venv/bin/python scripts/rakuten_probe.py \
        https://keiba.rakuten.co.jp/race_card/list/RACEID/202610183129090206

やること: ログイン済みのプロファイル（~/.cache/nar-orders/chrome-profile）で出馬表を開き、
「投票する」を押して開く投票画面を記録する（スクリーンショット・HTML・入力欄とボタンの一覧）。
それ以外は何も押さない。購入・確定はしない。
出力: ~/.cache/nar-orders/probe/<時刻>/ （ログイン情報は含まない）。この場所を Claude に伝える。

段階を進める場合（購入はしない）:
    --bet-type 馬連     投票画面で券種タブを押す
    --horses 3,7        馬番を選ぶ（上の券種に合う数）
    --amount 100        金額を入れる
    --confirm-screen    「確認」ボタンまで押して確認画面を記録する（確定ボタンは押さない）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROFILE = Path.home() / ".cache/nar-orders/chrome-profile"


def dump(page, out: Path, name: str) -> None:
    page.wait_for_timeout(1500)
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    (out / f"{name}.html").write_text(page.content(), encoding="utf-8")
    items = page.evaluate("""() => [...document.querySelectorAll('input,button,select,a,label')].map(e => ({
        tag: e.tagName, type: e.type || '', name: e.name || '', id: e.id || '', cls: (e.className || '').toString().slice(0, 60),
        text: (e.innerText || e.value || '').trim().slice(0, 40), href: (e.getAttribute('href') || '').slice(0, 80),
        visible: !!(e.offsetWidth || e.offsetHeight), checked: !!e.checked, disabled: !!e.disabled}))""")
    (out / f"{name}.elements.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] {page.url[:120]} | {page.title()} | 要素 {len(items)} 個")


SURVEY_JS = """() => {
  const vis = e => !!(e.offsetWidth || e.offsetHeight);
  const heads = [...document.querySelectorAll('thead.paddockList th.selecter')].filter(vis).map(e => e.className.replace('selecter','').trim() + ':' + e.innerText.trim());
  const rows = [...document.querySelectorAll('tbody.paddockList tr')];
  const cols = {};
  for (const td of rows[0] ? rows[0].querySelectorAll('td.selecter') : []) if (vis(td)) cols[td.className.replace('link','').replace('selecter','').trim()] = 1;
  const methods = [...document.querySelectorAll('li input[type=radio]')].filter(e => e.labels[0] && vis(e.labels[0])).map(e => e.id + ':' + e.labels[0].innerText.trim());
  const active = [...document.querySelectorAll('li.active, li.current, .on')].map(e => (e.innerText||'').trim().slice(0,12)).filter(Boolean).slice(0,6);
  return {heads, cols: Object.keys(cols), methods, active};
}"""


def survey(v, out: Path) -> None:
    res = {}
    for name in ["単勝", "複勝", "枠複", "枠単", "馬複", "馬単", "ワイド", "三連複", "三連単"]:
        try:
            v.get_by_text(name, exact=True).first.click()
            v.wait_for_timeout(900)
            res[name] = v.evaluate(SURVEY_JS)
            print(f"[{name}]", json.dumps(res[name], ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            res[name] = {"error": str(exc)[:120]}
            print(f"[{name}] ERROR", str(exc)[:120])
    (out / "survey.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("race_url")
    ap.add_argument("--bet-type")
    ap.add_argument("--horses")
    ap.add_argument("--amount")
    ap.add_argument("--confirm-screen", action="store_true")
    ap.add_argument("--clear-cart", action="store_true", help="馬券カゴを全削除する（購入ではない）")
    ap.add_argument("--survey", action="store_true", help="全券種のタブを順に押し、表示される選択列を記録する（カゴには入れない）")
    a = ap.parse_args()
    out = Path.home() / ".cache/nar-orders/probe" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(PROFILE), channel="chrome", headless=False,
                                                   viewport={"width": 1280, "height": 1000}, locale="ja-JP")
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(a.race_url, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        dump(page, out, "00_race")
        with ctx.expect_page(timeout=20000) as ev:
            page.get_by_text("投票する", exact=True).first.click()
        v = ev.value
        v.wait_for_load_state("domcontentloaded")
        v.wait_for_timeout(5000)
        dump(v, out, "01_vote_window")
        for i, f in enumerate(v.frames[1:], start=1):
            print("  frame", i, f.url[:100])
        v.on("dialog", lambda d: (print("ダイアログ:", d.message[:80]), d.accept()))
        if a.clear_cart:
            n0 = v.locator("#buyTotalCountTop").inner_text()
            v.locator("#btnBuyAmountAllDelTop").click()
            v.wait_for_timeout(1500)
            dump(v, out, "01b_cart_cleared")
            print("カゴ件数:", n0, "→", v.locator("#buyTotalCountTop").inner_text())
        if a.survey:
            survey(v, out)
        if a.bet_type:
            v.get_by_text(a.bet_type, exact=True).first.click()
            v.wait_for_timeout(900)
            dump(v, out, "02_bet_type")
        if a.horses:
            # 通常方式: 列（kaime1, kaime2, …）ごとに 1 頭ずつ。馬番は tr#umaban_NN の td.kaimeN
            for i, h in enumerate(a.horses.split(","), start=1):
                v.locator(f"tr#umaban_{int(h):02d} td.kaime{i} a").click()
            dump(v, out, "03_horses")
        if a.amount:
            v.locator("#baseAmount").fill(str(int(a.amount) // 100))      # 100 円単位
            v.locator("a.confirmBtn.set").first.click()
            v.wait_for_timeout(1200)
            dump(v, out, "04_set")
            print("カゴ:", re.sub(r"\s+", " ", v.locator("#buyTable").inner_text())[:200],
                  "| 件数", v.locator("#buyTotalCountTop").inner_text(), "| 合計", v.locator("#buyTotalAmountTop").inner_text())
        if a.confirm_screen:
            v.locator("#btnBuyConfirmTop").click()
            v.wait_for_timeout(2500)
            dump(v, out, "05_confirm")
            print("確認画面で止めました。確定ボタンは押していません。")
        if sys.stdin.isatty():
            input("記録しました。内容を確認して Enter でブラウザを閉じます… ")
        ctx.close()
    print("出力先:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
