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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("race_url")
    ap.add_argument("--bet-type")
    ap.add_argument("--horses")
    ap.add_argument("--amount")
    ap.add_argument("--confirm-screen", action="store_true")
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
        if a.bet_type:
            v.get_by_text(a.bet_type, exact=True).first.click()
            dump(v, out, "02_bet_type")
        if a.horses:
            for h in a.horses.split(","):
                v.get_by_text(re.sub(r"\D", "", h), exact=True).first.click()
            dump(v, out, "03_horses")
        if a.amount:
            box = v.locator("input[type=text], input[type=number], input[type=tel]").last
            box.fill(a.amount)
            dump(v, out, "04_amount")
        if a.confirm_screen:
            v.get_by_role("button", name=re.compile("確認|セット|追加")).first.click()
            dump(v, out, "05_confirm")
            print("確認画面で止めました。確定ボタンは押していません。")
        input("記録しました。内容を確認して Enter でブラウザを閉じます… ")
        ctx.close()
    print("出力先:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
