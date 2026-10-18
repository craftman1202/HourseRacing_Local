"""楽天競馬の発注ドライバ。

【未実装】楽天競馬の画面（ログイン・投票・確認・受付完了）を実機で確認してから書く。
画面を見ずに推測で書くと、誤った券種・金額で送信する／成功と誤認する事故につながるため、
実装されるまでは必ず error を返す（ランナーは連続エラーで自動停止する）。

実装時の要件は driver.py の docstring に従う。認証情報は環境変数 RAKUTEN_ID / RAKUTEN_PASS から
読み、ログ・通知・例外メッセージに出さない。ブラウザのプロファイルは ~/.cache/nar-orders/ に置く。
"""

from __future__ import annotations

from .driver import OrderResult
from .plan import OrderLine, OrderPlan


class RakutenDriver:
    def place(self, plan: OrderPlan, line: OrderLine, submit: bool) -> OrderResult:
        return OrderResult("error", "RakutenDriver は未実装です（楽天競馬の画面確認待ち）")

    def close(self) -> None:
        pass
