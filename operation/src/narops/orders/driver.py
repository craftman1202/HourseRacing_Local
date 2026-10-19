"""発注ドライバ（ブラウザ操作）のインターフェース。

楽天競馬の画面操作は RakutenDriver に実装する（楽天の画面を実機で確認してから）。
ここでは「何を満たす必要があるか」だけを決める:
  - submit=False では確認画面の手前まで進み、入力内容（券種・買い目・金額）を読み戻して一致を確認する
  - submit=True でも、確認画面の内容が計画と1点でも違えば送信せず error にする
  - 例外・想定外の画面は status="error" で返す（黙って成功扱いにしない）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .plan import OrderLine, OrderPlan


@dataclass
class OrderResult:
    status: str            # placed / dry_run / rejected / error
    detail: str = ""       # 受付番号・エラー文など（認証情報は入れない）
    balance_yen: int | None = None


class Driver(Protocol):
    def balance(self, plan: OrderPlan) -> int | None: ...      # 精算可能額（読めなければ None）
    def place(self, plan: OrderPlan, line: OrderLine, submit: bool) -> OrderResult: ...
    def close(self) -> None: ...


class FakeDriver:
    """テスト用。"""

    def __init__(self, fail_on: set[str] | None = None, balance_yen: int | None = 10_000) -> None:
        self.calls: list[tuple[str, str, bool]] = []
        self.fail_on = fail_on or set()
        self.balance_yen = balance_yen

    def balance(self, plan: OrderPlan) -> int | None:
        return self.balance_yen

    def place(self, plan: OrderPlan, line: OrderLine, submit: bool) -> OrderResult:
        self.calls.append((plan.race_id, line.bet_type, submit))
        if line.bet_type in self.fail_on:
            return OrderResult("error", "fake failure")
        return OrderResult("placed" if submit else "dry_run", "fake-receipt")

    def close(self) -> None:
        pass
