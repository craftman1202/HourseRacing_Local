"""発注結果の Discord 通知。"""

from __future__ import annotations

import os

from ..discord.client import DiscordSender
from ..discord.format import COLOR_GREEN, COLOR_GREY, COLOR_YELLOW, Embed, fmt_yen
from .driver import OrderResult
from .plan import OrderLine, OrderPlan

ICON = {"placed": "✅ 発注済み", "dry_run": "🧪 ドライラン（送信なし）", "rejected": "⛔ 受付されず",
        "error": "❌ エラー"}


def _tickets(line: OrderLine, limit: int = 8) -> str:
    sep = "→" if line.bet_type in ("馬単", "3連単", "枠単") else "-"
    shown = "　".join(sep.join(str(h) for h in t) for t in line.tickets[:limit])
    more = f" ほか{len(line.tickets) - limit}点" if len(line.tickets) > limit else ""
    return shown + more


class DiscordReporter:
    def __init__(self, webhook: str | None = None, name: str = "main") -> None:
        self.name = name
        url = webhook or os.environ.get("DISCORD_WEBHOOK_ORDERS") or os.environ.get("DISCORD_WEBHOOK_PREDICTION")
        self.sender = DiscordSender(url) if url else None

    def _send(self, title: str, desc: str, color: int) -> None:
        if self.sender is not None:
            self.sender.send([Embed(title=title, description=desc, color=color, footer=f"自動発注 {self.name}")])

    def started(self, limits, live: bool) -> None:
        self._send(f"🤖 自動発注ランナー起動（{self.name}）",
                   f"モード: **{'実発注' if live else 'ドライラン（送信しません）'}**\n"
                   f"対象券種: {'、'.join(limits.bet_types) or 'なし'}\n"
                   f"上限: 1券種 {fmt_yen(limits.line_cap_yen)} / 1レース {fmt_yen(limits.race_cap_yen)} / "
                   f"1日の損失 {fmt_yen(limits.daily_loss_cap_yen)}\n一時停止: `scripts/orders.sh -i {self.name} pause`",
                   COLOR_YELLOW if live else COLOR_GREY)

    def line(self, plan: OrderPlan, line: OrderLine, res: OrderResult, live: bool) -> None:
        bal = f"\n残高 {fmt_yen(res.balance_yen)}" if res.balance_yen is not None else ""
        self._send(f"{ICON.get(res.status, res.status)}  {plan.track_name} {plan.race_no}R  {line.bet_type}",
                   f"{_tickets(line)}\n{len(line.tickets)}点 × ¥{line.unit_yen:,} = **{fmt_yen(line.stake_yen)}**"
                   f"\n{res.detail}{bal}",
                   COLOR_GREEN if res.status in ("placed", "dry_run") else COLOR_YELLOW)

    def skipped(self, plan: OrderPlan, line: OrderLine, why: str) -> None:
        self._send(f"⏭ 見送り  {plan.track_name} {plan.race_no}R  {line.bet_type}",
                   f"{_tickets(line)}（{fmt_yen(line.stake_yen)}）\n理由: {why}", COLOR_GREY)

    def skip_all(self, plan: OrderPlan, why: str) -> None:
        self._send(f"⏭ 見送り  {plan.track_name} {plan.race_no}R（全券種）", f"理由: {why}", COLOR_GREY)

    def halt(self, n: int) -> None:
        self._send("🛑 発注を停止しました", f"連続エラー {n} 回。原因を確認し、ランナーを再起動するまで発注しません。", COLOR_YELLOW)
