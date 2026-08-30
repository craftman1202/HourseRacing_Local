"""Discord embed の組み立て。

意思決定に必要な情報だけを載せる。過剰な情報は判断を鈍らせる（設計書 §4.3）。
表示値は必ず bet_candidate の値と丸め規則込みで一致させる（DC-05）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from ..clock import to_jst

# EV 水準で色を変える（DC-06）
COLOR_GREEN = 0x2ECC71
COLOR_YELLOW = 0xF1C40F
COLOR_GREY = 0x95A5A6

MAX_EMBEDS = 10
MAX_CHARS = 5500        # 6,000 に余裕を持たせる。マルチバイトの境界事故を避ける


@dataclass
class Embed:
    title: str
    description: str
    color: int
    footer: str = ""
    fields: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"title": self.title, "description": self.description, "color": self.color}
        if self.footer:
            d["footer"] = {"text": self.footer}
        if self.fields:
            d["fields"] = self.fields
        return d

    def char_count(self) -> int:
        """Discord は embed の全テキストを合算して 6,000 で切る。"""
        n = len(self.title) + len(self.description) + len(self.footer)
        return n + sum(len(f.get("name", "")) + len(f.get("value", "")) for f in self.fields)


def ev_color(ev: float, green: float = 1.20, yellow: float = 1.05) -> int:
    if ev >= green:
        return COLOR_GREEN
    if ev >= yellow:
        return COLOR_YELLOW
    return COLOR_GREY


def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"


def fmt_ev(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.2f}"


def fmt_yen(x) -> str:
    return "—" if pd.isna(x) or x <= 0 else f"¥{int(x):,}"


def fmt_stake(stake_yen, stake_hint_yen=None) -> str:
    """推奨額の表示。

    `stake_yen` は実際に賭ける額（100円単位、0 なら「賭けない」）。
    EV は基準を満たすのに `stake_yen` が0のとき、理由は主に2つあり得る:
    Kelly 額が最低賭け金（100円）に届かない、か、自己インパクト補正後に
    EV が基準を割った。前者は `stake_hint_yen` に丸め前の額が残るので、
    それを括弧書きの参考額として出す。EV が良いのに推奨が空欄なだけだと、
    運用側が「なぜ」を読めない。
    """
    if not pd.isna(stake_yen) and stake_yen > 0:
        return fmt_yen(stake_yen)
    if stake_hint_yen is not None and not pd.isna(stake_hint_yen) and stake_hint_yen > 0:
        return f"(¥{int(stake_hint_yen):,})"
    return "—"


def race_embed(
    *, race_id: str, track_name: str, race_no: int, class_name: str, distance: int,
    start_ts: datetime, now: datetime, model_release: str, track_used: str,
    candidates: pd.DataFrame, horse_names: dict[int, str] | None = None,
    day_budget_remaining: int | None = None, pool_yen: float | None = None,
    min_ev: float = 1.05,
) -> Embed | None:
    """レース予測通知。

    EV が閾値未満のレースは通知自体を省略する。1日60レースすべてを通知すると
    通知疲れで見なくなり、それが最大の運用リスク（設計書 §4.3）。
    """
    shown = candidates[candidates["ev_adjusted"] >= min_ev]
    if shown.empty:
        return None

    names = horse_names or {}
    jst = to_jst(start_ts)
    remaining = (start_ts - now).total_seconds() / 60.0

    lines = ["```", " 馬番 馬名              予測   市場   EV    推奨"]
    for _, r in shown.sort_values("ev_adjusted", ascending=False).iterrows():
        no = int(r["horse_no"])
        name = (names.get(no, f"{no}番") + "　" * 9)[:9]
        lines.append(
            f" {no:>3}  {name}  {fmt_pct(r['p_win']):>6} {fmt_pct(r.get('p_market', float('nan'))):>6}"
            f" {fmt_ev(r['ev_adjusted']):>5}  {fmt_stake(r['stake_yen'], r.get('stake_hint_yen'))}")
    lines.append("```")

    total = int(shown["stake_yen"].sum())
    tail = [f"買い目合計 {fmt_yen(total)}"]
    if day_budget_remaining is not None:
        tail.append(f"本日残枠 {fmt_yen(day_budget_remaining)}")
    lines.append("　".join(tail))
    has_hint = ("stake_hint_yen" in shown.columns
               and ((shown["stake_yen"] <= 0) & (shown["stake_hint_yen"] > 0)).any())
    if has_hint:
        lines.append("（　）は最低賭け金（¥100）に届かない参考額。実際には賭けません")
    if pool_yen:
        lines.append(f"⚠ 想定プール ¥{pool_yen / 1e6:.1f}M（推定値・実測より小さめに見積り）")

    best = float(shown["ev_adjusted"].max())
    return Embed(
        title=f"🏇 {track_name} {race_no}R  {class_name} {distance}m  "
              f"発走 {jst:%H:%M} (残り{remaining:.0f}分)",
        description="\n".join(lines),
        color=ev_color(best),
        footer=f"{model_release} / トラック{track_used}",
    )


def batch(embeds: list[Embed], max_count: int = MAX_EMBEDS,
          max_chars: int = MAX_CHARS) -> list[list[Embed]]:
    """1メッセージあたり 10 embed・合計 5,500 文字で分割する（DC-03）。

    分割で欠落が出ないこと（入出力の embed 総数が一致）をテストで固定する。
    """
    chunks: list[list[Embed]] = []
    cur: list[Embed] = []
    cur_chars = 0
    for e in embeds:
        n = e.char_count()
        if cur and (len(cur) >= max_count or cur_chars + n > max_chars):
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append(e)
        cur_chars += n
    if cur:
        chunks.append(cur)
    return chunks


def daily_summary_embed(row: pd.Series, skew_verdict: str, coverage: float) -> Embed:
    """日次サマリ（DC-10）。値は pnl_daily と一致させる。"""
    desc = "\n".join([
        f"レース {int(row['n_races'])} / ベット {int(row['n_bets'])}",
        f"投資 {fmt_yen(row['stake_yen'])}  払戻 {fmt_yen(row['return_yen'])}",
        f"回収率 **{row['roi'] * 100:.1f}%**  的中率 {row['hit_rate'] * 100:.1f}%",
        f"Brier {row['brier']:.4f}  カバレッジ {coverage * 100:.1f}%",
        f"skew 検証: **{skew_verdict}**",
        ("_ペーパートレード（実投票なし）_" if row.get("is_paper", True) else "_実運用_"),
    ])
    return Embed(title=f"📊 日次サマリ {row['business_date']}", description=desc,
                 color=COLOR_GREEN if row["roi"] >= 1.0 else COLOR_GREY)


def alert_embed(kind: str, message: str, severity: str = "Critical") -> Embed:
    return Embed(title=f"🚨 [{severity}] {kind}", description=message, color=0xE74C3C)
