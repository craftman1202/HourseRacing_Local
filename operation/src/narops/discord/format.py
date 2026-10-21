"""Discord embed の組み立て。

意思決定に必要な情報だけを載せる。過剰な情報は判断を鈍らせる（設計書 §4.3）。
表示値は必ず bet_candidate の値と丸め規則込みで一致させる（DC-05）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
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
    min_ev: float = 1.05, exotic: list | None = None, plans_only: bool = False,
) -> Embed | None:
    """レース予測通知。

    plans_only=True は全券種の買い目の版（BetBundle を持つリリース）。規則に合う買い目（exotic）が
    あるレースだけ通知し、無ければ通知しない。旧来の「単勝 EV ≥ 基準」の通知へは落とさない。

    EV が閾値未満のレースは通知自体を省略する。1日60レースすべてを通知すると
    通知疲れで見なくなり、それが最大の運用リスク（設計書 §4.3）。
    """
    # 全券種の買い目（2026-10-17、全モデルアンサンブルのリリース）。BetPlan に structure があり、
    # 単勝・複勝 EV の表（ev_place）を持たない経路
    plans = [pl for pl in (exotic or []) if hasattr(pl, "structure")]
    if plans_only and not plans:
        return None
    if plans and "ev_place" not in candidates.columns:
        return _race_embed_plans(
            track_name=track_name, race_no=race_no, class_name=class_name, distance=distance,
            start_ts=start_ts, now=now, model_release=model_release, track_used=track_used,
            candidates=candidates, horse_names=horse_names,
            day_budget_remaining=day_budget_remaining, plans=plans)
    if "ev_place" in candidates.columns:
        return _race_embed_max_ev(
            race_id=race_id, track_name=track_name, race_no=race_no,
            class_name=class_name, distance=distance, start_ts=start_ts, now=now,
            model_release=model_release, track_used=track_used, candidates=candidates,
            horse_names=horse_names, day_budget_remaining=day_budget_remaining,
            min_ev=min_ev, exotic=exotic)
    shown = candidates[candidates["ev_adjusted"] >= min_ev]
    if shown.empty:
        return None

    names = horse_names or {}
    jst = to_jst(start_ts)
    remaining = (start_ts - now).total_seconds() / 60.0

    # 複勝（上位3着以内）は narops.inference が単勝確率から Harville 式で
    # 近似した p_top3 を使う。無い呼び出し元（テストなど）では列自体を出さない。
    has_top3 = "p_top3" in shown.columns
    header = " 馬番 馬名              単勝   複勝   市場   EV    推奨" if has_top3 \
        else " 馬番 馬名              単勝   市場   EV    推奨"
    lines = ["```", header]
    # 1着になる予想確率（単勝）が高い順。EV 順だと「万馬券候補が上、本命が下」
    # という直感に反する並びになる — EV は的中率と払戻の掛け算なので、
    # 単勝オッズが高い（人気が薄い）馬ほど同じ的中率でも EV が伸びやすく、
    # 実際に勝ちそうな順（p_win 順）と一致しない。
    for _, r in shown.sort_values("p_win", ascending=False).iterrows():
        no = int(r["horse_no"])
        name = (names.get(no, f"{no}番") + "　" * 9)[:9]
        top3 = f" {fmt_pct(r['p_top3']):>6}" if has_top3 else ""
        lines.append(
            f" {no:>3}  {name}  {fmt_pct(r['p_win']):>6}{top3}"
            f" {fmt_pct(r.get('p_market', float('nan'))):>6}"
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


BET_MARK = {"単勝": "単", "複勝": "複"}


def _race_embed_max_ev(
    *, race_id: str, track_name: str, race_no: int, class_name: str, distance: int,
    start_ts: datetime, now: datetime, model_release: str, track_used: str,
    candidates: pd.DataFrame, horse_names: dict[int, str] | None,
    day_budget_remaining: int | None, min_ev: float, exotic: list | None = None,
) -> Embed | None:
    """「単勝・複勝の EV の高い方を1点」の通知（2026-09-27）。

    単勝 EV・複勝 EV のどちらかが min_ev 以上の馬と、賭ける馬を載せる。
    推奨欄は券種（単/複）と金額。金額は bet_candidate の stake_yen と一致させる（DC-05）。
    `exotic`（3連単の流し、2026-10-06）があれば買い方と全点の合計額を別欄に載せる。
    単勝・複勝の候補が無くても、3連単を買うレースは通知する。
    2026-10-13: `exotic` は全券種の推奨（shared.BetPlan）も受ける。stake=False は「参考」欄に
    載せ、買い目合計・残枠には数えない。
    """
    exotic = list(exotic or [])
    ev_w = candidates["ev_adjusted"]
    ev_p = candidates["ev_place"]
    shown = candidates[(ev_w >= min_ev) | (ev_p >= min_ev) | (candidates["stake_yen"] > 0)]
    # 単独では通知しない参考（standalone=False、枠の券種）だけのレースは通知しない
    triggers = [pl for pl in exotic
                if getattr(pl, "stake", True) or getattr(pl, "standalone", True)]
    if shown.empty and not triggers:
        return None

    names = horse_names or {}
    jst = to_jst(start_ts)
    remaining = (start_ts - now).total_seconds() / 60.0
    lines = ["```", " 馬番 馬名              単勝   複勝  単EV  複EV  推奨"]
    for _, r in shown.sort_values("p_win", ascending=False).iterrows():
        no = int(r["horse_no"])
        name = (names.get(no, f"{no}番") + "　" * 9)[:9]
        rec = "—"
        if r["stake_yen"] > 0:
            rec = f"{BET_MARK.get(r.get('bet_type'), '?')}{fmt_yen(r['stake_yen'])}"
        lines.append(
            f" {no:>3}  {name}  {fmt_pct(r['p_win']):>6} {fmt_pct(r['p_top3']):>6}"
            f" {fmt_ev(r['ev_adjusted']):>5} {fmt_ev(r['ev_place']):>5}  {rec}")
    lines.append("```")
    if shown.empty:          # 3連単だけのレース。空の表は出さない
        lines = []

    bets = shown[shown["stake_yen"] > 0]
    staked = [pl for pl in exotic if getattr(pl, "stake", True)]
    ref = [pl for pl in exotic if not getattr(pl, "stake", True)]
    ex_total = sum(int(pl.stake_yen) for pl in staked)
    for pl in staked:
        lines.append(_plan_lines(pl, names))
    if ref:
        lines.append("📋 **参考（各券種の推奨。賭け金には数えない）**")
        for pl in ref:
            lines.append(_plan_lines(pl, names))
    tail = [f"買い目合計 {fmt_yen(int(bets['stake_yen'].sum()) + ex_total)}"]
    for t in ("単勝", "複勝"):
        part = bets[bets.get("bet_type") == t] if "bet_type" in bets.columns else bets.iloc[0:0]
        if len(part):
            tail.append(f"{t} {fmt_yen(int(part['stake_yen'].sum()))}")
    for pl in staked:
        tail.append(f"{getattr(pl, 'bet_type', '3連単')} {fmt_yen(int(pl.stake_yen))}")
    if day_budget_remaining is not None:
        tail.append(f"本日残枠 {fmt_yen(day_budget_remaining)}")
    lines.append("　".join(tail))
    if not shown.empty:
        lines.append("推奨: 単勝・複勝の EV の高い方を1点（EV≥1・的中確率≥60%・ケリー）")
    if not shown.empty and shown["ev_place"].isna().all():
        lines.append("⚠ 複勝オッズ未取得のため単勝だけで判定")

    best = float(np.nanmax(np.r_[ev_w.loc[shown.index].to_numpy(dtype=float),
                                 ev_p.loc[shown.index].to_numpy(dtype=float), -np.inf]))
    return Embed(
        title=f"🏇 {track_name} {race_no}R  {class_name} {distance}m  "
              f"発走 {jst:%H:%M} (残り{remaining:.0f}分)",
        description="\n".join(lines),
        color=COLOR_GREEN if len(bets) or staked else ev_color(best),
        footer=f"{model_release} / トラック{track_used}",
    )


BET_ORDER = ("単勝", "複勝", "ワイド", "馬連", "馬単", "3連複", "3連単", "枠連", "枠単")
MAX_TICKET_LINES = 8


def _horse(no: int, names: dict) -> str:
    return f"{no}番 {names.get(no, '')}".rstrip()


def _race_embed_plans(*, track_name, race_no, class_name, distance, start_ts, now, model_release,
                      track_used, candidates, horse_names, day_budget_remaining, plans) -> Embed | None:
    """全券種の買い目（軸・相手・買い目・金額）を券種ごとに1ブロックで出す。

    上に勝率上位の馬（印つき）、その下に券種ごとの「軸 / 相手 / 買い目 / 金額」。
    EV で決めた券種は買い目ごとに EV と予想配当（払戻予測モデル）も出す。
    """
    names = horse_names or {}
    staked = [pl for pl in plans if getattr(pl, "stake", True)]
    triggers = [pl for pl in plans if getattr(pl, "stake", True) or getattr(pl, "standalone", True)]
    if not triggers:
        return None
    jst = to_jst(start_ts)
    remaining = (start_ts - now).total_seconds() / 60.0
    top = candidates.sort_values("p_win", ascending=False).head(5)
    marks = "◎○▲△☆"
    lines = ["**予想（勝率）** " + "　".join(
        f"{marks[i]}{int(r['horse_no'])} {names.get(int(r['horse_no']), '')} {fmt_pct(r['p_win'])}"
        for i, (_, r) in enumerate(top.iterrows()))]
    order = {b: i for i, b in enumerate(BET_ORDER)}
    for pl in sorted(plans, key=lambda x: order.get(x.bet_type, 99)):
        lines.append(_plan_block(pl, names))
    total = sum(int(pl.stake_yen) for pl in staked)
    tail = [f"買い目合計 **{fmt_yen(total)}**"]
    if day_budget_remaining is not None:
        tail.append(f"本日残枠 {fmt_yen(day_budget_remaining)}")
    lines.append("　".join(tail))
    lines.append("_ペーパー運用。検証値は 2020〜2023 の walk-forward（オッズは予測払戻）_")
    return Embed(
        title=f"🏇 {track_name} {race_no}R  {class_name} {distance}m  "
              f"発走 {jst:%H:%M} (残り{remaining:.0f}分)",
        description="\n".join(lines),
        color=COLOR_GREEN if staked else COLOR_GREY,
        footer=f"{model_release} / トラック{track_used}",
    )


def _plan_block(pl, names: dict) -> str:
    """1 券種ぶん。軸と相手を名前つきで、買い目は券種の書き方で（→ は着順あり、- は着順なし）。"""
    ordered = pl.bet_type in ("馬単", "3連単", "枠単")
    sep = "→" if ordered else "-"
    head = f"🎯 **{pl.bet_type}**（{pl.structure}）"
    if pl.bet_type in ("枠連", "枠単"):
        who = f"　軸馬 {_horse(pl.axis, names)} の枠 ／ 相手 {_horse(pl.partners[0], names) + ' の枠' if pl.partners else '—'}"
    elif pl.bet_type in ("単勝", "複勝"):
        who = "　対象 " + "、".join(_horse(t[0], names) for t in pl.tickets)
    elif len(pl.tickets) == 1 and len(pl.tickets[0]) == 1:
        who = f"　軸 {_horse(pl.axis, names)}（勝率 {pl.p_axis * 100:.0f}%）"
    else:
        partners = sorted({h for t in pl.tickets for h in t if h != pl.axis},
                          key=lambda h: pl.partners.index(h) if h in pl.partners else 99)
        who = (f"　軸 {_horse(pl.axis, names)} ／ 相手 "
               + "、".join(_horse(h, names) for h in partners))
    detail = {tuple(d["ticket"]): d for d in (getattr(pl, "detail", None) or [])}
    rows = []
    for t in pl.tickets[:MAX_TICKET_LINES]:
        d = detail.get(tuple(t))
        extra = f"（EV {d['ev']:.2f}・予想 {d['odds_hat']:.1f}倍）" if d else ""
        rows.append(sep.join(str(h) for h in t) + extra)
    more = f" ほか{len(pl.tickets) - MAX_TICKET_LINES}点" if len(pl.tickets) > MAX_TICKET_LINES else ""
    money = f"{len(pl.tickets)}点 × ¥{pl.unit_yen:,} = **¥{pl.stake_yen:,}**"
    note = f"　{pl.note}" if pl.note else ""
    label = "買い目（枠番）" if pl.bet_type in ("枠連", "枠単") else "買い目"
    return f"{head}\n{who}\n　{label} " + "　".join(rows) + more + f"\n　{money}{note}"


def _plan_lines(plan, names: dict) -> str:
    """推奨買い目1つ。3連単の 1番人気1着固定流し（TrifectaPlan）は従来の書式のまま。"""
    if not hasattr(plan, "structure"):
        return _trifecta_lines(plan, names)
    axis = f"{plan.axis}番 {names.get(plan.axis, '')}".rstrip()
    sep = "→" if plan.bet_type in ("馬単", "3連単", "枠単") else "-"
    if len(plan.tickets) <= 6:
        body = "  ".join(sep.join(str(h) for h in t) for t in plan.tickets)
    else:
        body = f"軸 {axis} → 相手 {','.join(str(h) for h in plan.partners[:8])}"
    head = f"🎯 **{plan.structure}**"
    if plan.bet_type in ("枠連", "枠単"):
        head += f"（軸馬 {axis} の枠）"
    elif len(plan.tickets) <= 6:
        head += f"（軸 {axis}）"
    note = f"　{plan.note}" if plan.note else ""
    return (f"{head}  {body}\n"
            f"　{len(plan.tickets)}点 × ¥{plan.unit_yen:,} = **¥{plan.stake_yen:,}**"
            f"（軸の確率 {plan.p_axis * 100:.0f}%）{note}")


def _trifecta_lines(plan, names: dict) -> str:
    """3連単 1着固定流しの買い方。全点を書き出すと長いので、軸・相手・点数・金額で示す。"""
    axis = f"{plan.axis}番 {names.get(plan.axis, '')}".rstrip()
    partners = ",".join(str(h) for h in plan.partners)
    return (f"🎯 **3連単 1着固定流し**  1着 {axis} → 2・3着 {partners}\n"
            f"　{len(plan.tickets)}点 × ¥{plan.unit_yen:,} = **¥{plan.stake_yen:,}**"
            f"（1番人気の単勝確率 {plan.p_axis * 100:.0f}%、人気×頭数の勝率表）")


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
