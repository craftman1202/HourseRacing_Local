"""一括投票テキストの生成（DC-08）。

実運用で最も効く機能。ここが手作業だと入力ミスで損失が出る。
往復（生成 → パース）で元の候補が復元できることをテストで固定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 券種記号。サイトごとに違うので写像を1か所に集約する
RAKUTEN_BET = {"単勝": "T", "複勝": "F", "馬連": "U", "馬単": "E",
               "ワイド": "W", "3連複": "S", "3連単": "SE"}
SPAT4_BET = {"単勝": "01", "複勝": "02", "馬連": "05", "馬単": "06",
             "ワイド": "07", "3連複": "08", "3連単": "09"}
UNIT_YEN = 100


@dataclass(frozen=True)
class Slip:
    race_id: str
    bet_type: str
    horse_no: int
    stake_yen: int

    @property
    def units(self) -> int:
        return self.stake_yen // UNIT_YEN


def to_slips(candidates: pd.DataFrame, race_id: str, bet_type: str = "単勝") -> list[Slip]:
    return [
        Slip(race_id, bet_type, int(r["horse_no"]), int(r["stake_yen"]))
        for _, r in candidates.iterrows() if int(r["stake_yen"]) > 0
    ]


def _race_parts(race_id: str) -> tuple[str, str, str]:
    """race_id = 競馬場コード(2) + 年月日(8) + レース番号(2)。"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"race_id が12桁数字ではありません: {race_id!r}")
    return race_id[:2], race_id[2:10], race_id[10:12]


def format_rakuten(slips: list[Slip]) -> str:
    """楽天競馬形式: `場コード,日付,R,券種記号,馬番,口数`。"""
    lines = []
    for s in slips:
        baba, ymd, rno = _race_parts(s.race_id)
        lines.append(f"{baba},{ymd},{int(rno)},{RAKUTEN_BET[s.bet_type]},"
                     f"{s.horse_no},{s.units}")
    return "\n".join(lines)


def format_spat4(slips: list[Slip]) -> str:
    """SPAT4 形式: 固定長寄りの `場(2)日付(8)R(2)券種(2)馬番(2)口数(4)`。"""
    lines = []
    for s in slips:
        baba, ymd, rno = _race_parts(s.race_id)
        lines.append(f"{baba}{ymd}{rno}{SPAT4_BET[s.bet_type]}"
                     f"{s.horse_no:02d}{s.units:04d}")
    return "\n".join(lines)


def parse_rakuten(text: str) -> list[Slip]:
    inv = {v: k for k, v in RAKUTEN_BET.items()}
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        baba, ymd, rno, bet, no, units = line.split(",")
        out.append(Slip(f"{baba}{ymd}{int(rno):02d}", inv[bet], int(no),
                        int(units) * UNIT_YEN))
    return out


def parse_spat4(text: str) -> list[Slip]:
    inv = {v: k for k, v in SPAT4_BET.items()}
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        baba, ymd, rno, bet, no, units = (
            line[0:2], line[2:10], line[10:12], line[12:14], line[14:16], line[16:20])
        out.append(Slip(f"{baba}{ymd}{rno}", inv[bet], int(no), int(units) * UNIT_YEN))
    return out


FORMATS = {
    "rakuten": (format_rakuten, parse_rakuten),
    "spat4": (format_spat4, parse_spat4),
}


def render(slips: list[Slip], site: str) -> str:
    if site not in FORMATS:
        raise ValueError(f"未知の投票サイト形式: {site}（対応: {sorted(FORMATS)}）")
    return FORMATS[site][0](slips)


def roundtrip(slips: list[Slip], site: str) -> list[Slip]:
    fmt, parse = FORMATS[site]
    return parse(fmt(slips))
