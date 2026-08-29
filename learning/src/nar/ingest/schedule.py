"""月別開催日程ページから競馬場マスタを自動生成する。

設計書 §「キー設計」は競馬場コードのハードコードを禁じている。理由は廃止場の
取りこぼしで、実際 1998-2013 の実データには福山・荒尾・上山など16の廃止場が含まれ、
固定辞書だけでは 28年ぶんのうち相当な期間が silver に落ちなかった。

日程ページの各行は `<tr><td>競馬場名</td>` に続けて
`/KeibaWeb/TodayRaceInfo/RaceList?...&k_babaCode=NN` へのリンクを持つ。
開催の無い月は行自体が出ないので、廃止場を拾うには過去月を遡る必要がある。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import UnknownTrackError
from .client import NarClient

log = logging.getLogger(__name__)

SCHEDULE_URL = "https://www.keiba.go.jp/KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop"

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_NAME_RE = re.compile(r"<td[^>]*>\s*([^<\s][^<]*?)\s*</td>", re.S)
_CODE_RE = re.compile(r"k_babaCode=(\d+)")


def parse_codes(html: str) -> dict[str, int]:
    """1ページぶんの 競馬場名 → k_babaCode。

    行の先頭セルが名前、同じ行のリンクにコードが入る。名前セルとコードが
    同一行に揃っている行だけを採る（曜日ヘッダ行などはコードを持たない）。
    """
    out: dict[str, int] = {}
    for row in _ROW_RE.findall(html):
        codes = _CODE_RE.findall(row)
        if not codes:
            continue
        name_m = _NAME_RE.search(row)
        if name_m is None:
            continue
        name = name_m.group(1).strip()
        if not name or "k_babaCode" in name:
            continue
        uniq = set(codes)
        if len(uniq) != 1:
            log.warning("行 %r に複数コード %s。読み飛ばします", name, sorted(uniq))
            continue
        out[name] = int(codes[0])
    return out


@dataclass
class MasterBuild:
    mapping: dict[str, int]
    months_scanned: int
    conflicts: dict[str, list[int]]


def build_track_master(
    client: NarClient, months: list[tuple[int, int]],
    required_names: set[str] | None = None,
) -> MasterBuild:
    """月を順に走査してマスタを作る。

    同じ名前が異なるコードに割り当てられていたら例外。race_id の一意性が
    崩れるので、黙って後勝ちにしてはいけない。
    required_names を渡すと全部揃った時点で打ち切る（無駄なアクセスを避ける）。
    """
    mapping: dict[str, int] = {}
    seen: dict[str, set[int]] = {}
    scanned = 0
    for year, month in months:
        res = client.fetch(SCHEDULE_URL, {"k_year": year, "k_month": month})
        scanned += 1
        for name, code in parse_codes(res.content.decode("utf-8", errors="replace")).items():
            seen.setdefault(name, set()).add(code)
            mapping.setdefault(name, code)
        if required_names and required_names <= set(mapping):
            break

    conflicts = {n: sorted(c) for n, c in seen.items() if len(c) > 1}
    if conflicts:
        raise UnknownTrackError(
            f"競馬場名が複数コードに割り当てられています: {conflicts}。"
            "race_id が一意でなくなるため停止します。"
        )
    return MasterBuild(mapping, scanned, conflicts)


def save(mapping: dict[str, int], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
                 encoding="utf-8")
    return str(p)


def load(path: str | Path) -> dict[str, int] | None:
    p = Path(path)
    if not p.exists():
        return None
    return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
