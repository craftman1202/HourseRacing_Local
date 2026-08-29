"""払戻テーブルの縦持ち化。

54列の横持ちのままだと収支照合が地獄になる。オッズ側は元から縦持ちなので、
揃えておけば突合が単純な JOIN になる（TR-08/09）。
"""

from __future__ import annotations

import re

import pandas as pd

BET_TYPES = (
    "単勝", "複勝", "枠連", "馬連", "馬単", "ワイド", "3連複", "3連単",
)

LONG_COLUMNS = [
    "race_id", "bet_type", "comb_1", "comb_2", "comb_3",
    "payout_yen", "popularity", "dead_heat_seq",
]

_CELL_RE = re.compile(r"^(?P<bet>.+?)(?P<idx>\d*)_(?P<field>組番|払戻金|人気)$")


def unpivot(wide: pd.DataFrame, race_id_col: str = "race_id") -> pd.DataFrame:
    """横持ち払戻 → 縦持ち。

    列名は `{券種}{通番}_{組番|払戻金|人気}` を想定する（例: `複勝2_払戻金`）。
    同着で同一券種が複数行になる仕様は dead_heat_seq で吸収する。
    """
    records: list[dict[str, object]] = []
    parsed: dict[tuple[str, str], dict[str, str]] = {}
    for col in wide.columns:
        m = _CELL_RE.match(str(col))
        if m and m.group("bet") in BET_TYPES:
            parsed.setdefault((m.group("bet"), m.group("idx")), {})[m.group("field")] = col

    for _, row in wide.iterrows():
        seq: dict[str, int] = {}
        for (bet, idx), fields in sorted(parsed.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            comb = row.get(fields.get("組番"))
            yen = row.get(fields.get("払戻金"))
            if _blank(comb) or _blank(yen):
                continue
            c1, c2, c3 = _split_comb(str(comb))
            seq[bet] = seq.get(bet, 0) + 1
            records.append({
                "race_id": row[race_id_col],
                "bet_type": bet,
                "comb_1": c1, "comb_2": c2, "comb_3": c3,
                "payout_yen": int(float(yen)),
                "popularity": _int_or_none(row.get(fields.get("人気"))),
                "dead_heat_seq": seq[bet],
            })
    return pd.DataFrame(records, columns=LONG_COLUMNS)


def _blank(v: object) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in {"", "nan"}


def _split_comb(s: str) -> tuple[int | None, int | None, int | None]:
    parts = [int(p) for p in re.split(r"[-–—>=]", s.strip()) if p.strip().isdigit()]
    parts += [None] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def _int_or_none(v: object) -> int | None:
    return None if _blank(v) else int(float(v))


def assert_roundtrip(wide: pd.DataFrame, long: pd.DataFrame) -> None:
    """変換前後で総払戻金額が一致すること（TR-08）。"""
    yen_cols = [c for c in wide.columns if str(c).endswith("_払戻金")]
    before = pd.to_numeric(
        wide[yen_cols].stack(), errors="coerce"
    ).dropna().sum()
    after = long["payout_yen"].sum()
    if int(before) != int(after):
        raise ValueError(f"払戻総額が不一致: 変換前 {int(before)} / 変換後 {int(after)}")


def assert_no_duplicate_combination(long: pd.DataFrame) -> None:
    """同一 (race_id, bet_type, 組番) の重複が無いこと（TR-09）。"""
    key = ["race_id", "bet_type", "comb_1", "comb_2", "comb_3"]
    dup = long[long.duplicated(key, keep=False)]
    if len(dup):
        raise ValueError(f"同一組番の重複が {len(dup)} 件あります。同着処理を確認してください。")
