"""bronze 層: ZIP 展開 ＋ デコードのみ。型付けはしない。

bronze を silver から分離するのは、CSV パースの失敗切り分けを容易にするため。
展開とデコードだけを済ませた層があると「NAR の仕様が変わったのか、自分のパーサが
バグっているのか」を即座に判別できる（設計書 §2）。
"""

from __future__ import annotations

import csv
import io
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..ingest.unzip import extract
from ..io.store import Store

log = logging.getLogger(__name__)

# ZIP 内のファイル名から種別を判定する
KIND_BY_SUFFIX = {
    "racelist": "race", "horselist": "entry", "payback": "payout", "odds": "odds",
}


@dataclass
class BronzeTable:
    kind: str
    ym: str
    frame: pd.DataFrame
    codec: str
    source: str
    part: str = "00"


def part_of(inner_name: str) -> str:
    """ZIP 内ファイル名から分割番号を取る。

    オッズは1か月あたり `202602_01_odds.csv` 〜 `_03_odds.csv` の3本が
    同じ ZIP に入る。固定名で書くと相互に上書きされて2/3が消える。
    """
    m = re.match(r"^\d{6}_(\d{2})_", Path(inner_name).name)
    return m.group(1) if m else "00"


def classify(inner_name: str) -> str | None:
    stem = Path(inner_name).stem.lower()
    for suffix, kind in KIND_BY_SUFFIX.items():
        if stem.endswith(suffix):
            return kind
    return None


def to_frame(text: str) -> pd.DataFrame:
    """全列を文字列のまま読む。

    ここで型を推論すると、空文字とゼロ、先頭ゼロ付きコードの区別が失われる。
    型付けは silver の責務。
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame()
    header, body = rows[0], rows[1:]
    # 行ごとの列数のブレを許容して詰める（欠損は空文字）
    width = len(header)
    fixed = [(r + [""] * width)[:width] for r in body]
    return pd.DataFrame(fixed, columns=header, dtype=str)


def build_from_zip(zip_bytes: bytes, ym: str, source: str = "") -> list[BronzeTable]:
    out: list[BronzeTable] = []
    for csvfile in extract(zip_bytes):
        kind = classify(csvfile.name)
        if kind is None:
            log.warning("種別を判定できないファイルを飛ばします: %s", csvfile.name)
            continue
        frame = to_frame(csvfile.text)
        if csvfile.replaced:
            log.warning("%s は replace デコードを経由しています", csvfile.name)
        out.append(BronzeTable(kind, ym, frame, csvfile.codec,
                               source or csvfile.name, part_of(csvfile.name)))
    return out


def write(store: Store, table: BronzeTable) -> str:
    path = Path(store.path("bronze", table.kind, f"ym={table.ym}",
                           f"part-{table.part}.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.frame.to_parquet(path, index=False)
    return str(path)


def read(store: Store, kind: str, ym: str) -> pd.DataFrame:
    """その月の全 part を結合して返す（オッズは3分割で入っている）。"""
    base = Path(store.path("bronze", kind, f"ym={ym}"))
    if not base.exists():
        return pd.DataFrame()
    parts = sorted(base.glob("*.parquet"))
    if not parts:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in parts]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def available_months(store: Store, kind: str) -> list[str]:
    base = Path(store.path("bronze", kind))
    if not base.exists():
        return []
    return sorted(p.name.split("=", 1)[1] for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("ym="))
