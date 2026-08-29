"""当日成績表（NAR `RaceMarkTable`）の解析。

ライブ層（`entry_result_live`）はここからしか埋まらない。確定層は月次
ファイル由来で翌日以降にしか入らないので、**当日の先行レースを as-of 履歴に
入れる唯一の経路**がこの表になる。ここが空だと、同日の先行レースまで含めて
特徴量を作っている学習時に対し、推論時だけ騎手・調教師の累積が足りない状態
になる（train-serving skew）。

代理キーはこの表からは作らない。成績表には父名も生年月日も載っておらず、
`horse_sk` は (馬名, 生年月日, 父) から作るので、ここで組み立てると同じ馬に
別のキーが振られ、過去成績が1件も引けなくなる。キーは出馬表側から結合する。

着順・タイムを持つ「結果」なので、推論の入力には絶対に渡さない。履歴専用。
"""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd

# 「1:17.9」「17.9」
_TIME_RE = re.compile(r"(?:(\d+):)?(\d+)\.(\d)")
# 着順は数字。取消・除外・中止は「取」「除」「中」などが入る
_POS_RE = re.compile(r"^\s*(\d+)\s*$")

RESULT_COLUMNS = ["horse_no", "finish_pos", "is_win", "time_sec"]


def _time_sec(raw: object) -> float | None:
    m = _TIME_RE.search(str(raw))
    if not m:
        return None
    return int(m.group(1) or 0) * 60 + int(m.group(2)) + int(m.group(3)) / 10.0


def _finish_pos(raw: object) -> int | None:
    m = _POS_RE.match(str(raw))
    return int(m.group(1)) if m else None


def _result_table(html: str) -> pd.DataFrame | None:
    """成績表の本体を選ぶ。払戻・ラップ・通過順の表を掴まないよう列名で判定する。"""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("着順" in c for c in cols) and any("馬番" in c for c in cols):
            return t
    return None


def parse(html: str) -> pd.DataFrame:
    """成績表 1 レース分の着順・タイム。

    まだ確定していなければ空を返す。開催中は必ず「これから走るレース」が
    あるので、そこを例外にすると当日の更新が丸ごと止まる。
    """
    t = _result_table(html)
    if t is None or t.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    t = t.copy()
    t.columns = [str(c) for c in t.columns]
    pos_col = next((c for c in t.columns if "着順" in c), None)
    no_col = next((c for c in t.columns if "馬番" in c), None)
    time_col = next((c for c in t.columns if "タイム" in c), None)
    if pos_col is None or no_col is None:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    rows = []
    for _, r in t.iterrows():
        pos = _finish_pos(r[pos_col])
        horse_no = pd.to_numeric(r[no_col], errors="coerce")
        # 取消・除外・中止の行は履歴に入れない。走っていない
        if pos is None or pd.isna(horse_no):
            continue
        rows.append({
            "horse_no": int(horse_no),
            "finish_pos": pos,
            "is_win": int(pos == 1),
            "time_sec": _time_sec(r[time_col]) if time_col else None,
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
