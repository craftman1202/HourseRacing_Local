"""確定層への MERGE と後日訂正の扱い。

同一データを2回 MERGE しても行数・全列ハッシュが変わらないこと（DB-01）と、
訂正が入ったときに旧値が監査テーブルに残ること（DB-07）を保証する。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..clock import Clock, SystemClock
from .backend import Warehouse
from .schema import table_columns as schema_table_columns
from .types import assert_utc_start_ts, canonicalize_frame, comparable

log = logging.getLogger(__name__)

FINAL_COLUMNS = [
    "race_id", "horse_no", "horse_sk", "jockey_sk", "trainer_sk", "sire_sk",
    "race_date", "start_ts", "baba_code", "distance",
    "finish_pos", "is_win", "time_sec", "speed_index",
]
# NAR 申告の累積成績8列とレース属性。特徴量には使うが、取り込み元によっては
# 揃わないことがある。必須にはせず、来ていれば保存し比較対象にもする。
OPTIONAL_COLUMNS = [
    "jockey_record", "all_record", "dirt_left_record", "dirt_right_record",
    "track_record", "dist_record", "best_time", "best_time_good",
    "turn", "baba_condition",
]


def _columns_of(df: pd.DataFrame) -> list[str]:
    missing = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"必須列がありません: {missing}")
    return FINAL_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
# 訂正の監視対象。着順の事後訂正（降着・失格）と賞金訂正が実際に起こる
TRACKED_FOR_CORRECTION = ("finish_pos", "is_win", "time_sec")


@dataclass
class MergeResult:
    inserted: int
    updated: int
    unchanged: int
    corrections: int

    @property
    def is_noop(self) -> bool:
        return self.inserted == 0 and self.updated == 0


def table_hash(wh: Warehouse, table: str, columns: list[str] | None = None) -> str:
    """全列ハッシュ。冪等性の判定に使う（DB-01）。"""
    df = wh.table(table)
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    cols = columns or [c for c in df.columns if c not in ("merged_at", "captured_at")]
    body = df[cols].sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    return hashlib.sha256(
        body.to_csv(index=False, float_format="%.12g").encode("utf-8")).hexdigest()


def merge_final(wh: Warehouse, incoming: pd.DataFrame, source_sha256: str = "",
                clock: Clock | None = None, reason: str = "monthly-merge") -> MergeResult:
    """確定層への冪等 MERGE。

    既存行と値が同じなら触らない。違えば更新し、**旧値を監査テーブルに残す**。
    訂正を黙って上書きすると、過去の推論がどの数字を見ていたか追えなくなる。
    """
    clock = clock or SystemClock()
    now = clock.now()
    # DB 境界の型ゆれ（tz 落ち・date/datetime の混在）を先に潰す。
    # これを飛ばすと往復のたびに全行が「変化あり」と判定される。
    cols = _columns_of(incoming)
    incoming = canonicalize_frame(incoming[cols])
    assert_utc_start_ts(incoming)
    # `wh.table(...)` は全件ダンプで、ローカル Warehouse にしか実装が無い
    # （BigQueryWarehouse は意図的に持たない — パーティション必須/bytes_billed
    # 制約を回避してしまうため）。`incoming` の race_date だけに絞って
    # `query()` で取る。これなら両バックエンドで動き、DB-09 の制約にも従う
    # （実運用で BigQueryWarehouse に対して初めて実行した際に
    # AttributeError で気付いた — 2026-08-29）。
    days = ", ".join("'%s'" % d for d in sorted({str(d) for d in incoming["race_date"]}))
    existing = (wh.query(f"SELECT * FROM entry_result_final WHERE race_date IN ({days})")
               if days else pd.DataFrame())

    if existing.empty:
        _insert(wh, incoming, source_sha256, now)
        return MergeResult(len(incoming), 0, 0, 0)

    key = ["race_id", "horse_no"]
    # 比較は型を潰した写しの上だけで行い、書き込みは元の incoming から取る。
    # 比較用と書き込み用を混ぜると suffix 地獄になり、どちらを書いたか分からなくなる。
    compare_cols = [c for c in cols if c in existing.columns]
    left = comparable(incoming, compare_cols)
    right = comparable(existing, compare_cols)
    cmp = left.merge(right, on=key, how="left", suffixes=("", "_old"), indicator=True)

    is_new = cmp["_merge"] == "left_only"
    seen = cmp[~is_new]

    changed = pd.Series(False, index=seen.index)
    corrections: list[dict] = []
    for col in compare_cols:
        if col in key:
            continue
        differs = ~_equalish(seen[f"{col}_old"], seen[col])
        changed |= differs
        if col in TRACKED_FOR_CORRECTION:
            for _, r in seen[differs].iterrows():
                corrections.append({
                    "race_id": r["race_id"], "horse_no": int(r["horse_no"]),
                    "changed_at": now, "column_name": col,
                    "old_value": str(r[f"{col}_old"]), "new_value": str(r[col]),
                    "reason": reason,
                })

    new_keys = set(map(tuple, cmp.loc[is_new, key].to_numpy()))
    changed_keys = set(map(tuple, seen.loc[changed, key].to_numpy()))
    idx = pd.MultiIndex.from_frame(incoming[key])
    to_insert = incoming[[k in new_keys for k in idx]]
    to_update = incoming[[k in changed_keys for k in idx]]

    if len(to_insert):
        _insert(wh, to_insert, source_sha256, now)
    if len(to_update):
        # 一時テーブルの登録は DuckDB 固有。両バックエンドで動くよう、
        # 対象キーを値で並べた条件にする。更新は1日あたり数千行に収まる。
        pairs = ", ".join("('%s', %d)" % (r.race_id, int(r.horse_no))
                          for r in to_update[key].itertuples())
        # race_date の条件は必須。entry_result_final はパーティション必須で、
        # 条件が無いと BigQuery が実行を拒否する（訂正の反映が全部落ちる）。
        days = ", ".join("'%s'" % d for d in sorted({str(d) for d in
                                                    to_update["race_date"]}))
        wh.execute("DELETE FROM entry_result_final "
                   f"WHERE race_date IN ({days}) "
                   f"AND (race_id, horse_no) IN ({pairs})")
        _insert(wh, to_update, source_sha256, now)
    if corrections:
        wh.insert_frame("entry_result_audit", pd.DataFrame(corrections))

    return MergeResult(len(to_insert), len(to_update),
                       len(seen) - len(to_update), len(corrections))


def _insert(wh: Warehouse, df: pd.DataFrame, source_sha256: str, now: datetime) -> None:
    payload = canonicalize_frame(df)
    payload["source_sha256"] = source_sha256
    payload["merged_at"] = pd.Timestamp(now).tz_convert("UTC").tz_localize(None)
    insert_frame(wh, "entry_result_final", payload)





def insert_frame(wh: Warehouse, table: str, df: pd.DataFrame) -> None:
    """列名を明示して差し込む。

    `INSERT INTO t SELECT * FROM _ins` は列数と列順が完全一致していることを
    暗黙に要求する。テーブルに列を1つ足しただけで、無関係な呼び出し側が
    「26 columns but 24 values」で落ちる。テーブル側の列を引いて揃える。
    足りない列は NULL。余分な列は落とす（テーブルに無いものは入らない）。
    """
    cols = schema_table_columns(table)
    payload = df.reindex(columns=cols)
    extra = [c for c in df.columns if c not in cols]
    if extra:
        log.debug("%s に無い列を落としました: %s", table, extra)
    wh.insert_frame(table, payload)


def _equalish(a: pd.Series, b: pd.Series) -> pd.Series:
    """浮動小数の 1e-12 差で「訂正あり」と誤検出しないための比較。"""
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        both_na = a.isna() & b.isna()
        close = (a.fillna(0) - b.fillna(0)).abs() <= 1e-9
        return both_na | (close & ~(a.isna() ^ b.isna()))
    return a.astype(str) == b.astype(str)


def refresh_live(wh: Warehouse, incoming: pd.DataFrame, clock: Clock | None = None) -> int:
    """ライブ層の置き換え。

    ライブ層は当日ぶんの一時的な写しなので、履歴を残さず日単位で入れ替える。
    確定層に昇格したレースは entry_history 側で自動的に吸収される（DB-02）。
    """
    clock = clock or SystemClock()
    if incoming.empty:
        return 0
    payload = canonicalize_frame(incoming[_columns_of(incoming)])
    assert_utc_start_ts(payload)
    payload["captured_at"] = pd.Timestamp(clock.now()).tz_convert("UTC").tz_localize(None)
    dates = sorted({str(d) for d in payload["race_date"]})
    wh.execute("DELETE FROM entry_result_live WHERE race_date IN "
               f"({','.join(repr(d) for d in dates)})")
    insert_frame(wh, "entry_result_live", payload)
    return len(payload)
