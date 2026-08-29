"""運用 DWH のスキーマ。

確定層とライブ層を分けるのが中核。特徴量計算は両層の UNION を参照し、
`start_ts < 対象レースの発走時刻` で厳密にフィルタする（設計書 §2.2）。
"""

from __future__ import annotations

import re

from .backend import Warehouse

DDL = """
-- 確定層: 月次ファイル由来。日次 02:40 の取り込みで更新
CREATE TABLE IF NOT EXISTS entry_result_final (
  race_id      VARCHAR, horse_no INTEGER, horse_sk VARCHAR,
  jockey_sk    VARCHAR, trainer_sk VARCHAR, sire_sk VARCHAR,
  race_date    DATE, start_ts TIMESTAMP, baba_code INTEGER, distance INTEGER,
  finish_pos   INTEGER, is_win INTEGER, time_sec DOUBLE, speed_index DOUBLE,
  -- NAR 申告の累積成績8列。EDA（LK-09/10/11）で as-of-race を確定させたので
  -- 特徴量として使う。当日の出馬表にも同じ8列が載るため、学習と推論で同じ値が取れる。
  -- 履歴側にも持たないと、対象レース行だけ列があって concat で落ちる。
  jockey_record VARCHAR, all_record VARCHAR,
  dirt_left_record VARCHAR, dirt_right_record VARCHAR,
  track_record VARCHAR, dist_record VARCHAR,
  best_time VARCHAR, best_time_good VARCHAR,
  -- レース属性。`ダート左/右成績` は回り、`最高タイム良馬場` は馬場状態が条件
  turn VARCHAR, baba_condition VARCHAR,
  source_sha256 VARCHAR, merged_at TIMESTAMP,
  PRIMARY KEY (race_id, horse_no)
);

-- ライブ層: 当日ファイル由来。当日の確定した先行レース結果のみ
CREATE TABLE IF NOT EXISTS entry_result_live (
  race_id      VARCHAR, horse_no INTEGER, horse_sk VARCHAR,
  jockey_sk    VARCHAR, trainer_sk VARCHAR, sire_sk VARCHAR,
  race_date    DATE, start_ts TIMESTAMP, baba_code INTEGER, distance INTEGER,
  finish_pos   INTEGER, is_win INTEGER, time_sec DOUBLE, speed_index DOUBLE,
  -- NAR 申告の累積成績8列。EDA（LK-09/10/11）で as-of-race を確定させたので
  -- 特徴量として使う。当日の出馬表にも同じ8列が載るため、学習と推論で同じ値が取れる。
  -- 履歴側にも持たないと、対象レース行だけ列があって concat で落ちる。
  jockey_record VARCHAR, all_record VARCHAR,
  dirt_left_record VARCHAR, dirt_right_record VARCHAR,
  track_record VARCHAR, dist_record VARCHAR,
  best_time VARCHAR, best_time_good VARCHAR,
  -- レース属性。`ダート左/右成績` は回り、`最高タイム良馬場` は馬場状態が条件
  turn VARCHAR, baba_condition VARCHAR,
  captured_at  TIMESTAMP,
  PRIMARY KEY (race_id, horse_no)
);

-- 後日訂正の監査。旧値を必ず残す（DB-07）
CREATE TABLE IF NOT EXISTS entry_result_audit (
  race_id VARCHAR, horse_no INTEGER, changed_at TIMESTAMP,
  column_name VARCHAR, old_value VARCHAR, new_value VARCHAR, reason VARCHAR
);

CREATE TABLE IF NOT EXISTS race_schedule (
  race_id VARCHAR PRIMARY KEY, race_date DATE, baba_code INTEGER, race_no INTEGER,
  start_ts TIMESTAMP, status VARCHAR, updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds_snapshot (
  race_id VARCHAR, horse_no INTEGER, race_date DATE,
  odds_win DOUBLE, captured_at TIMESTAMP
);

-- 推論時点の DB 状態を後から再現するためのバイテンポラル記録（設計書 §2.3）
CREATE TABLE IF NOT EXISTS feature_snapshot (
  race_id VARCHAR, horse_no INTEGER,
  as_of_date DATE,
  computed_at TIMESTAMP,       -- 特徴量を計算した時刻
  as_of_ts TIMESTAMP,          -- as-of の基準時刻（= 発走時刻）
  db_watermark TIMESTAMP,      -- 参照した DB の最終更新時刻
  model_release VARCHAR,
  features VARCHAR,            -- JSON
  feature_spec_hash VARCHAR
);

CREATE TABLE IF NOT EXISTS prediction (
  race_id VARCHAR, horse_no INTEGER, race_date DATE,
  model_release VARCHAR, track_used VARCHAR,
  p_win DOUBLE, p_market DOUBLE, ev DOUBLE, ev_adjusted DOUBLE,
  computed_at TIMESTAMP, is_shadow BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS bet_candidate (
  race_id VARCHAR, horse_no INTEGER, race_date DATE,
  model_release VARCHAR, bet_type VARCHAR,
  stake_yen INTEGER, ev DOUBLE, ev_adjusted DOUBLE, kelly DOUBLE,
  computed_at TIMESTAMP
);

-- Discord は冪等にできないので送信済みを記録する（IN-08 / DC-07）
CREATE TABLE IF NOT EXISTS notification_log (
  race_id VARCHAR, channel VARCHAR, dedupe_key VARCHAR,
  sent_at TIMESTAMP, start_ts TIMESTAMP, status VARCHAR
);

CREATE TABLE IF NOT EXISTS pnl_daily (
  business_date DATE PRIMARY KEY,
  n_races INTEGER, n_bets INTEGER, stake_yen BIGINT, return_yen BIGINT,
  roi DOUBLE, hit_rate DOUBLE, brier DOUBLE, coverage DOUBLE,
  is_paper BOOLEAN, recomputed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_run (
  job_name VARCHAR, run_at TIMESTAMP, business_date DATE, status VARCHAR, detail VARCHAR
);

CREATE TABLE IF NOT EXISTS skew_check (
  business_date DATE, checked_at TIMESTAMP, n_compared INTEGER,
  n_mismatch INTEGER, mismatch_columns VARCHAR, verdict VARCHAR
);
"""

# 履歴ビューが返す列。学習側の silver と同じ名前に直して渡すため、DB 上の
# ASCII 列名と日本語の対応をここ1か所に置く。
HISTORY_COLUMNS = """race_id, horse_no, horse_sk, jockey_sk, trainer_sk, sire_sk,
       race_date, start_ts, baba_code, distance,
       finish_pos, is_win, time_sec, speed_index,
       jockey_record, all_record, dirt_left_record, dirt_right_record,
       track_record, dist_record, best_time, best_time_good,
       turn, baba_condition"""

# DB の列名（ASCII）→ 学習側の列名（月次ファイルの日本語ヘッダ）
RECORD_COLUMN_MAP = {
    "jockey_record": "騎手成績",
    "all_record": "全成績",
    "dirt_left_record": "ダート左成績",
    "dirt_right_record": "ダート右成績",
    "track_record": "当競馬場成績",
    "dist_record": "うち当距離成績",
    "best_time": "最高タイム",
    "best_time_good": "最高タイム良馬場",
}

# 二層の UNION。ライブ層は確定層に無いレースだけを出す（DB-02）
HISTORY_VIEW = f"""
CREATE OR REPLACE VIEW entry_history AS
SELECT {HISTORY_COLUMNS}, 'final' AS src
FROM entry_result_final
UNION ALL
SELECT {HISTORY_COLUMNS}, 'live' AS src
FROM entry_result_live
WHERE race_id NOT IN (SELECT race_id FROM entry_result_final);
"""


def table_columns(table: str) -> list[str]:
    """DDL からテーブルの列名を取る。

    `SELECT * LIMIT 0` で引くと BigQuery では課金上限の指定が要り、DuckDB でも
    テーブル全体を触ることになる。列名は DDL が唯一の情報源なので、そこから読む。
    """
    body = re.search(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(table)}\s*\((.*?)\n\);",
        DDL, re.S)
    if body is None:
        raise KeyError(f"{table} は DDL にありません")
    # コメント行を先に落とす。チャンク単位で落とすと、コメントの直後に書かれた
    # 列（`-- 説明` の次行にある jockey_record など）ごと消える。
    text = "\n".join(re.sub(r"--.*$", "", line) for line in body.group(1).splitlines())
    cols: list[str] = []
    for chunk in _split_top_level(text):
        head = chunk.strip()
        if not head or head.upper().startswith(("PRIMARY KEY", "FOREIGN KEY", "UNIQUE")):
            continue
        cols.append(head.split()[0])
    return cols


def _split_top_level(text: str) -> list[str]:
    """括弧の内側のカンマで割らないための分割。"""
    out, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return out


def create_all(wh: Warehouse) -> None:
    for stmt in DDL.split(";"):
        if stmt.strip():
            wh.execute(stmt)
    wh.execute(HISTORY_VIEW)
