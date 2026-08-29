"""ローカル確定層を BigQuery へ一括投入する。

DataFrame を1回で送ると数百MBのメモリを掴むので、年ごとに Parquet へ
書き出してロードする。パーティション（race_date）と require_partition_filter は
既存テーブルの設定をそのまま使う。
"""
import os
import sys
import tempfile
from pathlib import Path

import duckdb
from google.cloud import bigquery

PROJECT = "sample-335613"
DATASET = "nar_ops"
TABLE = "entry_result_final"
DB = "/work/nar_ops.duckdb"

client = bigquery.Client(project=PROJECT)
full = f"{PROJECT}.{DATASET}.{TABLE}"

con = duckdb.connect(DB, read_only=True)
years = [r[0] for r in con.execute(
    "SELECT DISTINCT EXTRACT(year FROM race_date) AS y FROM entry_result_final "
    "ORDER BY y").fetchall()]
print(f"{len(years)} 年分を投入します", flush=True)

total = 0
with tempfile.TemporaryDirectory() as tmp:
    for i, year in enumerate(years, 1):
        path = Path(tmp) / f"y{int(year)}.parquet"
        con.execute(
            "COPY (SELECT * FROM entry_result_final "
            "WHERE EXTRACT(year FROM race_date) = ?) TO ? (FORMAT PARQUET)",
            [year, str(path)])
        # 最初の年だけ truncate。以降は append（何度も流しても重複しない）
        disposition = ("WRITE_TRUNCATE" if i == 1 else "WRITE_APPEND")
        with path.open("rb") as fh:
            job = client.load_table_from_file(
                fh, full, job_config=bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=disposition))
        job.result()
        n = con.execute(
            "SELECT COUNT(*) FROM entry_result_final "
            "WHERE EXTRACT(year FROM race_date) = ?", [year]).fetchone()[0]
        total += n
        print(f"  {int(year)}: {n:,} 行  (累計 {total:,})", flush=True)

con.close()
got = list(client.query(
    f"SELECT COUNT(*) AS n FROM `{full}` WHERE race_date >= '1990-01-01'").result())[0].n
print(f"投入完了: ローカル {total:,} 行 / BigQuery {got:,} 行")
sys.exit(0 if got == total else 1)
