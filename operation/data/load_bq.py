"""書き出した Parquet を BigQuery の確定層へ投入する。

最初の1本だけ WRITE_TRUNCATE、以降は WRITE_APPEND。途中で失敗しても
同じコマンドをやり直せば同じ状態に収束する。
"""
import sys
from pathlib import Path

from google.cloud import bigquery

PROJECT = "sample-335613"
FULL = f"{PROJECT}.nar_ops.entry_result_final"
SRC = Path("/work/bq_export")

client = bigquery.Client(project=PROJECT)
files = sorted(SRC.glob("y*.parquet"))
print(f"{len(files)} ファイルを投入します", flush=True)

for i, path in enumerate(files):
    cfg = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=("WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND"))
    with path.open("rb") as fh:
        job = client.load_table_from_file(fh, FULL, job_config=cfg)
    job.result()
    print(f"  {path.name}: {job.output_rows:,} 行", flush=True)

got = list(client.query(
    f"SELECT COUNT(*) AS n FROM `{FULL}` WHERE race_date >= '1990-01-01'"
).result())[0].n
print(f"BigQuery 確定層: {got:,} 行")
sys.exit(0)
