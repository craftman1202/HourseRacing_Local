"""BigQuery の代替実装（DuckDB）。

テスト仕様 §1.3 が定める L1 代替そのもの。BigQuery 固有の制約
（`require_partition_filter`、`maximum_bytes_billed`）を**実際に強制する**ラッパを
被せておくことで、CO-01 / DB-09 をローカルで検証できる。

本番で BigQuery に差し替えるときは `query()` の中身だけを置き換える。
制約チェックは共通なので、そのまま残る。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from ..errors import BytesBilledUnbounded, PartitionFilterRequired

# パーティション列を持つテーブル。クエリで必ず絞られていること（DB-09）
PARTITIONED = {
    "entry_result_final": "race_date",
    "entry_result_live": "race_date",
    "feature_snapshot": "as_of_date",
    "prediction": "race_date",
    "bet_candidate": "race_date",
    "odds_snapshot": "race_date",
}


@dataclass
class QueryStats:
    sql: str
    bytes_scanned: int
    rows: int


@dataclass
class Warehouse:
    """クエリ実行の唯一の入口。制約チェックをここに集約する。"""

    path: str = ":memory:"
    max_bytes_billed: int | None = None
    stats: list[QueryStats] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path)

    def close(self) -> None:
        self.con.close()

    # ------------------------------------------------------------------ 実行
    def query(self, sql: str, params: list | None = None,
              max_bytes_billed: int | None = None,
              allow_full_scan: bool = False) -> pd.DataFrame:
        """SELECT 実行。

        `max_bytes_billed` は必ず与える。未設定のジョブを実行可能にしておくと、
        1本の事故クエリで無料枠を焼き切る（CO-01）。
        """
        limit = max_bytes_billed if max_bytes_billed is not None else self.max_bytes_billed
        if limit is None:
            raise BytesBilledUnbounded(
                "maximum_bytes_billed が未設定のクエリは実行できません。"
                "Warehouse(max_bytes_billed=...) か query(max_bytes_billed=...) を指定してください。")
        # パーティション表への条件は allow_full_scan では免除しない。
        # BigQuery 側は require_partition_filter=true で、条件の無いクエリを
        # 実行そのものから拒否する（本番の /health が 400 で落ちた）。
        # ローカルだけ通る抜け道を残すと、その差が本番で初めて出る。
        assert_partition_filter(sql)

        df = self.con.execute(sql, params or []).df()
        scanned = int(df.memory_usage(deep=True).sum())
        if scanned > limit:
            raise BytesBilledUnbounded(
                f"走査量 {scanned:,} バイトが上限 {limit:,} を超えました。"
                "パーティション条件を絞ってください。")
        self.stats.append(QueryStats(sql, scanned, len(df)))
        return df

    def execute(self, sql: str, params: list | None = None) -> None:
        """DDL / DML。

        スキャン量は発生しないが、パーティション条件の検査はここでも行う。
        BigQuery は require_partition_filter で DML も拒否するので、
        ここを素通しにすると**ローカルでは全部通り、本番だけが落ちる**。
        実際に予測・ベット候補・確定層訂正の削除文がその状態だった。
        """
        if sql.strip().split(None, 1)[0].upper() != "CREATE":
            assert_partition_filter(sql)
        self.con.execute(sql, params or [])

    def insert_frame(self, table: str, df: pd.DataFrame) -> int:
        """列名を明示して差し込む。両バックエンドの共通入口。

        `register` + `INSERT ... SELECT *` は DuckDB 固有で、BigQuery 実装には
        register 自体が無い。書き込み経路がそこに直接依存していると、
        ローカルでは通るのに本番で AttributeError になる（実際になった）。
        """
        from .schema import table_columns

        cols = table_columns(table)
        payload = df.reindex(columns=cols)
        self.register("_ins_frame", payload)
        quoted = ", ".join(f'"{c}"' for c in cols)
        self.execute(
            f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM _ins_frame")
        return len(payload)

    def register(self, name: str, df: pd.DataFrame) -> None:
        self.con.register(name, df)

    def table(self, name: str) -> pd.DataFrame:
        return self.con.execute(f"SELECT * FROM {name}").df()

    def row_count(self, name: str) -> int:
        return int(self.con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])

    def total_bytes_scanned(self) -> int:
        return sum(s.bytes_scanned for s in self.stats)


def assert_partition_filter(sql: str) -> None:
    """パーティション列での絞り込みが無いクエリを弾く（DB-09）。

    BigQuery の `require_partition_filter=true` と同じ意味論をローカルで再現する。
    完全な SQL 解析はしない — 目的は「フルスキャンを書いてしまう事故」を
    レビュー前に落とすことなので、テーブル名と列名の共起で判定すれば足りる。
    """
    lowered = re.sub(r"\s+", " ", sql.lower())
    for table, column in PARTITIONED.items():
        if re.search(rf"\b(from|join)\s+{table}\b", lowered) and column not in lowered:
            raise PartitionFilterRequired(
                f"{table} を参照するクエリに {column} の条件がありません。"
                "パーティションフィルタなしのクエリはフルスキャンになります。")
