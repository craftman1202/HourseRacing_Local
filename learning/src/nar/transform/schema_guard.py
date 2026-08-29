"""スキーマガード。

NAR は仕様を予告なく変更すると明記している。壊れたデータを黙って取り込むのが
最悪の失敗なので、ここは fail-fast にする（SG-01..04）。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from ..errors import SchemaDriftError

# 公式説明書からピン留めした期待列数（SG-01）
EXPECTED_COLUMNS = {
    "race": 66,      # レース一覧
    "entry": 36,     # 出馬表
    "odds": 10,
    "payout": 54,
}


@dataclass(frozen=True)
class CsvSchema:
    kind: str
    n_columns: int
    first_row: tuple[str, ...]

    def hash(self) -> str:
        # 列名集合ではなく「順序を含む系列」をハッシュする。列順の入れ替えを
        # 通してしまうと位置ベースのパーサが静かに壊れる（SG-03）。
        payload = json.dumps(
            {"kind": self.kind, "n": self.n_columns, "cols": list(self.first_row)},
            ensure_ascii=False, sort_keys=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_first_row(text: str) -> tuple[str, ...]:
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if row:
            return tuple(c.strip() for c in row)
    return ()


def infer_kind(n_columns: int) -> str | None:
    for kind, n in EXPECTED_COLUMNS.items():
        if n == n_columns:
            return kind
    return None


def schema_of(text: str, kind: str | None = None) -> CsvSchema:
    first = read_first_row(text)
    n = len(first)
    return CsvSchema(kind or infer_kind(n) or f"unknown_{n}", n, first)


def combined_hash(schemas: list[CsvSchema]) -> str:
    h = hashlib.sha256()
    for s in sorted(schemas, key=lambda x: x.kind):
        h.update(s.hash().encode())
    return h.hexdigest()


class SchemaRegistry:
    """meta/schema_hash.json の既知値。未知は「新規登録」ではなく drift として扱う。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.known: dict[str, str] = (
            json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        )

    def pin(self, kind: str, schema_hash: str) -> None:
        self.known[kind] = schema_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.known, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def check(self, schema: CsvSchema) -> None:
        expected_n = EXPECTED_COLUMNS.get(schema.kind)
        if expected_n is not None and schema.n_columns != expected_n:
            raise SchemaDriftError(
                f"{schema.kind}: 列数 {schema.n_columns}（期待 {expected_n}）。"
                "silver 昇格を停止します。"
            )
        if schema.kind not in self.known:
            raise SchemaDriftError(
                f"{schema.kind}: 既知の schema_hash がありません。"
                "新しい列構成を確認のうえ registry.pin() で明示的に登録してください。"
            )
        actual = schema.hash()
        if self.known[schema.kind] != actual:
            raise SchemaDriftError(
                f"{schema.kind}: schema_hash 不一致 "
                f"(known={self.known[schema.kind][:12]}… actual={actual[:12]}…)。"
                "列の追加・削除・並べ替えのいずれかが起きています。"
            )
