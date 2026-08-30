"""実 GCP バックエンド。

ローカル代替（DuckDB / ローカル FS / インプロセスキュー）と**同じインタフェース**を
実装する。差し替えは1行で済み、制約チェック（パーティション必須・bytes_billed）は
共通のまま残る。

このプロジェクト（sample-335613）は共用で、既存ワークロード（chx-* / jra-* /
hourse-racing-webui / finml-*）が動いている。本システムは `nar_ops` / `nar-*` の
名前空間だけを触り、それ以外には**読み書きしない**。
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .db.backend import QueryStats, assert_partition_filter
from .errors import ArtifactIntegrityError, BytesBilledUnbounded

# 本システムが触ってよい資源の接頭辞。ここを外れる操作は事故とみなす
OWNED_DATASET = "nar_ops"
OWNED_PREFIXES = ("nar-", "nar_")
# 同一プロジェクトに実在する別システムの資産。絶対に触らない
FOREIGN_RESOURCES = frozenset({
    "hourse_racing", "chx_mart", "chx_ops", "chx_ops_dev", "chx_raw",
    "database_stock", "finml_strategy", "Dataset_TradeAIApp",
    "hourse-racing-inference-sample-335613", "sample-335613-jra-artifacts",
})


log = logging.getLogger(__name__)


class ForeignResourceAccess(Exception):
    """他システムの資源に触ろうとした。"""


def assert_owned(name: str) -> None:
    """操作対象が本システムの名前空間内か。

    共用プロジェクトでは、テーブル名のタイプミスが他システムのデータ破壊になる。
    書き込み経路の入口で必ず通す。
    """
    base = name.split(".")[-2] if "." in name else name
    if base in FOREIGN_RESOURCES or name in FOREIGN_RESOURCES:
        raise ForeignResourceAccess(
            f"{name} は別システムの資産です。本システムからは触りません。")
    if not (base.startswith(OWNED_PREFIXES) or base == OWNED_DATASET):
        raise ForeignResourceAccess(
            f"{name} が本システムの名前空間（{OWNED_DATASET} / nar-*）の外です。")


@dataclass
class BigQueryWarehouse:
    """`db.backend.Warehouse` の BigQuery 実装。

    `maximum_bytes_billed` を必ず設定し、パーティションフィルタを強制する点は
    ローカル版と同じ。ここを緩めると1本の事故クエリで無料枠を焼く。
    """

    project: str
    dataset: str = OWNED_DATASET
    location: str = "asia-northeast1"
    max_bytes_billed: int | None = None
    stats: list[QueryStats] = field(default_factory=list)
    _client: Any = None

    def __post_init__(self) -> None:
        assert_owned(self.dataset)

    @property
    def client(self):
        if self._client is None:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project, location=self.location)
        return self._client

    def query(self, sql: str, params: list | None = None,
              max_bytes_billed: int | None = None,
              allow_full_scan: bool = False) -> pd.DataFrame:
        from google.cloud import bigquery

        limit = max_bytes_billed if max_bytes_billed is not None else self.max_bytes_billed
        if limit is None:
            raise BytesBilledUnbounded(
                "maximum_bytes_billed が未設定のクエリは実行できません（CO-01）")
        # パーティション条件は allow_full_scan では免除しない。BigQuery 側が
        # require_partition_filter で実行そのものを拒否するので、免除しても
        # 通らない（DuckDB 実装と同じ扱いに揃える）。
        assert_partition_filter(sql)

        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=limit,
            query_parameters=[
                bigquery.ScalarQueryParameter(None, _bq_type(v), v) for v in (params or [])
            ],
            default_dataset=f"{self.project}.{self.dataset}",
        )
        job = self.client.query(sql, job_config=job_config)
        df = job.to_dataframe()
        self.stats.append(QueryStats(sql, int(job.total_bytes_processed or 0), len(df)))
        return df

    def close(self) -> None:
        """`Warehouse.close()` とインタフェースを揃える（CLI が無条件で呼ぶ）。"""
        if self._client is not None:
            self._client.close()

    def dry_run_bytes(self, sql: str) -> int:
        """走査量の事前見積り。CO-04 の実測に使う。課金は発生しない。"""
        from google.cloud import bigquery

        job = self.client.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False,
            default_dataset=f"{self.project}.{self.dataset}"))
        return int(job.total_bytes_processed or 0)

    def load_table(self, table: str, df: pd.DataFrame, write_disposition: str = "WRITE_APPEND") -> int:
        from google.cloud import bigquery

        full = f"{self.project}.{self.dataset}.{table}"
        assert_owned(full)
        job = self.client.load_table_from_dataframe(
            df, full, job_config=bigquery.LoadJobConfig(
                write_disposition=write_disposition))
        job.result()
        return len(df)

    def execute(self, sql: str, params: list | None = None) -> None:
        """DDL / DML。DuckDB 実装と同じ入口。

        書き込み経路の冪等化に使う削除文が通らないと、再実行のたびに行が
        二重に積まれる。`?` プレースホルダは BigQuery の名前付きパラメータに
        直して渡す。

        CREATE TABLE / CREATE VIEW は受け付けない。BigQuery のテーブルは
        パーティションとフィルタ必須の設定を伴うので `narops deploy` で作る。
        アプリから作れてしまうと、設定の無いテーブルが本番にできる。
        """
        from google.cloud import bigquery

        head = sql.strip().split(None, 1)[0].upper()
        if head == "CREATE":
            log.debug("BigQuery ではスキーマ作成を行いません: %s", sql[:60])
            return

        # query と同じ検査を DML にも掛ける。ここが素通しだったため、
        # 予測・ベット候補・確定層訂正の削除文がそろって本番で拒否され、
        # ローカルのテストでは全部通っていた。
        assert_partition_filter(sql)

        values, converted = [], sql
        for i, value in enumerate(params or []):
            name = f"p{i}"
            converted = converted.replace("?", f"@{name}", 1)
            values.append(bigquery.ScalarQueryParameter(
                name, _bq_type(value), value))
        job = self.client.query(
            converted,
            job_config=bigquery.QueryJobConfig(
                query_parameters=values,
                maximum_bytes_billed=self.max_bytes_billed,
                # 既定データセットを渡さないと、修飾なしのテーブル名が
                # 「dataset で修飾しろ」と 400 になる（query 側にはあった）
                default_dataset=f"{self.project}.{self.dataset}"))
        job.result()

    def insert_frame(self, table: str, df: pd.DataFrame) -> int:
        """DuckDB 実装と同じ入口。列を DDL に合わせてから追記する。"""
        from .db.schema import table_columns

        cols = table_columns(table)
        return self.load_table(table, df.reindex(columns=cols),
                               write_disposition="WRITE_APPEND")

    def total_bytes_scanned(self) -> int:
        return sum(s.bytes_scanned for s in self.stats)


def _bq_type(v: Any) -> str:
    import datetime as _dt

    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int):
        return "INT64"
    if isinstance(v, float):
        return "FLOAT64"
    if isinstance(v, _dt.datetime):
        return "TIMESTAMP"
    if isinstance(v, _dt.date):
        return "DATE"
    return "STRING"


@dataclass
class GcsModelRegistry:
    """`model.registry.ModelRegistry` の GCS 実装。

    ディレクトリ構造はローカル版と同一（`releases/<id>/` と `current.json`）なので、
    ローカルで検証した MP-02/05/08 の挙動がそのまま移る。
    """

    bucket: str
    project: str
    # 競技ごとの current ポインタ。ローカル版 ModelRegistry と同じ規約
    # （flat=current.json / banei=current_banei.json）。
    family: str = "flat"
    _client: Any = None

    def __post_init__(self) -> None:
        assert_owned(self.bucket)

    @property
    def current_file(self) -> str:
        from .model.registry import current_pointer

        return current_pointer(self.family)

    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage

            self._client = storage.Client(project=self.project)
        return self._client

    def _bucket(self):
        return self.client.bucket(self.bucket)

    def releases(self) -> list[str]:
        prefix = "releases/"
        names = set()
        for blob in self.client.list_blobs(self.bucket, prefix=prefix):
            rest = blob.name[len(prefix):]
            if "/" in rest:
                names.add(rest.split("/", 1)[0])
        return sorted(names)

    def current_id(self) -> str | None:
        import json

        blob = self._bucket().blob(self.current_file)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text()).get("release_id")

    def set_current(self, release_id: str, actor: str = "system", reason: str = "") -> None:
        """GCS オブジェクトの上書きは原子的なので、書き換え中の読み取りは旧版を返す。"""
        import json
        from datetime import datetime

        if release_id not in self.releases():
            raise FileNotFoundError(f"{release_id} が releases 配下にありません")
        # ローカル版 ModelRegistry.set_current と同じ検査を必ず入れる。
        # 片側だけ緩いと「ローカルのテストは緑、本番だけ取り違えを受け付ける」
        # という一番悪い形になる（releases/ は系統をまたいで共有）。
        from .model.registry import family_of

        actual = family_of(self.load_manifest(release_id))
        if actual and actual != self.family:
            raise ArtifactIntegrityError(
                f"{release_id} は {actual} のモデルです。{self.family} の current"
                "には置けません。系統を取り違えると、別競技のレースを別競技の"
                "モデルで採点します。")
        self._bucket().blob(self.current_file).upload_from_string(
            json.dumps({"release_id": release_id,
                        "switched_at": datetime.now().isoformat(timespec="seconds")}),
            content_type="application/json")
        audit = self._bucket().blob("promotions.jsonl")
        prev = audit.download_as_text() if audit.exists() else ""
        audit.upload_from_string(prev + json.dumps({
            "at": datetime.now().isoformat(timespec="seconds"),
            "family": self.family,
            "actor": actor, "to": release_id, "reason": reason}, ensure_ascii=False) + "\n")

    def download_release(self, release_id: str, dest: str) -> str:
        from pathlib import Path

        out = Path(dest) / release_id
        out.mkdir(parents=True, exist_ok=True)
        for blob in self.client.list_blobs(self.bucket, prefix=f"releases/{release_id}/"):
            target = out / blob.name.split(f"releases/{release_id}/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(target)
        return str(out)

    def load_manifest(self, release_id: str):
        """manifest.json だけを読む（アーティファクト一式は落とさない）。

        `/models` 画面の一覧表示のように、リリースを検証せず要約だけ見たい
        読み取り専用ユースケース向け。`download_release` はモデル本体まで
        含めて全部落とすので一覧表示には重すぎる。
        """
        import json

        from .model.manifest import Manifest

        blob = self._bucket().blob(f"releases/{release_id}/manifest.json")
        if not blob.exists():
            raise FileNotFoundError(f"{release_id}/manifest.json が見つかりません")
        return Manifest.from_dict(json.loads(blob.download_as_text()))

    def audit_log(self) -> list[dict]:
        import json

        blob = self._bucket().blob("promotions.jsonl")
        if not blob.exists():
            return []
        text = blob.download_as_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class CloudTasksQueue:
    """`tasks.TaskQueue` の Cloud Tasks 実装。

    タスク名の一意制約はサービス側が持っているので、ALREADY_EXISTS を
    「重複なので何もしない」に翻訳するだけでよい（SC-03）。
    """

    project: str
    location: str
    queue: str
    service_url: str
    service_account: str
    _client: Any = None

    def __post_init__(self) -> None:
        assert_owned(self.queue)

    @property
    def client(self):
        if self._client is None:
            from google.cloud import tasks_v2

            self._client = tasks_v2.CloudTasksClient()
        return self._client

    def count(self, endpoint: str | None = None) -> int:
        """キューに残っているタスク数。

        インメモリ実装と同じ入口を持たせる。持たせないと、
        「何件積んだか」を返すだけの箇所で本番だけ AttributeError になる。
        Cloud Tasks は endpoint での絞り込みを持たないので、名前の接頭辞で見る。
        """
        parent = self.client.queue_path(self.project, self.location, self.queue)
        prefix = f"{parent}/tasks/{endpoint.strip('/')}" if endpoint else None
        n = 0
        for t in self.client.list_tasks(request={"parent": parent}):
            if prefix is None or t.name.startswith(prefix):
                n += 1
        return n

    def enqueue(self, task) -> bool:
        from google.api_core import exceptions
        from google.cloud import tasks_v2
        from google.protobuf import timestamp_pb2

        parent = self.client.queue_path(self.project, self.location, self.queue)
        ts = timestamp_pb2.Timestamp()
        ts.FromDatetime(task.scheduled_for)
        import json

        req = {
            "name": f"{parent}/tasks/{task.name}",
            "schedule_time": ts,
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{self.service_url}{task.endpoint}",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(task.payload).encode(),
                "oidc_token": {"service_account_email": self.service_account},
            },
        }
        try:
            self.client.create_task(parent=parent, task=req)
            return True
        except exceptions.AlreadyExists:
            return False        # 同名タスクは既にある = 冪等に成功扱い

    def delete(self, name: str) -> bool:
        from google.api_core import exceptions

        path = self.client.task_path(self.project, self.location, self.queue, name)
        try:
            self.client.delete_task(name=path)
            return True
        except exceptions.NotFound:
            return False


@dataclass
class SecretManagerResolver:
    """Secret Manager から webhook を読む。起動時に1回だけ読む想定。"""

    project: str
    _client: Any = None

    @property
    def client(self):
        if self._client is None:
            from google.cloud import secretmanager

            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    def access(self, name: str) -> str:
        return self.client.access_secret_version(
            request={"name": name}).payload.data.decode("utf-8")
