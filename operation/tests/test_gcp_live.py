"""実 GCP（sample-335613）の状態検証。

既定では走らない。実行するには明示的にオプトインする:

    pytest tests/test_gcp_live.py -m costly --run-live

課金は発生しない（すべて読み取り、BQ はドライラン）が、実プロジェクトの状態に
依存するので CI の既定には入れない。テスト仕様 §1.3 の `costly` の扱いに従う。

このプロジェクトは共用なので、**他システムの資源に触っていないこと**の検証も含める。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.costly]

PROJECT = "sample-335613"
REGION = "asia-northeast1"
INFRA = Path(__file__).resolve().parents[1] / "infra" / "services.json"


def _run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


@pytest.fixture(scope="module")
def cfg() -> dict:
    return json.loads(INFRA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables() -> list[dict]:
    rc, out = _run(["bq", f"--project_id={PROJECT}", "ls", "--format=json", "nar_ops"])
    if rc != 0:
        pytest.skip(f"BigQuery に到達できません: {out[:200]}")
    return json.loads(out)


# ------------------------------------------------------------------ DB-09
def test_all_large_tables_require_partition_filter(tables):
    """パーティションフィルタ必須が実際に設定されていること。

    ここが外れていると、1本の事故クエリで無料枠 1TiB を焼く。
    """
    expected = {"entry_result_final": "race_date", "entry_result_live": "race_date",
                "feature_snapshot": "as_of_date", "prediction": "race_date",
                "bet_candidate": "race_date", "odds_snapshot": "race_date"}
    by_id = {t["tableReference"]["tableId"]: t for t in tables}
    for name, col in expected.items():
        assert name in by_id, f"{name} が存在しません"
        rc, out = _run(["bq", f"--project_id={PROJECT}", "show", "--format=json",
                        f"{PROJECT}:nar_ops.{name}"])
        meta = json.loads(out)
        part = meta.get("timePartitioning", {})
        assert part.get("field") == col, f"{name} のパーティション列が {part.get('field')}"
        assert part.get("requirePartitionFilter") is True, \
            f"{name} で require_partition_filter が無効です"


def test_unfiltered_query_is_rejected_by_bigquery():
    """フィルタ無しクエリが実際に拒否されること（ドライラン、課金なし）。"""
    rc, out = _run(["bq", f"--project_id={PROJECT}", "query", "--use_legacy_sql=false",
                    "--dry_run",
                    f"SELECT COUNT(*) FROM `{PROJECT}.nar_ops.entry_result_final`"])
    assert rc != 0, "フィルタ無しクエリが通ってしまいました"
    assert "partition elimination" in out or "filter over column" in out


def test_filtered_query_is_accepted():
    rc, out = _run(["bq", f"--project_id={PROJECT}", "query", "--use_legacy_sql=false",
                    "--dry_run",
                    f"SELECT COUNT(*) FROM `{PROJECT}.nar_ops.entry_result_final` "
                    "WHERE race_date = '2026-08-25'"])
    assert rc == 0, out[:300]


# ------------------------------------------------------------------ スキーマ整合
def test_bq_schema_matches_local_ddl(tables):
    """BigQuery の列構成がローカル DDL と一致すること。

    ローカル（DuckDB）で検証したクエリが本番で落ちる、を防ぐ。
    """
    by_id = {t["tableReference"]["tableId"] for t in tables}
    for schema_file in (INFRA.parent / "bq").glob("*.json"):
        name = schema_file.stem
        assert name in by_id, f"{name} が BigQuery にありません"
        expected = {f["name"] for f in json.loads(schema_file.read_text())}
        rc, out = _run(["bq", f"--project_id={PROJECT}", "show", "--format=json",
                        f"{PROJECT}:nar_ops.{name}"])
        actual = {f["name"] for f in json.loads(out)["schema"]["fields"]}
        assert actual == expected, f"{name} の列が乖離: {actual ^ expected}"


# ------------------------------------------------------------------ CO-03
def test_buckets_have_lifecycle_rules(cfg):
    for key, bucket in cfg["gcs"].items():
        rc, out = _run(["gcloud", "storage", "buckets", "describe", f"gs://{bucket}",
                        "--format=json"])
        assert rc == 0, f"{bucket} が見つかりません"
        meta = json.loads(out)
        rules = (meta.get("lifecycle_config") or {}).get("rule") or []
        assert rules, f"{bucket} にライフサイクルが設定されていません（CO-03）"
        if key == "raw":
            classes = {r["action"].get("storageClass") for r in rules}
            assert {"NEARLINE", "COLDLINE"} <= classes


def test_buckets_are_private(cfg):
    """公開アクセス防止と均一バケットレベルアクセス。"""
    for bucket in cfg["gcs"].values():
        rc, out = _run(["gcloud", "storage", "buckets", "describe", f"gs://{bucket}",
                        "--format=json"])
        meta = json.loads(out)
        assert meta.get("public_access_prevention") == "enforced", f"{bucket} が公開可能"
        # gcloud のバージョンで bool を返す場合と {"enabled": bool} を返す場合がある
        ubla = meta.get("uniform_bucket_level_access")
        enabled = ubla.get("enabled") if isinstance(ubla, dict) else ubla
        assert enabled is True, f"{bucket} で UBLA が無効"


# ------------------------------------------------------------------ RL-06
def test_service_accounts_exist_with_designed_roles(cfg):
    rc, out = _run(["gcloud", "projects", "get-iam-policy", PROJECT, "--format=json"])
    assert rc == 0
    policy = json.loads(out)
    granted: dict[str, set[str]] = {}
    for binding in policy["bindings"]:
        for member in binding.get("members", []):
            if member.startswith("serviceAccount:nar-"):
                granted.setdefault(member.split(":", 1)[1], set()).add(binding["role"])

    for sa, roles in cfg["service_accounts"].items():
        assert sa in granted, f"{sa} に権限が付与されていません"
        assert granted[sa] == set(roles), (
            f"{sa} の権限が設計と不一致（余剰 {granted[sa] - set(roles)} / "
            f"不足 {set(roles) - granted[sa]}）")


# ------------------------------------------------------------------ 共用プロジェクト
def test_we_did_not_touch_foreign_resources():
    """他システムの資源が無事であること。

    共用プロジェクトでは、自分の作成物より「壊していないこと」の確認が重要。
    """
    rc, out = _run(["bq", f"--project_id={PROJECT}", "ls", "--format=json"])
    datasets = {d["datasetReference"]["datasetId"] for d in json.loads(out)}
    for foreign in ("hourse_racing", "chx_mart", "chx_ops", "chx_raw",
                    "database_stock", "finml_strategy"):
        assert foreign in datasets, f"{foreign} が消えています"

    rc, out = _run(["gcloud", "storage", "buckets", "list", "--project", PROJECT,
                    "--format=value(name)"])
    buckets = set(out.split())
    assert "hourse-racing-inference-sample-335613" in buckets
    assert "sample-335613-jra-artifacts" in buckets


def test_our_resources_are_namespaced(cfg):
    rc, out = _run(["bq", f"--project_id={PROJECT}", "ls", "--format=json"])
    datasets = {d["datasetReference"]["datasetId"] for d in json.loads(out)}
    assert "nar_ops" in datasets
    assert cfg["bigquery"]["dataset"] == "nar_ops"


# ------------------------------------------------------------------ Cloud Tasks
def test_task_queue_retry_matches_design():
    """IN-11: 最大3回・30〜270秒のバックオフ。"""
    rc, out = _run(["gcloud", "tasks", "queues", "describe", "nar-queue",
                    "--project", PROJECT, "--location", REGION, "--format=json"])
    assert rc == 0
    q = json.loads(out)
    retry = q.get("retryConfig", {})
    assert int(retry.get("maxAttempts", 0)) == 3
    assert retry.get("minBackoff") == "30s" and retry.get("maxBackoff") == "270s"


# ------------------------------------------------------------------ DC-02
def test_discord_secrets_exist_in_secret_manager():
    """Secret Manager から解決できること。値は取得しない（DC-01）。"""
    rc, out = _run(["gcloud", "secrets", "list", "--project", PROJECT,
                    "--format=value(name)"])
    assert rc == 0
    names = set(out.split())
    for key in ("DISCORD_WEBHOOK_PREDICTION", "DISCORD_WEBHOOK_ALERT",
                "DISCORD_WEBHOOK_DAILY"):
        assert key in names, f"{key} が Secret Manager にありません"
