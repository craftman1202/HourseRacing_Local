"""デプロイ計画と実行。

**既定は plan（何も作らない）**。`--apply --confirm` を明示したときだけ実行する。
このプロジェクトは共用かつ課金有効なので、作成は必ず人の承認を挟む。

計画には「既存資源との衝突がないこと」の検査を含める。共用プロジェクトでは
名前の衝突がそのまま他システムのデータ破壊になる。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

INFRA = Path(__file__).resolve().parents[2] / "infra" / "services.json"
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "infra" / "bq"

# Cloud Scheduler の起動時刻（設計書 §3.1）。cron は JST で解釈させる
SCHEDULE_CRON = {
    "nar-ingest-and-refresh": "40 2 * * *",
    "nar-plan-day": "0 8 * * *",
    "nar-weekly-report": "0 4 * * 1",
}
ENDPOINT = {
    "nar-ingest-and-refresh": "/ingest-and-refresh",
    "nar-plan-day": "/plan-day",
    "nar-weekly-report": "/weekly-report",
}
# パーティション列（DB-09 の require_partition_filter 対象）
PARTITION_COLUMN = {
    "entry_result_final": "race_date", "entry_result_live": "race_date",
    "feature_snapshot": "as_of_date", "prediction": "race_date",
    "bet_candidate": "race_date", "odds_snapshot": "race_date",
}
# パーティションの粒度。BigQuery は1テーブル 10,000 パーティションが上限で、
# 1998年からの履歴を日次で切ると 10,210 個になり投入が拒否される
# （実際に 2025 年以降が入らなかった）。履歴を持つ表は月次にする。
# 当日ぶんしか持たない表は日次のままでよく、そのほうが刈り込みが効く。
PARTITION_TYPE = {
    "entry_result_final": "MONTH",   # 1998年からの全履歴
    "entry_result_live": "DAY",      # 当日のみ
    "feature_snapshot": "MONTH",     # 毎日積む。数年で日次上限に近づく
    "prediction": "MONTH",
    "bet_candidate": "MONTH",
    "odds_snapshot": "DAY",          # 当日のみ
}
LIFECYCLE = {
    # 原本 ZIP は不変層。30日 Nearline / 365日 Coldline（設計書 §6.2）
    "raw": [
        {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
         "condition": {"age": 30}},
        {"action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
         "condition": {"age": 365}},
    ],
    # リリースは10世代保持。古い版はロールバック先として残す必要がある
    "model": [
        {"action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
         "condition": {"age": 90}},
    ],
}


@dataclass
class Step:
    kind: str
    name: str
    command: list[str]
    exists: bool = False
    billable: bool = False
    note: str = ""
    stdin: str | None = None

    @property
    def action(self) -> str:
        return "skip（既存）" if self.exists else "create"


@dataclass
class Plan:
    project: str
    region: str
    steps: list[Step] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)

    @property
    def to_create(self) -> list[Step]:
        return [s for s in self.steps if not s.exists]

    @property
    def billable(self) -> list[Step]:
        return [s for s in self.to_create if s.billable]

    def render(self) -> str:
        lines = [f"プロジェクト: {self.project} / リージョン: {self.region}", ""]
        for s in self.steps:
            flag = "💰" if s.billable and not s.exists else "  "
            lines.append(f"{flag} [{s.action:>12}] {s.kind:<16} {s.name}")
            if s.note:
                lines.append(f"                     └ {s.note}")
        if self.collisions:
            lines += ["", "⚠ 衝突:"] + [f"  - {c}" for c in self.collisions]
        if self.deferred:
            lines += ["", "後回し（前提が揃っていない）:"] + [f"  - {d}" for d in self.deferred]
        lines += ["", f"新規作成 {len(self.to_create)} 件"
                      f"（うち課金対象 {len(self.billable)} 件）"]
        return "\n".join(lines)


def _run(cmd: list[str], timeout: int = 120, stdin: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           input=stdin)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def _cfg(infra_path: Path | None = None) -> dict:
    return json.loads((infra_path or INFRA).read_text(encoding="utf-8"))


def build_plan(infra_path: Path | None = None, service_url: str | None = None,
               image: str | None = None, mode: str = "shadow") -> Plan:
    cfg = _cfg(infra_path)
    project, region = cfg["project"], cfg["region"]
    plan = Plan(project, region)
    dataset = cfg["bigquery"]["dataset"]

    # ---------------------------------------------------------------- BigQuery
    rc, out = _run(["bq", f"--project_id={project}", "ls", "-d", "--format=json"])
    existing_ds = set()
    if rc == 0 and out.strip().startswith("["):
        existing_ds = {d.get("datasetReference", {}).get("datasetId")
                       for d in json.loads(out)}
    plan.steps.append(Step(
        "bq_dataset", dataset,
        ["bq", f"--project_id={project}", "mk", "--location", region,
         "--dataset", f"{project}:{dataset}"],
        exists=dataset in existing_ds,
        note="ストレージ 10GiB / クエリ 1TiB は無料枠。既存の hourse_racing とは別物"))

    existing_tables: set[str] = set()
    if dataset in existing_ds:
        rc, out = _run(["bq", f"--project_id={project}", "ls", "--format=json", dataset])
        if rc == 0 and out.strip().startswith("["):
            existing_tables = {t.get("tableReference", {}).get("tableId")
                               for t in json.loads(out)}
    for table, part_col in PARTITION_COLUMN.items():
        schema_file = SCHEMA_DIR / f"{table}.json"
        plan.steps.append(Step(
            "bq_table", f"{dataset}.{table}",
            ["bq", f"--project_id={project}", "mk", "--table",
             "--time_partitioning_field", part_col,
             "--time_partitioning_type", PARTITION_TYPE.get(table, "DAY"),
             "--require_partition_filter",
             f"{project}:{dataset}.{table}", str(schema_file)],
            exists=table in existing_tables,
            note=f"{part_col} で{PARTITION_TYPE.get(table, 'DAY')}"
                 "パーティション＋フィルタ必須（DB-09）"))
    for table in ("race_schedule", "notification_log", "pnl_daily", "job_run",
                  "skew_check", "entry_result_audit"):
        plan.steps.append(Step(
            "bq_table", f"{dataset}.{table}",
            ["bq", f"--project_id={project}", "mk", "--table",
             f"{project}:{dataset}.{table}", str(SCHEMA_DIR / f"{table}.json")],
            exists=table in existing_tables,
            note="小規模。パーティション不要"))

    # -------------------------------------------------------------------- GCS
    rc, out = _run(["gcloud", "storage", "buckets", "list", "--project", project,
                    "--format=value(name)"])
    existing_buckets = set(out.split()) if rc == 0 else set()
    for key, bucket in cfg["gcs"].items():
        plan.steps.append(Step(
            "gcs_bucket", bucket,
            ["gcloud", "storage", "buckets", "create", f"gs://{bucket}",
             "--project", project, "--location", region,
             "--uniform-bucket-level-access", "--public-access-prevention"],
            exists=bucket in existing_buckets, billable=True,
            note="原本ZIP 20〜35GB想定で月$0.4〜0.8" if key == "raw"
                 else "リリース10世代で月$0.02〜0.06"))
        plan.steps.append(Step(
            "gcs_lifecycle", f"{bucket}:lifecycle",
            ["gcloud", "storage", "buckets", "update", f"gs://{bucket}",
             "--lifecycle-file", "-"],
            exists=False,
            stdin=json.dumps({"lifecycle": {"rule": LIFECYCLE[key]}}),
            note="30日 Nearline / 365日 Coldline（CO-03）" if key == "raw"
                 else "90日 Nearline"))

    # ------------------------------------------------------------- IAM
    for sa_email, roles in cfg["service_accounts"].items():
        sa_id = sa_email.split("@")[0]
        rc, _ = _run(["gcloud", "iam", "service-accounts", "describe", sa_email,
                      "--project", project])
        plan.steps.append(Step(
            "service_account", sa_email,
            ["gcloud", "iam", "service-accounts", "create", sa_id,
             "--project", project, "--display-name", f"nar-ops {sa_id}"],
            exists=(rc == 0)))
        for role in roles:
            plan.steps.append(Step(
                "iam_binding", f"{sa_id} → {role}",
                ["gcloud", "projects", "add-iam-policy-binding", project,
                 "--member", f"serviceAccount:{sa_email}", "--role", role,
                 "--condition", "None", "--quiet"],
                exists=False, note="冪等（既存なら no-op）"))

    # ------------------------------------------------------------ Cloud Tasks
    queue = "nar-queue"
    rc, _ = _run(["gcloud", "tasks", "queues", "describe", queue,
                  "--project", project, "--location", region])
    plan.steps.append(Step(
        "tasks_queue", queue,
        ["gcloud", "tasks", "queues", "create", queue, "--project", project,
         "--location", region,
         "--max-attempts", "3", "--min-backoff", "30s", "--max-backoff", "270s"],
        exists=(rc == 0), note="月100万オペレーション無料。リトライ 3回/30-270秒（IN-11）"))

    # -------------------------------------------------------------- Cloud Run
    rc, out = _run(["gcloud", "run", "services", "list", "--project", project,
                    "--region", region, "--format=value(metadata.name)"])
    existing_services = set(out.split()) if rc == 0 else set()
    for name, spec in cfg.get("cloud_run", {}).items():
        if image is None:
            plan.deferred.append(
                f"cloud run {name}: イメージ未指定のため保留。"
                "Artifact Registry へ push してから --image を渡して再実行")
            continue
        command = [
            "gcloud", "run", "deploy", name,
            "--project", project, "--region", spec.get("region", region),
            "--image", image,
            "--service-account", spec["service_account"],
            "--min-instances", str(spec["min_instance_count"]),
            "--max-instances", str(spec["max_instance_count"]),
            "--memory", spec["memory"], "--cpu", str(spec["cpu"]),
            "--no-allow-unauthenticated" if not spec.get("allow_unauthenticated")
            else "--allow-unauthenticated",
            "--set-env-vars", ",".join([
                f"NAROPS_PROJECT={project}",
                f"NAROPS_DATASET={cfg['bigquery']['dataset']}",
                # モデルを読むのは推論を持つ nar-ops だけ。API と Web は
                # BigQuery の結果を出すだけで、モデルを読む権限も持たない
                # （持たせると最小権限でなくなる）。バケットを渡すと起動時に
                # 取りに行って 403 で落ちる。
                *([f"NAROPS_MODEL_BUCKET={cfg['gcs']['model']}"]
                  if name == "nar-ops" else []),
                f"NAROPS_RAW_BUCKET={cfg['gcs']['raw']}",
                # 役割。通知とモデル読み込みを持つのは ops だけ
                f"NAROPS_ROLE={name.replace('nar-', '')}",
                # 推論タスクの予約先。渡さないとインメモリのままになり、
                # 予約がインスタンス終了で消える
                *([f"NAROPS_SERVICE_URL={service_url}"]
                  if name == "nar-ops" and service_url else []),
                # 実データで学習したモデルであることの印。合成データのままでは
                # live に上げられない（mode.validate）
                "MODEL_PROVENANCE=real",
                # モードは deploy の引数で決める。既定は shadow のまま。
                f"NAROPS_MODE={mode}",
            ]),
            "--quiet",
        ]
        if spec.get("cpu_idle", True):
            command.append("--cpu-throttling")
        plan.steps.append(Step(
            # Cloud Run は「作成」ではなく「デプロイ」。既存でも設定や
            # イメージを変えるために再実行する必要がある。exists で飛ばすと
            # モード変更が反映されない。
            "cloud_run", name, command,
            exists=False, billable=True,
            note=f"min-instances={spec['min_instance_count']} なので待機課金なし。"
                 "起動は shadow 固定"))

    # -------------------------------------------------------------- Scheduler
    rc, out = _run(["gcloud", "scheduler", "jobs", "list", "--project", project,
                    "--location", region, "--format=value(name)"])
    existing_jobs = {j.split("/")[-1] for j in out.split()} if rc == 0 else set()
    n_total = len(existing_jobs)
    ops_sa = next(s for s in cfg["service_accounts"] if s.startswith("nar-ops@"))
    for job in cfg["scheduler_jobs"]:
        if service_url is None:
            plan.deferred.append(
                f"scheduler {job}: 対象の Cloud Run URL が未確定のため保留。"
                "nar-ops をデプロイしてから --service-url を渡して再実行")
            continue
        plan.steps.append(Step(
            "scheduler", job,
            ["gcloud", "scheduler", "jobs", "create", "http", job,
             "--project", project, "--location", region,
             "--schedule", SCHEDULE_CRON[job], "--time-zone", "Asia/Tokyo",
             "--uri", f"{service_url}{ENDPOINT[job]}", "--http-method", "POST",
             "--oidc-service-account-email", ops_sa, "--attempt-deadline", "540s"],
            exists=job in existing_jobs, billable=True,
            note=f"既存ジョブ {n_total} 個。無料枠は請求先アカウント単位で3個なので "
                 "追加分は $0.10/月/ジョブ"))

    plan.collisions = _detect_collisions(cfg)
    return plan


def _detect_collisions(cfg: dict) -> list[str]:
    """他システムの資源名と被っていないか（共用プロジェクトの必須チェック）。"""
    from .gcp import FOREIGN_RESOURCES

    out = []
    if cfg["bigquery"]["dataset"] in FOREIGN_RESOURCES:
        out.append(f"BigQuery データセット {cfg['bigquery']['dataset']} は他システムの資産")
    for bucket in cfg["gcs"].values():
        if bucket in FOREIGN_RESOURCES:
            out.append(f"バケット {bucket} は他システムの資産")
    return out


def apply_plan(plan: Plan, confirm: bool = False) -> list[tuple[str, bool, str]]:
    """計画の実行。confirm=False なら何もしない。"""
    if not confirm:
        raise PermissionError(
            "apply には明示的な承認が必要です。課金の発生する資源を作成します。")
    if plan.collisions:
        raise RuntimeError(f"衝突があるため実行しません: {plan.collisions}")

    results = []
    for step in plan.to_create:
        command = step.command
        tmp: Path | None = None
        if step.stdin is not None and "-" in command:
            # `gcloud storage buckets update --lifecycle-file -` は標準入力を
            # 受け付けず、`-` をファイル名として開こうとして失敗する。
            # 一時ファイルに落としてから渡す。
            fd, name = tempfile.mkstemp(suffix=".json")
            tmp = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(step.stdin)
            command = [str(tmp) if a == "-" else a for a in command]
        try:
            rc, out = _run(command, timeout=300,
                           stdin=None if tmp else step.stdin)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        results.append((f"{step.kind} {step.name}", rc == 0, out.strip()[:400]))
    return results
