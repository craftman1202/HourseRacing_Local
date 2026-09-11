"""コスト見積とガード。

「実測してから気付く」では手遅れになるので、見積をテストとして固定する（CO-04）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Cloud Run 無料枠（月）
FREE_VCPU_SEC = 240_000
FREE_GIB_SEC = 450_000
VCPU_SEC_PRICE = 0.000024
GIB_SEC_PRICE = 0.0000025
TARGET_MONTHLY_USD = 1.1          # CO-04


@dataclass
class ServiceUsage:
    name: str
    invocations_per_day: float
    avg_seconds: float
    vcpu: float
    memory_gib: float

    def monthly_vcpu_sec(self) -> float:
        return self.invocations_per_day * 30 * self.avg_seconds * self.vcpu

    def monthly_gib_sec(self) -> float:
        return self.invocations_per_day * 30 * self.avg_seconds * self.memory_gib


@dataclass
class CostEstimate:
    services: list[ServiceUsage]
    storage_usd: float = 0.5
    breakdown: dict[str, float] = field(default_factory=dict)

    def compute(self) -> float:
        vcpu = sum(s.monthly_vcpu_sec() for s in self.services)
        gib = sum(s.monthly_gib_sec() for s in self.services)
        vcpu_cost = max(0.0, vcpu - FREE_VCPU_SEC) * VCPU_SEC_PRICE
        gib_cost = max(0.0, gib - FREE_GIB_SEC) * GIB_SEC_PRICE
        self.breakdown = {
            "vcpu_seconds": vcpu, "gib_seconds": gib,
            "vcpu_usd": round(vcpu_cost, 4), "gib_usd": round(gib_cost, 4),
            "storage_usd": self.storage_usd,
        }
        return round(vcpu_cost + gib_cost + self.storage_usd, 4)

    def within_target(self, target: float = TARGET_MONTHLY_USD) -> bool:
        return self.compute() <= target


def default_usage(odds_snapshots_per_day: int = 110,
                  refresh_avg_seconds: float = 700) -> list[ServiceUsage]:
    """設計書 §6.1 の想定。

    オッズ収集を締切直前帯に絞る調整（起動回数 500 → 約 300/日）を採用した後の値。
    この調整なしでは Cloud Run の無料枠を超えて月 $1.4 が発生する。

    `nar-refresh`（Cloud Run Job、2026-08-29 追加、16Gi・4vCPU、日次1回）は
    2026-08-28 時点のこの見積もりに一度も入っていなかった。bronze の確定月
    キャッシュ（`refresh.py` 参照、2026-09-10）を入れる前は毎日全履歴を ZIP から
    再展開しており、実測で1回あたり13分36秒・14分52秒かかっていた
    （キャッシュが機能する前提の `is_final` が当時は1件も立っていなかった
    ため、無キャッシュと同じ挙動だった）。

    `refresh_avg_seconds` の既定値 700 秒は、確定済み月の再展開キャッシュに加え、
    過去分（1998年〜、約340か月）を一度きりの確定化スイープ
    （`operation/scripts/finalize_history_backlog.py`）で `is_final=True` に
    した**後**の実測値（2026-09-10、本番で11分32秒＝692秒を計測、キャッシュ
    導入前比で約20%減）。残る所要時間の大半は bronze キャッシュの GCS 往復
    （`sync_dir` が月×表ごとの小さな parquet を個別コピーしている）と、
    speed_index の全履歴再計算（`build_silver_frames`、これは仕様上削れない）。
    さらに縮めるには GCS 側の一括コピーへの置き換えが要るが、本セッションの
    範囲外とした（申し送り、`docs/MaintenanceRunbook.md` §7）。
    """
    return [
        ServiceUsage("nar-ops-infer", 70, 20, 1.0, 1.0),
        ServiceUsage("nar-ops-odds", odds_snapshots_per_day, 8, 1.0, 0.5),
        ServiceUsage("nar-ops-batch", 6, 60, 1.0, 1.0),
        ServiceUsage("nar-refresh", 1, refresh_avg_seconds, 4.0, 16.0),
        ServiceUsage("nar-api", 100, 0.3, 0.5, 0.5),
        ServiceUsage("nar-web", 67, 0.5, 1.0, 0.5),
    ]


# ------------------------------------------------------------------ CO-01/02
def assert_billing_guard(job_config: dict) -> None:
    """全 BQ ジョブに maximum_bytes_billed が設定されていること。"""
    if not job_config.get("maximum_bytes_billed"):
        raise ValueError(
            "maximum_bytes_billed が未設定の BigQuery ジョブは実行できません（CO-01）")


def assert_no_always_on(terraform: dict) -> list[str]:
    """Terraform 上で min-instances=0、リージョンが asia-northeast1（CO-02）。"""
    problems = []
    for name, svc in terraform.get("cloud_run", {}).items():
        if svc.get("min_instance_count", 0) != 0:
            problems.append(f"{name}: min_instance_count={svc.get('min_instance_count')}")
        if svc.get("cpu_idle") is False:
            problems.append(f"{name}: CPU 常時割当が有効")
        if svc.get("region") != "asia-northeast1":
            problems.append(f"{name}: region={svc.get('region')}")
        if svc.get("allow_unauthenticated"):
            problems.append(f"{name}: 未認証アクセスが許可されています")
    return problems
