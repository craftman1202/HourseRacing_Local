"""再現性の記録（MLflow）。

乱数シード固定だけでは再現性は担保できない。各実行について、Git コミットハッシュ、
conf のスナップショット、入力データの manifest SHA-256 集約値（dataset_version）、
ライブラリバージョン、全評価指標を記録する（RP-02/03）。
"""

from __future__ import annotations

import json
import platform
import subprocess
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

TRACKED_LIBS = (
    "pandas", "numpy", "scipy", "duckdb", "pyarrow", "lightgbm",
    "torch", "numpyro", "jax", "optuna", "scikit-learn", "mlflow",
)


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in TRACKED_LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "not-installed"
    return out


def git_commit(repo: str | Path = ".") -> str:
    """Git 管理外なら 'not-a-git-repo' を返す。

    ここで例外にすると実験自体が止まる。値が取れなかったことを記録に残して
    先へ進めるほうが、記録が欠けるより良い。
    """
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "not-a-git-repo"
    except (OSError, subprocess.SubprocessError):
        return "not-a-git-repo"


def conf_snapshot(conf_dir: str | Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(conf_dir).glob("*.yaml"))}


class RunTracker:
    """MLflow が無い環境でも記録が落ちないよう、JSON へのフォールバックを持つ。"""

    def __init__(self, experiment: str, artifacts_dir: str | Path,
                 tracking_uri: str | None = None) -> None:
        self.experiment = experiment
        self.dir = Path(artifacts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backend = "json"
        self._mlflow = None
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri or f"file://{self.dir.resolve()}/mlruns")
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
            self.backend = "mlflow"
        except Exception:  # noqa: BLE001 - 記録基盤の不在で実験を止めない
            pass
        self.runs: list[dict[str, Any]] = []

    @contextmanager
    def run(self, name: str, dataset_version: str, conf_dir: str | Path,
            tags: dict[str, str] | None = None) -> Iterator["RunHandle"]:
        record: dict[str, Any] = {
            "run_name": name,
            "git_commit": git_commit(),
            "dataset_version": dataset_version,
            "libraries": library_versions(),
            "conf": conf_snapshot(conf_dir),
            "params": {}, "metrics": {}, "tags": dict(tags or {}),
        }
        handle = RunHandle(record)
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=name):
                self._mlflow.set_tags({
                    "git_commit": record["git_commit"],
                    "dataset_version": dataset_version, **(tags or {})})
                self._mlflow.log_dict(record["libraries"], "libraries.json")
                for fname, text in record["conf"].items():
                    self._mlflow.log_text(text, f"conf/{fname}")
                yield handle
                if handle.record["params"]:
                    self._mlflow.log_params(_flatten(handle.record["params"]))
                numeric = {k: v for k, v in handle.record["metrics"].items()
                           if isinstance(v, (int, float))}
                if numeric:
                    self._mlflow.log_metrics(numeric)
        else:
            yield handle
        self.runs.append(handle.record)
        self._flush()

    def _flush(self) -> None:
        (self.dir / "runs.json").write_text(
            json.dumps(self.runs, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def completeness(self) -> dict[str, bool]:
        """RP-02: 記録に欠落が無いか。"""
        if not self.runs:
            return {"has_runs": False}
        r = self.runs[-1]
        return {
            "has_runs": True,
            "git_commit": bool(r["git_commit"]),
            "conf_snapshot": len(r["conf"]) > 0,
            "dataset_version": bool(r["dataset_version"]),
            "libraries": all(v != "not-installed" or k in ("numpyro", "jax", "mlflow")
                             for k, v in r["libraries"].items()),
            "metrics": len(r["metrics"]) > 0,
        }


class RunHandle:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    def log_params(self, params: dict[str, Any]) -> None:
        self.record["params"].update(params)

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.record["metrics"].update(metrics)


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = str(v)
    return out
