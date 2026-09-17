"""学習成果物からモデルリリースを作る（`nar model publish` 相当）。

設計書 §8.1: ONNX 変換と PyTorch 版との出力一致検証、feature_spec.json の生成、
GCS へのアップロード、シャドー推論の起動。

公開ゲート（RL-01）を必ず通す。学習側の Blocker が全 GREEN でなければ publish しない。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .model.manifest import FeatureSpec, Manifest, sha256_file
from .release import assert_publish_gate


@dataclass
class PublishResult:
    release_id: str
    path: Path
    manifest: Manifest
    onnx_check: dict | None = None
    notes: list[str] = None


def build_release(
    *,
    release_id: str,
    out_dir: str | Path,
    feature_names: list[str],
    dataset_version: str,
    train_period: dict[str, str],
    oos_metrics: dict[str, float],
    ensemble_weights: dict[str, float],
    lookback_days: int,
    temperature: float,
    test_results: dict[str, str],
    track: str = "A",
    git_commit: str = "",
    clogit_beta: dict[str, float] | None = None,
    lgbm_booster_path: str | Path | None = None,
    tabm_onnx_path: str | Path | None = None,
    standardizer: dict[str, dict[str, float]] | None = None,
    purpose: str = "evaluation",
    oos_evaluated_on: str | None = None,
    model_temperatures: dict[str, float] | None = None,
) -> PublishResult:
    """リリースディレクトリを組み立てる。

    feature_spec は**学習が実際に出した列**から作る。手書きすると推論側と
    一致せず、MP-03 が本番で初めて落ちる。
    """
    assert_publish_gate(test_results)          # RL-01

    out = Path(out_dir) / release_id
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    spec = FeatureSpec(tuple(feature_names),
                       {n: "float64" for n in feature_names},
                       {n: "nan" for n in feature_names})
    (out / "feature_spec.json").write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if standardizer is None:
        raise ValueError(
            "standardizer が指定されていません。学習は補完値と標準化を通した特徴量で"
            "係数を推定しているので、これを配らないと推論だけ生の値をモデルに渡す"
            "ことになります。")
    (out / "standardizer.json").write_text(
        json.dumps(standardizer, ensure_ascii=False, indent=2), encoding="utf-8")

    if clogit_beta:
        (out / "clogit_beta.json").write_text(
            json.dumps({"beta": clogit_beta}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    if lgbm_booster_path:
        shutil.copy2(lgbm_booster_path, out / "lgbm_rank.txt")
    if tabm_onnx_path:
        # ONNX は重みを外部ファイル（tabm.onnx.data など）に持つ形式で書き出される
        # ことがある。本体だけ配ると onnxruntime が
        # 「External data path validation failed」で読めない。
        # 同じ名前で始まる書き出しをまとめて運ぶ。
        src = Path(tabm_onnx_path)
        for f in sorted(src.parent.glob(src.name + "*")):
            shutil.copy2(f, out / f.name)

    # 実体のあるモデルにだけ重みを残す。無いモデルに重みが付いたままだと
    # 推論側が起動時に落ちる（runtime.load_models）
    available = set()
    if clogit_beta:
        available.add("clogit")
    if lgbm_booster_path:
        available.add("lgbm")
    if tabm_onnx_path:
        available.add("tabm")
    weights = {k: v for k, v in ensemble_weights.items() if k in available and v > 0}
    if not weights:
        raise ValueError("配布できるモデルが1つもありません")
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    dropped = {k: v for k, v in ensemble_weights.items() if k not in weights and v > 0}
    if dropped:
        notes.append(f"実体が無いため重みから除外: {dropped}（残りを再正規化）")

    # 配布しないモデル（重み0・実体無し）の温度は推論側に渡す意味が無い。
    # weights と同じ集合に揃えておく（blend() が見るモデル名と一致させる）。
    model_temps = {k: v for k, v in (model_temperatures or {}).items() if k in weights}

    sha = {p.name: sha256_file(p) for p in sorted(out.iterdir())
           if p.name not in ("manifest.json",)}
    manifest = Manifest(
        model_id=release_id, dataset_version=dataset_version,
        train_period=train_period, feature_spec_hash=spec.hash(), feature_spec=spec,
        model_sha256=sha,
        oos_metrics=oos_metrics, lookback_days=lookback_days, track=track,
        calibration={"temperature": temperature}, git_commit=git_commit,
        ensemble_weights=weights, purpose=purpose,
        oos_evaluated_on=oos_evaluated_on, model_temperatures=model_temps)
    manifest.write(out / "manifest.json")
    return PublishResult(release_id, out, manifest, notes=notes)


def export_tabm_onnx(model, n_features: int, out_path: str | Path,
                     opset: int = 17) -> Path:
    """TabM を ONNX 化する（設計書 §1.1）。

    Cloud Run から PyTorch（数百MB）を排除してコールドスタートを短縮するのが目的。
    変換後は必ず元モデルとの出力一致を検証してから公開する（MP-07）。
    """
    import torch

    net = model._fitted().eval()
    mu = torch.as_tensor(model._mu, dtype=torch.float32)
    sd = torch.as_tensor(model._sd, dtype=torch.float32)

    class Wrapped(torch.nn.Module):
        """標準化を含めて ONNX に埋める。

        推論側で標準化を再実装すると、学習時の統計量とズレる余地ができる。
        統計量ごと固めてしまうのが安全（SK-02 と同じ思想）。
        """

        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("mu", mu)
            self.register_buffer("sd", sd)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = (x - self.mu) / self.sd
            return self.net(z).transpose(0, 1)      # (n, k)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(4, n_features, dtype=torch.float32)
    # ラッパ自身も eval にする。内側の net だけ eval にしても、新しく作った
    # Module の training フラグは True のままで、書き出し時に学習モードの
    # 挙動が固定される余地が残る。
    wrapped = Wrapped().eval()
    torch.onnx.export(
        wrapped, (dummy,), str(out), opset_version=opset,
        input_names=["features"], output_names=["member_scores"],
        dynamic_axes={"features": {0: "n"}, "member_scores": {0: "n"}})
    return out


def verify_onnx(model, onnx_path: str | Path, sample: np.ndarray,
                race_sizes: list[int]) -> dict:
    """PyTorch 版と ONNX 版の出力一致（MP-07）。"""
    import onnxruntime as ort
    import torch

    from .model.onnx_check import assert_equivalent

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: sample.astype(np.float32)})[0]
    onnx_scores = np.asarray(got).mean(axis=1, keepdims=True)

    net = model._fitted().eval()
    with torch.no_grad():
        z = (torch.as_tensor(sample, dtype=torch.float32)
             - torch.as_tensor(model._mu, dtype=torch.float32)) / torch.as_tensor(
                 model._sd, dtype=torch.float32)
        torch_scores = net(z).transpose(0, 1).mean(dim=1, keepdim=True).numpy()

    res = assert_equivalent(torch_scores, onnx_scores, race_sizes)
    return {"max_abs_diff": res.max_abs_diff, "top1_match": res.top1_match,
            "n_races": res.n_races, "passed": res.passed}


def upload_release(local_dir: str | Path, bucket: str, project: str,
                   release_id: str) -> list[str]:
    """GCS へアップロード。current ポインタは触らない（昇格は別操作）。"""
    from google.cloud import storage

    from .gcp import assert_owned

    assert_owned(bucket)
    client = storage.Client(project=project)
    b = client.bucket(bucket)
    uploaded = []
    for f in sorted(Path(local_dir).iterdir()):
        if f.is_file():
            blob = b.blob(f"releases/{release_id}/{f.name}")
            blob.upload_from_filename(f)
            uploaded.append(blob.name)
    return uploaded
