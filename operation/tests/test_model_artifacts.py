"""MP-01..08: モデル配布・アーティファクト。

配布物の取り違えを manifest の自己検証で防ぐ。
"""

from __future__ import annotations

import json

import pytest

from narops.errors import ArtifactIntegrityError, FeatureSpecMismatch, VersionMixError
from narops.model.manifest import (
    FeatureSpec, Manifest, verify_artifacts, verify_feature_spec, verify_lookback,
    verify_no_version_mix,
)

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------------ MP-01
def test_mp01_required_fields_present(release_dir):
    m = Manifest.read(release_dir / "manifest.json")
    for f in ("model_id", "dataset_version", "train_period", "feature_spec_hash",
              "model_sha256", "oos_metrics", "lookback_days", "track", "calibration"):
        assert getattr(m, f), f"{f} が空です"


@pytest.mark.parametrize("missing", ["model_id", "dataset_version", "lookback_days",
                                     "feature_spec_hash", "oos_metrics"])
def test_mp01_missing_field_is_rejected(release_dir, missing):
    d = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    d.pop(missing)
    with pytest.raises(ArtifactIntegrityError, match="必須フィールド"):
        Manifest.from_dict(d)


def test_mp01_invalid_track_rejected(release_dir):
    d = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    d["track"] = "C"
    with pytest.raises(ArtifactIntegrityError, match="track"):
        Manifest.from_dict(d)


def test_mp01_ensemble_weights_must_be_simplex(release_dir):
    d = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    d["ensemble_weights"] = {"lgbm": 0.6, "tabm": 0.6}
    with pytest.raises(ArtifactIntegrityError, match="総和1"):
        Manifest.from_dict(d)


# ------------------------------------------------------------- MP-02 / RL-04
@pytest.mark.component
def test_mp02_artifact_hashes_match(release_dir):
    verify_artifacts(release_dir, Manifest.read(release_dir / "manifest.json"))


@pytest.mark.component
def test_mp02_tampered_artifact_fails_startup(release_dir):
    """改竄したアーティファクトはハッシュ不一致で読み込まれない（RL-04）。"""
    (release_dir / "lgbm_rank.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="ハッシュ不一致"):
        verify_artifacts(release_dir, Manifest.read(release_dir / "manifest.json"))


@pytest.mark.component
def test_mp02_missing_artifact_fails_startup(release_dir):
    (release_dir / "clogit_beta.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="ファイルがありません"):
        verify_artifacts(release_dir, Manifest.read(release_dir / "manifest.json"))


# ------------------------------------------------------------------------ MP-03
def test_mp03_feature_spec_hash_matches(release_dir, feature_spec):
    verify_feature_spec(feature_spec, Manifest.read(release_dir / "manifest.json"))


def test_mp03_column_order_change_is_detected(release_dir, feature_spec):
    """列名集合が同じでも順序が違えば別物。位置ベースの推論が静かに壊れる。"""
    swapped = FeatureSpec(
        tuple(list(feature_spec.names[1:2]) + list(feature_spec.names[0:1])
              + list(feature_spec.names[2:])),
        feature_spec.dtypes, feature_spec.missing)
    with pytest.raises(FeatureSpecMismatch):
        verify_feature_spec(swapped, Manifest.read(release_dir / "manifest.json"))


def test_mp03_dtype_change_is_detected(release_dir, feature_spec):
    first = feature_spec.names[0]
    other = "float32" if feature_spec.dtypes[first] != "float32" else "int64"
    changed = FeatureSpec(feature_spec.names,
                          {**feature_spec.dtypes, first: other},
                          feature_spec.missing)
    with pytest.raises(FeatureSpecMismatch):
        verify_feature_spec(changed, Manifest.read(release_dir / "manifest.json"))


def test_mp03_missing_representation_is_frozen(release_dir, feature_spec):
    """欠損表現（NaN か 0 埋めか）まで凍結する。SK-04 の前提。"""
    zero_filled = FeatureSpec(feature_spec.names, feature_spec.dtypes,
                              {n: "zero" for n in feature_spec.names})
    with pytest.raises(FeatureSpecMismatch):
        verify_feature_spec(zero_filled, Manifest.read(release_dir / "manifest.json"))


# ------------------------------------------------------------------------ MP-04
def test_mp04_lookback_must_cover_longest_window(release_dir):
    m = Manifest.read(release_dir / "manifest.json")
    verify_lookback(m, 180)
    with pytest.raises(ArtifactIntegrityError, match="lookback_days"):
        verify_lookback(m, 365)


# ------------------------------------------------------------------------ MP-05
@pytest.mark.component
def test_mp05_current_points_to_existing_release(registry):
    assert registry.current_id() == "v2026.08.24-A"
    assert registry.current().path.is_dir()


@pytest.mark.component
def test_mp05_cannot_point_current_at_missing_release(registry):
    with pytest.raises(ArtifactIntegrityError, match="実在しません"):
        registry.set_current("v9999.99.99-Z")


@pytest.mark.component
def test_mp05_switch_is_atomic_no_tmp_left(registry, release_dir, tmp_path):
    registry.publish("v2026.09.01-A", release_dir)
    registry.set_current("v2026.09.01-A", actor="tester")
    assert registry.current_id() == "v2026.09.01-A"
    assert not (registry.root / "current.json.tmp").exists()


@pytest.mark.component
def test_mp05_broken_release_is_not_promoted(registry, release_dir, tmp_path):
    registry.publish("v2026.09.02-A", release_dir)
    (registry.root / "releases" / "v2026.09.02-A" / "lgbm_rank.txt").write_text("x")
    with pytest.raises(ArtifactIntegrityError):
        registry.set_current("v2026.09.02-A")
    assert registry.current_id() == "v2026.08.24-A", "壊れた版に切り替わってしまいました"


# ------------------------------------------------------------------------ MP-06
def test_mp06_version_mix_is_rejected(release_dir):
    a = Manifest.read(release_dir / "manifest.json")
    b = Manifest.read(release_dir / "manifest.json")
    b.dataset_version = "ds-different"
    verify_no_version_mix([a, a])
    with pytest.raises(VersionMixError, match="dataset_version"):
        verify_no_version_mix([a, b])


# ------------------------------------------------------------------------ MP-07
@pytest.mark.component
def test_mp07_onnx_equivalence_check():
    """変換後の予測差が 1e-5 未満、Top-1 一致率 100%。"""
    import numpy as np

    from narops.model.onnx_check import compare_outputs

    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 8))
    ok = compare_outputs(a, a + 1e-7, race_sizes=[10] * 5)
    assert ok.max_abs_diff < 1e-5 and ok.top1_match == 1.0 and ok.passed

    bad = compare_outputs(a, a + 1e-3, race_sizes=[10] * 5)
    assert not bad.passed, "1e-3 の差を許容してはいけません"


@pytest.mark.component
def test_mp07_top1_disagreement_fails_even_with_small_diff():
    """最大差が小さくても Top-1 が入れ替わったら不合格にする。"""
    import numpy as np

    from narops.model.onnx_check import compare_outputs

    a = np.array([[0.0], [0.5], [0.5000001], [0.2]])
    b = a.copy()
    b[1], b[2] = a[2], a[1]
    res = compare_outputs(a, b, race_sizes=[4])
    assert res.max_abs_diff < 1e-5
    assert res.top1_match < 1.0 and not res.passed


# ------------------------------------------------------------------------ MP-08
@pytest.mark.component
def test_mp08_rollback_returns_to_previous_release(registry, release_dir):
    registry.publish("v2026.09.10-A", release_dir)
    registry.set_current("v2026.09.10-A", actor="tester", reason="promote")
    assert registry.current_id() == "v2026.09.10-A"

    previous = registry.rollback(actor="oncall")
    assert previous == "v2026.08.24-A"
    assert registry.current_id() == "v2026.08.24-A"


@pytest.mark.component
def test_promotion_is_audited(registry, release_dir):
    registry.publish("v2026.09.11-A", release_dir)
    registry.set_current("v2026.09.11-A", actor="alice", reason="promote")
    last = registry.audit_log()[-1]
    assert last["actor"] == "alice"
    assert last["from"] == "v2026.08.24-A" and last["to"] == "v2026.09.11-A"
    assert last["at"]


@pytest.mark.component
def test_releases_are_immutable(registry, release_dir):
    with pytest.raises(ArtifactIntegrityError, match="既に存在"):
        registry.publish("v2026.08.24-A", release_dir)


# ------------------------------------------------------------ ONNX の外部重み
def test_release_carries_onnx_sidecar_files(tmp_path, feature_spec):
    """ONNX の重みが別ファイルに出る形式でも、まとめて配ること。

    本体だけ配ると onnxruntime が
    「External data path validation failed」で読めない（実際にそうなった）。
    """
    from narops.publish import build_release

    src = tmp_path / "final"
    src.mkdir()
    (src / "tabm.onnx").write_text("graph", encoding="utf-8")
    (src / "tabm.onnx.data").write_text("weights", encoding="utf-8")

    green = {t: "GREEN" for t in
             ("IG-14", "LK-05", "LK-06", "EV-03", "RF-01", "RF-02", "RF-03",
              "RF-07", "RF-08", "CV-07")}
    res = build_release(
        release_id="v-onnx", out_dir=tmp_path / "out",
        feature_names=list(feature_spec.names), dataset_version="d",
        train_period={"start": "1998-01-01", "end": "2023-08-04"},
        oos_metrics={}, ensemble_weights={"tabm": 1.0}, lookback_days=180,
        temperature=1.0, test_results=green,
        standardizer={n: {"median": 0.0, "mean": 0.0, "std": 1.0}
                      for n in feature_spec.names},
        tabm_onnx_path=src / "tabm.onnx")

    assert (res.path / "tabm.onnx").exists()
    assert (res.path / "tabm.onnx.data").exists(), "外部重みが配られていません"
    assert "tabm.onnx.data" in res.manifest.model_sha256, "改竄検知の対象外です"


def test_app_can_load_a_release_from_gcs(monkeypatch, tmp_path, release_dir):
    """Cloud Run は永続ディスクを持たない。配布物は GCS から取る。

    ローカルディレクトリだけを見る作りだと、本番では常に「モデル無し」で
    起動して推論できない。
    """
    from types import SimpleNamespace

    from narops import app as app_mod
    from narops.model.registry import ModelRegistry

    # GCS の代わりに、ローカルのリリースを配る偽レジストリ
    class FakeRemote:
        def __init__(self, **kw):
            pass

        def current_id(self):
            return "v-gcs"

        def download_release(self, release_id, dest):
            import shutil
            from pathlib import Path

            out = Path(dest) / release_id
            shutil.copytree(release_dir, out)
            return str(out)

    monkeypatch.setattr("narops.gcp.GcsModelRegistry", FakeRemote)
    monkeypatch.delenv("NAROPS_MODEL_ROOT", raising=False)
    monkeypatch.setenv("NAROPS_MODEL_BUCKET", "nar-model-test")
    monkeypatch.setenv("NAROPS_MODEL_CACHE", str(tmp_path))

    svc = SimpleNamespace()
    # 検証したいのは「GCS の current を見て取得し、レジストリに載せる」まで。
    # ダミーの booster を実際に読み込ませる必要はない。
    monkeypatch.setattr("narops.runtime.load_models",
                        lambda release: ({}, object()))
    app_mod._attach_model(svc, None)

    assert isinstance(svc.registry, ModelRegistry)
    assert svc.registry.current_id() == "v-gcs"
    assert svc.manifest is not None
    assert svc.standardizer is not None


def test_app_warns_instead_of_crashing_without_any_model_source(monkeypatch):
    from types import SimpleNamespace

    from narops import app as app_mod

    monkeypatch.delenv("NAROPS_MODEL_ROOT", raising=False)
    monkeypatch.delenv("NAROPS_MODEL_BUCKET", raising=False)
    svc = SimpleNamespace()
    app_mod._attach_model(svc, None)
    assert not hasattr(svc, "manifest") or svc.manifest is None
