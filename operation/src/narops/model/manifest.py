"""モデル配布の契約（manifest）。

ローカル学習とクラウド推論を分離した設計の弱点は「配布物の取り違え」。
manifest を単一の真実とし、実行時に自己検証させる（テスト仕様 §2）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..errors import ArtifactIntegrityError, FeatureSpecMismatch, VersionMixError

REQUIRED_FIELDS = (
    "model_id", "dataset_version", "train_period", "feature_spec_hash",
    "model_sha256", "oos_metrics", "lookback_days", "track", "calibration",
)
VALID_TRACKS = ("A", "B")


@dataclass
class FeatureSpec:
    """特徴量の名前・順序・dtype・欠損補完規則。

    順序をハッシュに含めるのが要点。列名集合が同じでも順序が違えば、
    位置ベースで行列を組む推論側が静かに壊れる。
    """

    names: tuple[str, ...]
    dtypes: dict[str, str]
    # 「欠損をどう表すか」を凍結する。ゼロ・平均代入を運用側で足せないようにする（SK-04）
    missing: dict[str, str] = field(default_factory=dict)

    def hash(self) -> str:
        payload = json.dumps(
            {"names": list(self.names),
             "dtypes": [self.dtypes[n] for n in self.names],
             "missing": [self.missing.get(n, "nan") for n in self.names]},
            ensure_ascii=False, sort_keys=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "dtypes": self.dtypes, "missing": self.missing}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatureSpec":
        return cls(tuple(d["names"]), dict(d["dtypes"]), dict(d.get("missing", {})))


@dataclass
class Manifest:
    model_id: str
    dataset_version: str
    train_period: dict[str, str]
    feature_spec_hash: str
    model_sha256: dict[str, str]        # 相対パス -> SHA-256
    oos_metrics: dict[str, float]
    lookback_days: int
    track: str
    calibration: dict[str, float]
    git_commit: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    # 特徴量仕様の本体。ハッシュだけだと、不一致が出たときに「名前・順序・dtype・
    # 欠損表現のどれか」としか言えず、本番で原因を特定できない。
    # 旧リリースには無いので省略可能。
    feature_spec: "FeatureSpec | None" = None
    # evaluation（OOS 境界で打ち切り）か production（全データ）か。
    # oos_metrics は evaluation 版で測った数字なので、これが無いと
    # 「この重みでこの OOS が出た」と誤読される。
    purpose: str = "evaluation"
    # OOS を測ったモデルの学習期間末。production 版では自分の train_period と違う。
    oos_evaluated_on: str | None = None

    # ------------------------------------------------------------------ MP-01
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Manifest":
        missing = [f for f in REQUIRED_FIELDS if f not in d]
        if missing:
            raise ArtifactIntegrityError(f"manifest の必須フィールドが欠けています: {missing}")
        payload = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        spec = payload.get("feature_spec")
        if isinstance(spec, dict):
            payload["feature_spec"] = FeatureSpec(
                names=tuple(spec.get("names", ())),
                dtypes=dict(spec.get("dtypes", {})),
                missing=dict(spec.get("missing", {})))
        m = cls(**payload)
        m.validate()
        return m

    def validate(self) -> None:
        if self.track not in VALID_TRACKS:
            raise ArtifactIntegrityError(f"track は {VALID_TRACKS} のいずれか: {self.track!r}")
        if not isinstance(self.lookback_days, int) or self.lookback_days <= 0:
            raise ArtifactIntegrityError(f"lookback_days が不正: {self.lookback_days!r}")
        if not self.model_sha256:
            raise ArtifactIntegrityError("model_sha256 が空です")
        for key in ("start", "end"):
            if key not in self.train_period:
                raise ArtifactIntegrityError(f"train_period.{key} がありません")
            date.fromisoformat(self.train_period[key])
        for name, h in self.model_sha256.items():
            if len(h) != 64 or not all(c in "0123456789abcdef" for c in h):
                raise ArtifactIntegrityError(f"{name} のハッシュが SHA-256 形式ではありません")
        if self.ensemble_weights:
            total = sum(self.ensemble_weights.values())
            if abs(total - 1.0) > 1e-9 or any(w < 0 for w in self.ensemble_weights.values()):
                raise ArtifactIntegrityError(
                    f"アンサンブル重みが非負かつ総和1ではありません（総和 {total}）")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------- MP-02
def verify_artifacts(release_dir: str | Path, manifest: Manifest) -> None:
    """実ファイルの SHA-256 が manifest と一致することを起動時に確認する。

    不一致なら起動失敗（ヘルスチェック赤）。改竄されたアーティファクトを
    読み込まないための最後の砦（RL-04）。
    """
    root = Path(release_dir)
    problems: list[str] = []
    for rel, expected in manifest.model_sha256.items():
        f = root / rel
        if not f.exists():
            problems.append(f"{rel}: ファイルがありません")
            continue
        actual = sha256_file(f)
        if actual != expected:
            problems.append(f"{rel}: ハッシュ不一致 (期待 {expected[:12]}… / 実測 {actual[:12]}…)")
    if problems:
        raise ArtifactIntegrityError(
            "配布物の完全性検証に失敗しました:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------- MP-03
def verify_feature_spec(spec: FeatureSpec, manifest: Manifest) -> None:
    actual = spec.hash()
    if actual == manifest.feature_spec_hash:
        return
    raise FeatureSpecMismatch(
        f"feature_spec が一致しません（manifest {manifest.feature_spec_hash[:12]}… / "
        f"推論側 {actual[:12]}…）。" + _spec_diff(spec, manifest))


def _spec_diff(spec: FeatureSpec, manifest: Manifest) -> str:
    """どこが違うかを名指しする。

    ハッシュだけ出しても「名前・順序・dtype・欠損表現のどれか」としか言えず、
    調査に毎回スクリプトを書くことになる。manifest に spec 本体があるなら比較する。
    """
    want = getattr(manifest, "feature_spec", None)
    if want is None:
        return ("特徴量の名前・順序・dtype・欠損表現のいずれかが学習時と違います"
                "（manifest に spec 本体が無いため差分は出せません）。")
    lines = []
    if tuple(want.names) != tuple(spec.names):
        missing = [n for n in want.names if n not in spec.names]
        extra = [n for n in spec.names if n not in want.names]
        if missing:
            lines.append(f"推論側に無い列: {missing}")
        if extra:
            lines.append(f"推論側にだけある列: {extra}")
        if not missing and not extra:
            lines.append("列の順序が違います")
    for name in spec.names:
        if name not in want.names:
            continue
        if want.dtypes.get(name) != spec.dtypes.get(name):
            lines.append(f"{name}: dtype {want.dtypes.get(name)} → {spec.dtypes.get(name)}")
        if want.missing.get(name) != spec.missing.get(name):
            lines.append(
                f"{name}: 欠損表現 {want.missing.get(name)} → {spec.missing.get(name)}")
    return "\n  " + "\n  ".join(lines) if lines else "差分を特定できませんでした。"


# ---------------------------------------------------------------------- MP-04
def verify_lookback(manifest: Manifest, max_window_days: int) -> None:
    if manifest.lookback_days < max_window_days:
        raise ArtifactIntegrityError(
            f"manifest の lookback_days={manifest.lookback_days} が as-of 集計の最長窓 "
            f"{max_window_days} 日を下回っています。履歴が足りないまま推論すると"
            "学習時と違う特徴量になります。")


# ---------------------------------------------------------------------- MP-06
def verify_no_version_mix(manifests: list[Manifest]) -> None:
    versions = {m.dataset_version for m in manifests}
    if len(versions) > 1:
        raise VersionMixError(
            f"1推論内で dataset_version が混在しています: {sorted(versions)}。"
            "モデルごとに違う学習データを使うと、アンサンブル重みの前提が崩れます。")
