"""配布済みリリースからモデルを読み込む。

推論側は PyTorch を持たない（コールドスタート短縮のため）。TabM は ONNX、
LightGBM は Booster テキスト、条件付きロジットは係数ベクトルだけで動く。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import ArtifactIntegrityError


@dataclass
class LinearScorer:
    """条件付きロジット。係数だけあれば推論できる。"""

    name: str
    beta: np.ndarray
    features: list[str]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = _matrix(frame, self.features)
        return x @ self.beta


@dataclass
class LgbmScorer:
    name: str
    booster: object
    features: list[str]

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(_frame(frame, self.features)))


@dataclass
class OnnxScorer:
    """TabM の ONNX 版。onnxruntime だけで動く（PyTorch 不要）。"""

    name: str
    session: object
    features: list[str]
    input_name: str

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = _matrix(frame, self.features).astype(np.float32)
        out = self.session.run(None, {self.input_name: x})[0]
        return np.asarray(out).reshape(len(frame), -1).mean(axis=1)


def _frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"特徴量が足りません: {missing}")
    return frame[features]


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """既に標準化済みの列を行列にする。

    補完と標準化は `Standardizer`（配布物の standardizer.json）が済ませている。
    ここで 0 埋めするのは、標準化後にも残る非有限値（元が inf など）への保険。
    """
    x = _frame(frame, features).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class Standardizer:
    """学習時に凍結した補完値・標準化統計量。

    学習は `prepare()` を通した特徴量で係数と分割点を決めている。推論で生の値を
    渡すと、線形モデルは係数の単位が合わず、GBDT は分割点が合わない。
    推論時は1レース分しか手元に無いので、その場で計算しても代わりにならない
    （「レース内での z 値」という別の量になる）。
    """

    stats: dict[str, dict[str, float]]

    def apply(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = frame.copy()
        for c in features:
            st = self.stats.get(c)
            if st is None:
                raise KeyError(
                    f"{c} の標準化統計量が配布物にありません。"
                    "学習と推論で入力のスケールが揃いません。")
            s = pd.to_numeric(out[c], errors="coerce").fillna(st["median"])
            out[c] = (s - st["mean"]) / (st["std"] or 1.0)
        return out


def load_models(release) -> tuple[dict, "Standardizer"]:
    """リリースディレクトリから、使えるモデルと標準化器を読む。

    無いモデルは黙って飛ばす。ただしアンサンブル重みが付いているのに実体が
    無い場合は、重みの前提が崩れるので例外にする。

    標準化器は必須。学習は補完・標準化を通した特徴量で係数と分割点を決めて
    いるので、これが無いまま推論すると生の値をモデルに渡すことになる。
    """
    path = Path(release.path)
    spec = json.loads((path / "feature_spec.json").read_text(encoding="utf-8"))
    features = list(spec["names"])
    models: dict = {}

    std_file = path / "standardizer.json"
    if not std_file.exists():
        raise ArtifactIntegrityError(
            f"{release.release_id}: standardizer.json がありません。"
            "学習時の補完値・標準化統計量が無いと、推論だけ生の値をモデルに"
            "渡すことになります。")
    standardizer = Standardizer(json.loads(std_file.read_text(encoding="utf-8")))

    beta_file = path / "clogit_beta.json"
    if beta_file.exists():
        payload = json.loads(beta_file.read_text(encoding="utf-8"))
        models["clogit"] = LinearScorer(
            "clogit", np.asarray([payload["beta"][f] for f in features], dtype=float),
            features)

    lgbm_file = path / "lgbm_rank.txt"
    if lgbm_file.exists():
        import lightgbm as lgb

        models["lgbm"] = LgbmScorer(
            "lgbm", lgb.Booster(model_file=str(lgbm_file)), features)

    onnx_file = path / "tabm.onnx"
    if onnx_file.exists():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_file),
                                    providers=["CPUExecutionProvider"])
        models["tabm"] = OnnxScorer("tabm", sess, features,
                                    sess.get_inputs()[0].name)

    weighted = {k for k, w in (release.manifest.ensemble_weights or {}).items() if w > 0}
    missing = weighted - set(models)
    if missing:
        raise RuntimeError(
            f"アンサンブル重みが付いているのに実体が無いモデル: {sorted(missing)}。"
            "重みの前提が崩れるので推論しません。")
    return models, standardizer


@dataclass
class PlaceModels:
    """複勝専用モデル（2026-09-27 追加）。単勝と同じ特徴量・標準化統計量を使う。

    リリースに `place_meta.json` がある版だけが持つ。ばんえい・旧リリースには無く、
    その場合の推論は単勝だけの従来経路のまま（複勝確率を単勝モデルから作って
    賭け金を決めることはしない — 検証していない組み合わせになる）。
    """

    models: dict
    temperatures: dict[str, float]
    weights: dict[str, float]
    ensemble_temperature: float
    release_meta: dict


def load_place_models(release) -> PlaceModels | None:
    path = Path(release.path)
    meta_file = path / "place_meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    spec = json.loads((path / "feature_spec.json").read_text(encoding="utf-8"))
    features = list(spec["names"])
    if list(meta.get("feature_names", features)) != features:
        raise ArtifactIntegrityError(
            f"{release.release_id}: 複勝モデルの特徴量が単勝モデルと一致しません。"
            "標準化統計量を共有できないので読み込みません。")
    models: dict = {}
    beta_file = path / "place_clogit_beta.json"
    if beta_file.exists():
        payload = json.loads(beta_file.read_text(encoding="utf-8"))
        models["clogit"] = LinearScorer(
            "place_clogit", np.asarray([payload["beta"][f] for f in features], dtype=float),
            features)
    lgbm_file = path / "place_lgbm_rank.txt"
    if lgbm_file.exists():
        import lightgbm as lgb

        models["lgbm"] = LgbmScorer("place_lgbm", lgb.Booster(model_file=str(lgbm_file)),
                                    features)
    onnx_file = path / "place_tabm.onnx"
    if onnx_file.exists():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
        models["tabm"] = OnnxScorer("place_tabm", sess, features, sess.get_inputs()[0].name)

    weights = {k: float(v) for k, v in meta["ensemble_weights"].items() if v > 0}
    missing = set(weights) - set(models)
    if missing:
        raise ArtifactIntegrityError(
            f"{release.release_id}: 複勝モデルの重みが付いているのに実体が無い: {sorted(missing)}")
    return PlaceModels(models=models, temperatures=dict(meta["temperatures"]),
                       weights=weights,
                       ensemble_temperature=float(meta["ensemble_temperature"]),
                       release_meta=meta)
