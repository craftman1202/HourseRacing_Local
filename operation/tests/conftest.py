"""共有フィクスチャ。

テスト仕様 §1.3 に従い、外部依存はすべて代替実装に差し替える。
実 GCP・実 Discord に打つテストは `costly` マーカーでオプトインとし、既定では走らない。
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from narops.db.schema import RECORD_COLUMN_MAP
import pytest

from narops.clock import FixedClock, jst_datetime, to_utc
from narops.config import OpsConfig
from narops.db import schema
from narops.db.backend import Warehouse
from narops.model.manifest import FeatureSpec, Manifest

BASE_DAY = jst_datetime(2026, 8, 25, 10, 0)


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(BASE_DAY)


@pytest.fixture
def cfg() -> OpsConfig:
    return OpsConfig.load()


@pytest.fixture
def wh() -> Warehouse:
    w = Warehouse(":memory:", max_bytes_billed=2_000_000_000)
    schema.create_all(w)
    yield w
    w.close()


def make_results(n_races: int = 12, start_day: int = 20, seed: int = 0,
                 base_month: int = 8) -> pd.DataFrame:
    """確定層に入れる結果データ。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        day = start_day + i // 4
        baba = 20 + (i % 3)
        race_no = i % 12 + 1
        rid = f"{baba:02d}2026{base_month:02d}{day:02d}{race_no:02d}"
        start = jst_datetime(2026, base_month, day, 10 + race_no % 10, 30)
        n = int(rng.integers(6, 13))
        finish = rng.permutation(np.arange(1, n + 1))
        for k in range(n):
            rows.append({
                "race_id": rid, "horse_no": k + 1,
                "horse_sk": f"H{(i * 7 + k) % 40:04d}",
                "jockey_sk": f"J{(i + k) % 12:03d}",
                "trainer_sk": f"T{(i + k) % 9:03d}",
                "sire_sk": f"S{(i + k) % 6:03d}",
                "race_date": start.date(), "start_ts": to_utc(start),
                "baba_code": baba, "distance": 1200 + 200 * (i % 4),
                "finish_pos": int(finish[k]), "is_win": int(finish[k] == 1),
                "time_sec": 72.0 + float(finish[k]) * 0.2,
                "speed_index": float(-finish[k] * 0.3 + rng.normal(0, 0.1)),
            })
    df = pd.DataFrame(rows)
    return _attach_declared(df)


def _attach_declared(df: pd.DataFrame) -> pd.DataFrame:
    """NAR 申告の累積成績8列を as-of で足す。

    これが無いと申告値由来の特徴量が全行 NaN になり、運用側のテストだけ
    「その16列が存在しない世界」で通ってしまう。実データには必ず入っている。
    """
    from nar import synth

    out = df.copy()
    out["turn"] = np.where(out["baba_code"] % 2 == 0, "右", "左")
    out["baba_condition"] = "良"
    race = out[["race_id", "turn", "baba_condition"]].drop_duplicates("race_id")
    with_records = synth._declared_records(out.drop(columns=["turn", "baba_condition"]),
                                           race)
    rename = {v: k for k, v in RECORD_COLUMN_MAP.items()}
    return with_records.rename(columns=rename).assign(
        turn=out["turn"].to_numpy(), baba_condition=out["baba_condition"].to_numpy())


@pytest.fixture
def results() -> pd.DataFrame:
    return make_results()


@pytest.fixture
def populated(wh, results, clock):
    """確定層に履歴が入った状態。"""
    from narops.db.merge import merge_final

    merge_final(wh, results, source_sha256="a" * 64, clock=clock)
    return wh


@pytest.fixture(scope="session")
def feature_spec() -> FeatureSpec:
    """実際にビルダーが出す特徴量から spec を導出する。

    手書きの spec を manifest に入れると、推論側が作る実物と一致しないまま
    テストだけ通ってしまう。publish 時も同じ経路（学習出力 → spec）にする。
    """
    from nar.config import feature_config
    from nar.features.builder import build as build_features

    from narops.features import feature_spec_of

    results = make_results(n_races=8, seed=1).rename(columns=RECORD_COLUMN_MAP)
    race_meta = (results[["race_id", "race_date", "start_ts", "baba_code", "distance",
                          "turn", "baba_condition"]]
                 .drop_duplicates("race_id")
                 .assign(race_no=1, surface="ダ", class_level=1, prize_yen=1_000_000))
    race_meta = race_meta.join(
        results.groupby("race_id").size().rename("n_runners"), on="race_id")
    feat = build_features(results, race_meta, feature_config())
    return feature_spec_of(feat)


@pytest.fixture
def release_dir(tmp_path, feature_spec) -> Path:
    """manifest とアーティファクトが揃ったリリースディレクトリ。"""
    d = tmp_path / "src_release"
    d.mkdir()
    (d / "lgbm_rank.txt").write_text("booster-bytes", encoding="utf-8")
    (d / "clogit_beta.parquet").write_text("beta-bytes", encoding="utf-8")
    # 学習時の補完値・標準化統計量。これが無いリリースは読み込ませない設計なので、
    # フィクスチャにも必ず入れる（本番と同じ形にしておく）
    std = json.dumps({n: {"median": 0.0, "mean": 0.0, "std": 1.0}
                      for n in feature_spec.names}, ensure_ascii=False)
    (d / "standardizer.json").write_text(std, encoding="utf-8")

    sha = {
        "lgbm_rank.txt": hashlib.sha256(b"booster-bytes").hexdigest(),
        "clogit_beta.parquet": hashlib.sha256(b"beta-bytes").hexdigest(),
        "standardizer.json": hashlib.sha256(std.encode("utf-8")).hexdigest(),
    }
    Manifest(
        model_id="v2026.08.24-A", dataset_version="ds-" + "0" * 8,
        train_period={"start": "2010-01-01", "end": "2023-12-31"},
        feature_spec_hash=feature_spec.hash(), feature_spec=feature_spec,
        model_sha256=sha,
        oos_metrics={"race_nll": 1.90, "top1": 0.34, "ece": 0.003},
        lookback_days=180, track="A", calibration={"temperature": 1.2},
        git_commit="abc1234", ensemble_weights={"lgbm": 0.3, "tabm": 0.7},
    ).write(d / "manifest.json")
    (d / "feature_spec.json").write_text(
        json.dumps(feature_spec.to_dict(), ensure_ascii=False), encoding="utf-8")
    return d


@pytest.fixture
def registry(tmp_path, release_dir):
    from narops.model.registry import ModelRegistry

    reg = ModelRegistry(tmp_path / "nar-model")
    reg.publish("v2026.08.24-A", release_dir)
    reg.set_current("v2026.08.24-A", actor="test", reason="initial")
    return reg


@pytest.fixture
def schedule() -> pd.DataFrame:
    rows = []
    for i in range(6):
        start = jst_datetime(2026, 8, 25, 15 + i // 2, 30 * (i % 2))
        rows.append({
            "race_id": f"20202608{25:02d}{i + 1:02d}", "race_date": start.date(),
            "baba_code": 20, "race_no": i + 1, "start_ts": to_utc(start),
            "status": "scheduled",
        })
    return pd.DataFrame(rows)
