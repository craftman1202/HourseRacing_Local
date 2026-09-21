"""ばんえい競馬の推論経路。

ばんえいは 200m 直線・そりの重量が勝敗を決める別競技で、平地とは別モデルで
推論する（TR-11 の「必要なら別モデルを立てる」側）。ここで固定するのは
「取り違えが起きないこと」— 平地のモデルにばんえいのレースを流さない、
ばんえいの設定で平地の履歴を拾わない、の2点。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nar.config import feature_config
from narops.errors import InsufficientData
from narops.model.registry import ModelRegistry, current_pointer
from narops.service import ModelBundle, Services

pytestmark = pytest.mark.component

# 学習側のばんえい設定。運用側からは絶対パスで指す（実行位置に依存させない）
BANEI_CONF = str((Path(__file__).resolve().parents[2] / "learning" / "conf_banei"))


class _Scorer:
    def __init__(self, v: float) -> None:
        self.v = v

    def score(self, frame):
        return [self.v] * len(frame)


def _bundle(manifest) -> ModelBundle:
    return ModelBundle(manifest=manifest, models={"a": _Scorer(1.0)})


# ------------------------------------------------------------------ 束の選択
def test_banei_race_is_refused_when_no_banei_model_is_loaded(wh, cfg, clock):
    """ばんえいのレースを平地モデルに流さない。

    黙ってフォールバックすると、距離も回りも無い競技に距離と回りの特徴量で
    学習したモデルを当てた結果で賭け金が決まる。
    """
    svc = Services(wh=wh, cfg=cfg, clock=clock, manifest=object(),
                   models={"a": _Scorer(1.0)})
    for code in (1, 2, 3, 4):
        with pytest.raises(InsufficientData, match="ばんえい"):
            svc.bundle_for(code)


def test_flat_races_are_unaffected_by_the_banei_bundle(wh, cfg, clock):
    flat_manifest, banei_manifest = object(), object()
    svc = Services(wh=wh, cfg=cfg, clock=clock, manifest=flat_manifest,
                   models={"a": _Scorer(1.0)}, banei=_bundle(banei_manifest))
    assert svc.bundle_for(20).manifest is flat_manifest
    assert svc.bundle_for(3).manifest is banei_manifest


def test_supports_banei_is_false_for_a_declared_but_empty_bundle(wh, cfg, clock):
    """manifest だけ差さってモデルが空、という半端な状態を「使える」と数えない。"""
    svc = Services(wh=wh, cfg=cfg, clock=clock,
                   banei=ModelBundle(manifest=object(), models={}))
    assert not svc.supports_banei()
    with pytest.raises(InsufficientData):
        svc.bundle_for(3)


# -------------------------------------------------------------- current の分離
def _banei_release_dir(src: Path, dst: Path) -> Path:
    """平地のリリースを複製し、feature_spec だけばんえいの列に差し替える。

    系統の判定は配布物が宣言している特徴量で行うので、ID の綴りではなく
    spec を変える必要がある。
    """
    import shutil

    shutil.copytree(src, dst)
    spec_path = dst / "feature_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["names"] = ["b_weight_carried", "b_load_ratio", "b_moisture"]
    spec["dtypes"] = {n: "float64" for n in spec["names"]}
    spec["missing"] = {n: "nan" for n in spec["names"]}
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    mf_path = dst / "manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf["feature_spec"] = spec
    from narops.model.manifest import FeatureSpec, sha256_file

    mf["feature_spec_hash"] = FeatureSpec.from_dict(spec).hash()
    mf["model_sha256"] = {k: sha256_file(dst / k) for k in mf.get("model_sha256", {})}
    mf_path.write_text(json.dumps(mf, ensure_ascii=False), encoding="utf-8")
    return dst


def test_each_family_has_its_own_current_pointer(tmp_path, release_dir):
    """平地の昇格がばんえいの current を動かさないこと（およびその逆）。"""
    bdir = _banei_release_dir(release_dir, tmp_path / "banei-src")
    flat = ModelRegistry(tmp_path / "nar-model", family="flat")
    banei = ModelRegistry(tmp_path / "nar-model", family="banei")
    flat.publish("v2026.08.24-A", release_dir)
    banei.publish("v2026.08.24-A-banei", bdir)

    flat.set_current("v2026.08.24-A", actor="test", reason="flat")
    assert banei.current_id() is None, "ばんえいの current が平地の昇格で動いています"

    banei.set_current("v2026.08.24-A-banei", actor="test", reason="banei")
    assert flat.current_id() == "v2026.08.24-A"
    assert banei.current_id() == "v2026.08.24-A-banei"
    assert (tmp_path / "nar-model" / current_pointer("banei")).exists()


def test_a_flat_release_cannot_be_pointed_at_current_banei(tmp_path, release_dir):
    """releases/ は系統をまたいで共有。ID を打ち間違えても止まること。

    平地のモデルを current_banei に置くと、200m 直線のレースが距離と回りの
    特徴量で学習したモデルで採点され、その結果で賭け金が決まる。
    """
    from narops.errors import ArtifactIntegrityError

    banei = ModelRegistry(tmp_path / "nar-model", family="banei")
    banei.publish("v2026.08.24-A", release_dir)          # 中身は平地
    with pytest.raises(ArtifactIntegrityError, match="flat のモデル"):
        banei.set_current("v2026.08.24-A", actor="test", reason="取り違え")
    assert banei.current_id() is None


def test_a_banei_release_cannot_be_pointed_at_current_flat(tmp_path, release_dir):
    from narops.errors import ArtifactIntegrityError

    bdir = _banei_release_dir(release_dir, tmp_path / "banei-src")
    flat = ModelRegistry(tmp_path / "nar-model", family="flat")
    flat.publish("v2026.08.24-A-banei", bdir)
    with pytest.raises(ArtifactIntegrityError, match="banei のモデル"):
        flat.set_current("v2026.08.24-A-banei", actor="test", reason="取り違え")


def test_rollback_stays_inside_its_own_family(tmp_path, release_dir):
    """監査ログは系統をまたいで1本。ロールバックが他系統の ID を掴まないこと。"""
    bdir = _banei_release_dir(release_dir, tmp_path / "banei-src")
    flat = ModelRegistry(tmp_path / "nar-model", family="flat")
    banei = ModelRegistry(tmp_path / "nar-model", family="banei")
    for rid in ("v1-A", "v2-A"):
        flat.publish(rid, release_dir)
    for rid in ("v1-A-banei", "v2-A-banei"):
        banei.publish(rid, bdir)

    flat.set_current("v1-A", actor="t", reason="")
    banei.set_current("v1-A-banei", actor="t", reason="")
    flat.set_current("v2-A", actor="t", reason="")
    banei.set_current("v2-A-banei", actor="t", reason="")

    assert banei.rollback(actor="t") == "v1-A-banei"
    assert flat.current_id() == "v2-A", "平地の current が巻き込まれています"
    assert flat.rollback(actor="t") == "v1-A"


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="未知のモデル系統"):
        current_pointer("jra")


# ------------------------------------------------------- 当日ページ（含水率）
BANEI_TITLE_HTML = (
    '<div class="raceTitle">いつもラブラブ義樹と弘子杯２歳Ｂ－３ ダート 200ｍ（直）'
    ' 天候：晴 馬場：1.8 混合 2歳 規定 賞金 1着390,000円 2着144,000円</div>')
FLAT_TITLE_HTML = (
    '<div class="raceTitle">馬い！記念Ｃ３一 ダート 1500ｍ（左） 天候：曇 馬場：良'
    ' サラブレッド系 一般 賞金 1着900,000円</div>')


def test_banei_race_header_reads_the_moisture_reading():
    """ばんえいの `馬場：1.8` は含水率（%）。平地の 良/稍重 と同じ位置に出る。

    ここを落とすと b_moisture が推論時だけ欠測になり、学習と推論で入力が
    食い違う（そのまま賭け金に効く）。
    """
    from narops.deba_table import parse_race_header

    h = parse_race_header(BANEI_TITLE_HTML)
    assert h["baba_condition"] == "1.8"
    assert h["distance"] == 200
    assert h["turn"] == "直"


def test_flat_race_header_still_reads_the_categorical_condition():
    from narops.deba_table import parse_race_header

    h = parse_race_header(FLAT_TITLE_HTML)
    assert h["baba_condition"] == "良"
    assert h["distance"] == 1500


# 確定層にばんえいの過去走が残ることは、実データ（golden fixture）で
# tests/test_refresh.py::test_load_history_keeps_banei_so_it_has_past_form が固定する。


# ------------------------------------------------------- skew（2系統の同居）
def test_skew_compare_tolerates_mixed_families_in_one_day():
    """同じ営業日に平地とばんえいの snapshot が並んでも Blocker にしない。

    `load_snapshot` は features の JSON を展開するので、両系統が混ざると
    列は和集合になり、相手の系統の列は NaN で並ぶ。ここを乖離と数えると
    ばんえいを載せた日から毎日 skew Blocker が鳴り、**平地の配信まで止まる**。
    """
    from datetime import date

    from narops.skew import compare

    flat = pd.DataFrame({
        "race_id": ["20260829A"] * 2, "horse_no": [1, 2],
        "distance": [1500.0, 1500.0], "d_turn_starts": [12.0, 8.0],
        "b_weight_carried": [np.nan, np.nan], "b_moisture": [np.nan, np.nan],
    })
    banei = pd.DataFrame({
        "race_id": ["03260829A"] * 2, "horse_no": [1, 2],
        "distance": [np.nan, np.nan], "d_turn_starts": [np.nan, np.nan],
        "b_weight_carried": [600.0, 620.0], "b_moisture": [1.8, 1.8],
    })
    day = pd.concat([flat, banei], ignore_index=True)

    report = compare(day, day.copy(), frozenset(), date(2026, 8, 29))
    assert report.is_clean, report.mismatches.to_string()
    assert report.n_compared == 4


def test_skew_compare_still_catches_a_real_divergence_in_a_banei_column():
    """ばんえい固有の列でも乖離はちゃんと検出されること。"""
    from datetime import date

    from narops.skew import compare

    snap = pd.DataFrame({"race_id": ["03260829A"], "horse_no": [1],
                         "b_weight_carried": [600.0]})
    recalc = snap.copy()
    recalc["b_weight_carried"] = [620.0]

    report = compare(snap, recalc, frozenset(), date(2026, 8, 29))
    assert not report.is_clean
    assert report.offending_columns == ["b_weight_carried"]


# ------------------------------------------------------- /infer（経路まるごと）
BANEI_RACE_ID = "032026082905"


def _banei_history(n_races: int = 40, start_day: int = 18, seed: int = 5) -> pd.DataFrame:
    """確定層に入れるばんえいの過去走。

    `conftest.make_results` は平地向け（距離 1200-1800・回りあり）なので、
    ばんえい（200m 直線・含水率が数値）の形で作り直す。
    """
    from narops.clock import jst_datetime, to_utc
    from tests.conftest import _attach_declared

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        day = start_day + i // 4
        race_no = i % 12 + 1
        rid = f"03202608{day:02d}{race_no:02d}"
        start = jst_datetime(2026, 8, day, 10 + race_no % 10, 30)
        n = 10
        finish = rng.permutation(np.arange(1, n + 1))
        for k in range(n):
            rows.append({
                "race_id": rid, "horse_no": k + 1,
                "horse_sk": f"H{(i * 7 + k) % 40:04d}",
                "jockey_sk": f"J{(i + k) % 12:03d}",
                "trainer_sk": f"T{(i + k) % 9:03d}",
                "sire_sk": f"S{(i + k) % 6:03d}",
                "race_date": start.date(), "start_ts": to_utc(start),
                "baba_code": 3, "distance": 200,
                "finish_pos": int(finish[k]), "is_win": int(finish[k] == 1),
                "time_sec": 100.0 + float(finish[k]) * 2.0,
                "speed_index": float(-finish[k] * 0.3 + rng.normal(0, 0.1)),
            })
    df = _attach_declared(pd.DataFrame(rows))
    # ばんえいは直線しかなく、馬場は含水率（数値）
    df["turn"] = "直"
    df["baba_condition"] = "1.8"
    return df


def _banei_card() -> pd.DataFrame:
    """当日の出馬表。当日ページ（DebaTable）が返す形に合わせる。

    ばんえいでは ダート左/右成績 と 最高タイム が空欄で返る（実ページで確認済み）。
    空欄を 0 で埋めないことが declared.parse_record の要件。
    """
    rows = []
    for i in range(10):
        rows.append({
            "race_id": BANEI_RACE_ID, "horse_no": i + 1,
            "horse_sk": f"H{i:04d}", "jockey_sk": f"J{i % 12:03d}",
            "trainer_sk": f"T{i % 9:03d}", "sire_sk": f"S{i % 6:03d}",
            "baba_code": 3, "race_date": pd.Timestamp("2026-08-25").date(),
            "weight_carried": "☆600" if i == 0 else str(600 + (i % 3) * 10),
            "weight_kg": 950.0 + i * 5, "weight_diff": "+2",
            "sex": "牝" if i % 3 == 0 else "牡", "age": 4 + i % 5,
            "騎手成績": f"{i % 3}-1-1-{4 + i}",
            "全成績": f"{i % 4}-2-1-{10 + i}",
            "当競馬場成績": f"{i % 4}-2-1-{10 + i}",
            "うち当距離成績": f"{i % 4}-2-1-{10 + i}",
            "ダート左成績": "", "ダート右成績": "",
            "最高タイム": "", "最高タイム良馬場": "",
        })
    return pd.DataFrame(rows)


def test_infer_runs_end_to_end_for_a_banei_race(wh, cfg):
    """ばんえいのレースが /infer を最後まで通り、予測が書かれること。

    これが落ちていたのが 2026-08-29 の実障害（確定層に過去走が無く
    InsufficientData）。専用モデルと専用設定で通ることを固定する。
    """
    from datetime import timedelta

    from narops.clock import FixedClock, jst_datetime, to_utc
    from narops.db.merge import merge_final
    from narops.mode import Mode, ModelProvenance, OperatingState
    from narops.model.manifest import Manifest
    from narops.service import infer_endpoint, plan_day_endpoint
    from nar.features.builder import asof_features

    start = jst_datetime(2026, 8, 25, 16, 5)
    clock = FixedClock(start - timedelta(minutes=13))
    merge_final(wh, _banei_history(n_races=40, start_day=18, seed=5), clock=clock)

    names = tuple(asof_features(feature_config(BANEI_CONF)))
    spec_names = names
    spec = __import__("narops.model.manifest", fromlist=["FeatureSpec"]).FeatureSpec(
        spec_names, {n: "float64" for n in spec_names}, {n: "nan" for n in spec_names})
    manifest = Manifest(
        model_id="v2026.08.25-A-banei", dataset_version="banei-test",
        train_period={"start": "1998-01-01", "end": "2026-08-24"},
        feature_spec_hash=spec.hash(), model_sha256={}, oos_metrics={},
        lookback_days=180, track="A", calibration={"temperature": 1.0},
        ensemble_weights={"a": 1.0}, feature_spec=spec)

    class _Src:
        def __enter__(self): return self
        def __exit__(self, *e): return None
        def fetch_schedule(self, day):
            return pd.DataFrame([{
                "race_id": BANEI_RACE_ID, "race_date": day, "baba_code": 3,
                "race_no": 5, "track_name": "帯広ば", "start_ts": to_utc(start),
                "status": "scheduled"}])
        def fetch_entry_card(self, race_id, baba_code, day, race_no):
            card = _banei_card()
            card["distance"] = 200
            card["class_level"] = 2
            card["prize_yen"] = 390000.0
            card["surface"] = "ダ"
            card["turn"] = "直"
            card["baba_condition"] = "1.8"
            return card
        def fetch_odds(self, race_id, baba_code, day, race_no):
            return pd.DataFrame(columns=["race_id", "horse_no", "odds_win"])

    svc = Services(
        wh=wh, cfg=cfg, clock=clock,
        state=OperatingState(Mode.SHADOW, ModelProvenance.REAL),
        source_factory=_Src,
        banei=ModelBundle(manifest=manifest, models={"a": _Scorer(1.0)},
                          feature_config=feature_config(BANEI_CONF)))

    plan = plan_day_endpoint(svc, pd.Timestamp("2026-08-25").date())
    assert plan["infer_tasks"] == 1, "ばんえいの推論タスクが積まれていません"

    out = infer_endpoint(svc, BANEI_RACE_ID)
    assert out.status == "ok", out.reason

    preds = wh.query("SELECT * FROM prediction WHERE race_date = ?",
                     [pd.Timestamp("2026-08-25").date()], allow_full_scan=True)
    assert len(preds) == 10
    assert preds["model_release"].eq("v2026.08.25-A-banei").all(), (
        "ばんえいの予測が平地のリリース ID で記録されています")
    assert preds["p_win"].sum() == pytest.approx(1.0)

    snap = wh.query("SELECT * FROM feature_snapshot WHERE as_of_date = ?",
                    [pd.Timestamp("2026-08-25").date()], allow_full_scan=True)
    stored = json.loads(snap["features"].iloc[0])
    for c in ("b_weight_carried", "b_load_ratio", "b_moisture"):
        assert stored.get(c) is not None, f"{c} がスナップショットに残っていません"
    assert "distance" not in stored, "ばんえいで落とすべき列が契約に残っています"


# ------------------------------------------------------- RF ガード（系統ごと）
def test_rf_guards_are_evaluated_per_family(wh, clock):
    """ばんえいの病的な数値が平地に薄められて見逃されないこと。

    ガードは「良すぎる結果を信じて資金を投じる」ことを止めるためのもの。
    合算した1つの数字に閾値を当てると、どちらの異常も検出できない。
    ばんえいはほぼ 10 頭固定、平地は 5-16 頭で、そもそも水準が違う。
    """
    from narops.clock import jst_datetime, to_utc
    from narops.db.merge import merge_final
    from narops.service import _recent_performance_by_family
    from tests.conftest import make_results

    flat = make_results(n_races=20, start_day=18, seed=4)
    banei = _banei_history(n_races=20, start_day=18, seed=6)
    banei["race_id"] = "03" + banei["race_id"].str[2:]
    merge_final(wh, pd.concat([flat, banei], ignore_index=True), clock=clock)

    # 予測は「1番人気を必ず的中」させた極端な値をばんえいにだけ入れる
    rows = []
    for _, r in pd.concat([flat, banei], ignore_index=True).iterrows():
        banei_row = int(r["baba_code"]) == 3
        hit = int(r["finish_pos"]) == 1
        rows.append({
            "race_id": r["race_id"], "horse_no": int(r["horse_no"]),
            "race_date": r["race_date"],
            "p_win": (0.9 if hit else 0.01) if banei_row else 0.1,
            "model_release": "banei" if banei_row else "flat",
            "track_used": "A", "is_shadow": False,
            "computed_at": clock.now(),
        })
    wh.insert_frame("prediction", pd.DataFrame(rows))

    perf = _recent_performance_by_family(
        wh, pd.Timestamp("2026-08-01").date())
    assert set(perf) >= {"all", "flat", "banei"}
    assert perf["banei"]["top1"] > 0.9, "ばんえい側の異常が見えていません"
    assert perf["flat"]["top1"] < 0.6
    assert perf["flat"]["top1"] < perf["all"]["top1"] < perf["banei"]["top1"], (
        "合算は両者の中間になるはず（＝閾値を当てても異常が埋もれる）")


# --------------------------------------------------- 系統ごとの学習設定の解決
def test_feature_config_is_resolved_per_family():
    """`feature_config()` を引数なしで呼ぶとプロセス全体の既定（平地）が効く。

    運用側は1プロセスで両系統を扱うので、系統を明示して読まないと
    manifest に別競技の lookback_days が載り、推論が引く履歴の窓もずれる。
    """
    from narops.shared import feature_config_for, learning_conf_dir

    assert learning_conf_dir("flat").name == "conf"
    assert learning_conf_dir("banei").name == "conf_banei"
    assert learning_conf_dir("banei").is_dir(), "conf_banei が配置されていません"

    assert feature_config_for("flat").variant == "flat"
    assert feature_config_for("banei").variant == "banei"
    assert sorted(feature_config_for("banei").include_baba_codes) == [1, 2, 3, 4]
    assert sorted(feature_config_for("flat").exclude_baba_codes) == [1, 2, 3, 4]


def test_unknown_family_conf_is_rejected():
    from narops.shared import learning_conf_dir

    with pytest.raises(ValueError, match="未知のモデル系統"):
        learning_conf_dir("jra")


# ---------------------------------------------- GCS 側にも同じ検査があること
class _FakeBlob:
    def __init__(self, store: dict, name: str) -> None:
        self.store, self.name = store, name

    def exists(self) -> bool:
        return self.name in self.store

    def download_as_text(self) -> str:
        return self.store[self.name]

    def upload_from_string(self, data, content_type=None) -> None:
        self.store[self.name] = data

    def download_to_filename(self, target) -> None:  # pragma: no cover
        Path(target).write_text(self.store[self.name], encoding="utf-8")


class _FakeBucket:
    def __init__(self, store: dict) -> None:
        self.store = store

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self.store, name)


class _FakeGcsClient:
    def __init__(self, store: dict) -> None:
        self.store = store

    def bucket(self, name):
        return _FakeBucket(self.store)

    def list_blobs(self, bucket, prefix=""):
        return [_FakeBlob(self.store, n) for n in self.store if n.startswith(prefix)]


def _gcs_store_with(release_id: str, names: list[str]) -> dict:
    from narops.model.manifest import FeatureSpec

    spec = FeatureSpec(tuple(names), {n: "float64" for n in names},
                       {n: "nan" for n in names})
    manifest = {
        "model_id": release_id, "dataset_version": "d",
        "train_period": {"start": "1998-01-01", "end": "2026-08-24"},
        "feature_spec_hash": spec.hash(), "oos_metrics": {},
        "lookback_days": 180, "track": "A", "calibration": {},
        "model_sha256": {"clogit_beta.json": "0" * 64},
        "feature_spec": spec.to_dict(),
    }
    return {f"releases/{release_id}/manifest.json": json.dumps(manifest)}


def test_gcs_set_current_refuses_a_cross_family_release():
    """GCS 側にもローカルと同じ系統検査があること。

    片側だけ緩いと「ローカルのテストは緑、本番だけ取り違えを受け付ける」という
    一番悪い形になる。releases/ は系統をまたいで共有なので、ID の打ち間違いは
    実際に起こりうる。
    """
    from narops.errors import ArtifactIntegrityError
    from narops.gcp import GcsModelRegistry

    store = _gcs_store_with("v1-A", ["h_winrate_prior", "class_level"])   # 平地
    reg = GcsModelRegistry(bucket="nar-model-test", project="p", family="banei")
    reg._client = _FakeGcsClient(store)

    with pytest.raises(ArtifactIntegrityError, match="flat のモデル"):
        reg.set_current("v1-A", actor="t", reason="取り違え")
    assert "current_banei.json" not in store


def test_gcs_set_current_accepts_the_matching_family():
    from narops.gcp import GcsModelRegistry

    store = _gcs_store_with("v1-A-banei", ["b_weight_carried", "b_moisture"])
    reg = GcsModelRegistry(bucket="nar-model-test", project="p", family="banei")
    reg._client = _FakeGcsClient(store)

    reg.set_current("v1-A-banei", actor="t", reason="初回導入")
    assert json.loads(store["current_banei.json"])["release_id"] == "v1-A-banei"
    assert "current.json" not in store, "平地の current まで書いています"


# --------------------------------------- 重量が未掲載のときは推論しない
def test_banei_inference_refuses_a_card_without_body_weights():
    """馬体重が未掲載の出馬表では推論しない。

    標準化器は欠測を学習時の中央値で埋める。ばんえいの重量は競技のハンデ
    そのものなので、埋めた時点で事実と違う前提の予測になり、それが paper
    モードのベット額まで流れる。実測（2026-08-30 早朝）で、開催日の早朝の
    出馬表には馬体重が1頭も載らないことを確認している。
    """
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    declared = ("b_weight_carried", "b_body_weight", "h_winrate_prior")
    frame = pd.DataFrame({
        "b_weight_carried": [600.0] * 10,
        "b_body_weight": [np.nan] * 10,
        "h_winrate_prior": [0.1] * 10,
    })
    # DataNotYetPublished は InsufficientData のサブクラス。infer_endpoint が
    # これをリトライ対象として区別できるよう、具体的な型で固定する
    # （2026-09-21、032026092103 でリトライ無しのまま推論が止まっていた実例）。
    from narops.errors import DataNotYetPublished

    with pytest.raises(DataNotYetPublished, match="b_body_weight"):
        assert_serving_inputs(frame, cfg, declared)


def test_banei_inference_refuses_when_only_some_runners_lack_the_weight():
    """一部だけ欠測が最も危ない。レース内の相対量が直接歪む。"""
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    declared = ("b_weight_carried",)
    frame = pd.DataFrame({"b_weight_carried": [600.0] * 8 + [np.nan] * 2})
    with pytest.raises(InsufficientData, match="2/10"):
        assert_serving_inputs(frame, cfg, declared)


def test_complete_banei_card_passes_the_serving_check():
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    frame = pd.DataFrame({"b_weight_carried": [600.0] * 10,
                          "b_body_weight": [950.0] * 10})
    assert_serving_inputs(frame, cfg, ("b_weight_carried", "b_body_weight"))


def test_serving_check_ignores_columns_the_model_does_not_use():
    """モデルが使っていない列の欠測で止める理由は無い。"""
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    frame = pd.DataFrame({"b_weight_carried": [600.0] * 10,
                          "b_body_weight": [np.nan] * 10})
    assert_serving_inputs(frame, cfg, ("b_weight_carried",))


def test_serving_check_does_not_apply_to_flat():
    """平地の欠測は従来どおり中央値で埋める（履歴の薄い馬の成績で、妥当な事前）。"""
    from narops.features import assert_serving_inputs

    frame = pd.DataFrame({"b_weight_carried": [np.nan] * 8})
    assert_serving_inputs(frame, feature_config(), ("b_weight_carried",))


# ------------------------------------------- 減量印つき負担重量のパース
@pytest.mark.parametrize("cell,expected", [
    ("590 2-3-1-17", "590"),
    ("☆ 580 1-4-1-2", "☆580"),      # 減量騎手。印と数値の間に空白が入る
    ("☆580 1-4-1-2", "☆580"),
    ("▲ 600 0-0-0-1", "▲600"),
    ("", ""),
    ("1-4-1-2", "1"),               # 数値が先に来ない異常系は拾えるだけ拾う
])
def test_weight_carried_keeps_the_allowance_mark(cell, expected):
    """`re.match` は先頭からしか見ないので、減量印つきの負担重量を丸ごと落としていた。

    実測（2026-08-29 帯広 R1）で 9 頭中 2 頭が該当。月次ファイル側は `☆580` を
    持っているので、学習と推論で入力が食い違っていた。ばんえいでは負担重量が
    競技のハンデそのもので、レース内の重量差（b_weight_rel）は落ちた馬だけでなく
    **同じレースの全馬**の値を歪める。
    """
    from narops.deba_table import _weight_carried

    assert _weight_carried(cell) == expected


def test_day_page_weight_matches_the_monthly_file_format():
    """当日ページと月次ファイルが同じ表記であること（train-serving skew の防止）。

    silver の `weight_carried` は `☆760` の形（印＋数値、空白なし）。
    """
    from narops.deba_table import _weight_carried

    assert _weight_carried("☆ 760 0-0-0-1") == "☆760"
    # 学習側のパーサが同じ値を同じに読むこと
    from nar.features.banei import build as banei_build

    e = pd.DataFrame({"race_id": ["R"] * 2, "horse_no": [1, 2],
                      "weight_carried": ["☆760", _weight_carried("☆ 760 0-0-0-1")],
                      "weight_kg": [1000.0, 1000.0], "weight_diff": ["+0", "+0"],
                      "sex": ["牡", "牡"], "age": [5, 5]})
    out = banei_build(e, pd.DataFrame([{"race_id": "R", "baba_condition": "1.8"}]))
    assert out["b_weight_carried"].tolist() == [760.0, 760.0]
    assert out["b_is_apprentice"].tolist() == [1.0, 1.0]


# ------------------------------------------- 4桁の馬体重（ばんえい特有）
@pytest.mark.parametrize("cell,expected_kg,expected_diff", [
    ("父 調教師 956 (+2)", 956.0, "+2"),
    ("父 調教師 1022 (+4)", 1022.0, "+4"),     # ばんえいは4桁が普通
    ("父 調教師 1266 (-10)", 1266.0, "-10"),
    ("父 調教師 902 (±0)", 902.0, "±0"),
])
def test_body_weight_accepts_four_digits(cell, expected_kg, expected_diff):
    """`\\d{3}` だと 1022(+4) が「022」と読まれ、22kg の馬が 610kg を曳くことになる。

    ばんえいの馬体重は 664-1266kg、実データの中央値は 997kg なので出走馬の
    およそ半分が4桁。月次ファイル側は正しい4桁を持つため、推論側だけが壊れる
    形の train-serving skew だった（実測 2026-08-29 帯広 R1 で 9 頭中 3 頭）。
    """
    from narops.deba_table import _WEIGHT_RE

    m = _WEIGHT_RE.search(cell)
    assert m is not None
    assert float(m.group(1)) == expected_kg
    assert m.group(2) == expected_diff


def test_serving_check_rejects_an_implausible_body_weight():
    """欠測ではなく『値はあるが明らかにおかしい』を捕まえること。

    解析ミスは欠測ではなく異常値として出ることがあり、そちらのほうが静かで危ない。
    22kg の馬が 610kg を曳く予測は、欠測チェックでは絶対に止まらない。
    """
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    declared = ("b_weight_carried", "b_body_weight")
    frame = pd.DataFrame({
        "b_weight_carried": [600.0] * 9,
        "b_body_weight": [956.0, 880.0, 982.0, 943.0, 22.0, 2.0, 913.0, 11.0, 931.0],
    })
    from narops.errors import DataNotYetPublished

    with pytest.raises(InsufficientData, match="妥当域") as exc_info:
        assert_serving_inputs(frame, cfg, declared)
    assert not isinstance(exc_info.value, DataNotYetPublished), (
        "解析ミスによる異常値はリトライしても直らないので、DataNotYetPublished"
        "にしてはいけません")


def test_serving_check_accepts_real_banei_body_weights():
    from narops.features import assert_serving_inputs

    cfg = feature_config(BANEI_CONF)
    frame = pd.DataFrame({
        "b_weight_carried": [590.0, 610.0, 1000.0],
        "b_body_weight": [664.0, 1022.0, 1266.0],   # 実データの最小・4桁・最大
    })
    assert_serving_inputs(frame, cfg, ("b_weight_carried", "b_body_weight"))
