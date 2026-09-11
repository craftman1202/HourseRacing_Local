"""配布する最終モデルを OOS 期間で評価する。

`nar learn --oos` は walk-forward の副産物としての評価で、`nar evaluate` は
条件付きロジット単体の軽量版。どちらも「実際に出荷するモデル」を測っていない。
manifest に載せる oos_metrics は、配布物そのものの数字でなければ意味がない。

OOS は施錠されている（CV-07）。開封は台帳に記録し、1回で終える。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CVConfig
from ..eval import guards, metrics
from ..models import baselines
from ..train.pipeline import apply_stats, trainable

log = logging.getLogger(__name__)


@dataclass
class OosResult:
    metrics: dict[str, dict[str, float]]
    ensemble: dict[str, float]
    tripwires: list
    n_rows: int
    n_races: int
    period: tuple[str, str]
    # 行単位の予測。集計指標だけだとロック区間そのものの較正（リライアビリティ図・
    # 人気帯別の較正誤差）が描けない。OOS の開封は1回きりなので、そのとき出た
    # 予測は必ず残す。走らせ直すには再開封が要る（＝開封回数の意味が薄れる）。
    predictions: pd.DataFrame | None = None


def load_release(final_dir: str | Path) -> dict:
    """配布物を読む。学習側の実装を通さず、運用側と同じ読み方をする。

    ここで学習側のモデルオブジェクトを再構築すると、配布物が壊れていても
    気付けない。運用側が実際に読むのと同じ経路で読む。
    """
    import sys

    root = Path(final_dir).resolve()
    ops_src = root.parents[2] / "operation" / "src"
    if str(ops_src) not in sys.path:
        sys.path.insert(0, str(ops_src))
    from narops.runtime import Standardizer, LgbmScorer, LinearScorer, OnnxScorer

    names = json.loads((root / "feature_names.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "standardizer.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "final_meta.json").read_text(encoding="utf-8"))

    models: dict = {}
    beta_file = root / "clogit_beta.json"
    if beta_file.exists():
        beta = json.loads(beta_file.read_text(encoding="utf-8"))["beta"]
        models["clogit"] = LinearScorer(
            "clogit", np.asarray([beta[n] for n in names], dtype=float), names)
    lgbm_file = root / "lgbm_rank.txt"
    if lgbm_file.exists():
        import lightgbm as lgb

        models["lgbm"] = LgbmScorer("lgbm", lgb.Booster(model_file=str(lgbm_file)),
                                    names)
    onnx_file = root / "tabm.onnx"
    if onnx_file.exists():
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_file),
                                    providers=["CPUExecutionProvider"])
        models["tabm"] = OnnxScorer("tabm", sess, names, sess.get_inputs()[0].name)

    return {"models": models, "features": names, "meta": meta,
            "standardizer": Standardizer(stats)}


def evaluate(feat: pd.DataFrame, ccfg: CVConfig, final_dir: str | Path,
             weights: dict[str, float] | None = None,
             cv_nll: float | None = None) -> OosResult:
    release = load_release(final_dir)
    names = release["features"]
    temps = release["meta"].get("temperatures", {})

    dates = pd.to_datetime(feat["race_date"])
    lo = pd.Timestamp(ccfg.oos[0])
    hi = pd.Timestamp(ccfg.oos[1]) if ccfg.oos[1] else dates.max()
    oos = trainable(feat[(dates >= lo) & (dates <= hi)])
    if oos.empty:
        raise ValueError(f"OOS 期間（{lo.date()} 〜 {hi.date()}）に評価対象がありません")

    # 標準化は配布物の統計量で行う。ここで再計算すると、本番と違う入力を測る
    scored = release["standardizer"].apply(oos, names).sort_values(
        ["race_id", "horse_no"]).reset_index(drop=True)
    rid = scored["race_id"].to_numpy()
    y = scored["is_win"].to_numpy()
    pos = scored["finish_pos"].to_numpy()

    per_model: dict[str, np.ndarray] = {}
    out: dict[str, dict[str, float]] = {}
    for name, model in release["models"].items():
        raw = np.asarray(model.score(scored), dtype=float).ravel()
        p = metrics.race_softmax(raw, rid)
        t = float(temps.get(name, 1.0)) or 1.0
        p = metrics.normalize_within_race(np.power(np.clip(p, 1e-12, 1), 1.0 / t), rid)
        per_model[name] = p
        out[name] = metrics.summary(p, y, pos, rid)

    # レース内一様分布。頭数の逆数
    sizes = pd.Series(rid).map(pd.Series(rid).value_counts()).to_numpy(dtype=float)
    out["baseline_uniform"] = metrics.summary(1.0 / sizes, y, pos, rid)

    if "odds_win" in scored.columns and scored["odds_win"].notna().any():
        out["baseline_market"] = metrics.summary(
            baselines.market(scored), y, pos, rid)

    w = {k: v for k, v in (weights or {}).items() if k in per_model and v > 0}
    if w:
        total = sum(w.values())
        blend = sum(per_model[k] * (v / total) for k, v in w.items())
        ens = metrics.normalize_within_race(blend, rid)
    else:
        ens = np.mean(list(per_model.values()), axis=0)
    out["ensemble"] = metrics.summary(ens, y, pos, rid)

    tw = guards.check(oos_top1=out["ensemble"]["top1"],
                      oos_nll=out["ensemble"]["race_nll"], cv_nll=cv_nll)

    # oof_predictions.parquet と同じ形。較正ノートブックが両方を同じコードで
    # 読めるよう、列名（race_id / horse_no / is_win / finish_pos / モデル名）を揃える。
    preds = pd.DataFrame({
        "race_id": rid, "horse_no": scored["horse_no"].to_numpy(),
        "race_date": scored["race_date"].to_numpy(),
        "is_win": y, "finish_pos": pos,
    })
    for name, p in per_model.items():
        preds[name] = p
    preds["ensemble"] = ens
    if "odds_win" in scored.columns:
        preds["odds_win"] = scored["odds_win"].to_numpy()

    return OosResult(out, {k: float(v) for k, v in (weights or {}).items()},
                     tw, len(scored), int(pd.Series(rid).nunique()),
                     (str(lo.date()), str(hi.date())), predictions=preds)
