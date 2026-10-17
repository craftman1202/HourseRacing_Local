"""全モデルアンサンブル（TabM / LightGBM / CatBoost、pop 系統）の本番リリースを作る（2026-10-17）。

  - 各モデルは all_models.py の HPO 最良設定（best_<model>.json）で、データ末尾までの
    8 年窓（末尾 180 日で較正・早期停止）で学習する。手順は walk-forward / OOS と同じ関数
  - TabM・CatBoost は ONNX、LightGBM は Booster テキスト。モデルごとの特徴量は model_features.json
  - アンサンブル重みは fold1〜5 の OOF から推定済みのもの（ensemble.json の all_fold_weights、
    重み 0 の clogit・bayes は配らない）
  - 払戻予測モデル（券種ごと、scripts/payout_ev.py と同じ学習）と人気の勝率表、買い方の規則を
    payout_meta.json に
出力: operation/data/nar-model/releases/<release_id>/
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_models as A  # noqa: E402

ROOT = A.ROOT
REL_ROOT = ROOT.parent / "operation/data/nar-model/releases"
sys.path.insert(0, str(ROOT.parent / "operation/src"))
MODELS = ("tabm", "lgbm", "catboost")
EV_RULES = [  # fold2〜5 で選んだもの（artifacts/payout_ev/selected.csv・nested.csv）
    {"bet": "trio", "t": 2.0, "mode": "top1", "note": "fold2〜5 ROI 1.16・nested 1.06"},
    {"bet": "quinella", "t": 2.0, "mode": "top1", "note": "fold2〜5 ROI 1.15・nested 1.06"},
    {"bet": "wide", "t": 2.0, "mode": "top1", "note": "fold2〜5 ROI 1.06・nested 1.00"},
    {"bet": "exacta", "t": 2.0, "mode": "top1", "note": "fold2〜5 ROI 1.01・nested 0.94"},
    {"bet": "trifecta", "t": 2.0, "mode": "all", "note": "fold2〜5 ROI 0.96・nested 0.92"},
]
ODDS_FREE_RULES = [  # artifacts/strategy_ens/selected.csv（オッズ不要、人気との合成 λ）
    {"structure": "単勝 1位", "lam": 0.5, "min_p": 0.6, "note": "fold2〜5 ROI 0.94・nested 0.93"},
    {"structure": "複勝 1位", "lam": 0.5, "min_p": 0.6, "note": "fold2〜5 ROI 0.98・nested 0.98"},
    {"structure": "枠単 1点", "lam": 0.0, "min_p": 0.6, "note": "fold2〜5 ROI 1.03・nested 0.93"},
    {"structure": "枠連 1点", "lam": 0.0, "min_p": 0.6, "note": "fold2〜5 ROI 0.91・nested 0.89"},
]


def main(release_id: str) -> int:
    A.KEEP_MODEL = True
    df, _ = A.load("pop")
    rank_list, _ = A.ranking("pop")
    data_end = df["race_date"].max()
    valid_start = (data_end + pd.Timedelta(days=A.EMBARGO_DAYS + 1)).strftime("%Y-%m-%d")
    out = REL_ROOT / release_id
    if out.exists():
        raise SystemExit(f"{out} は既にあります")
    tmp = ROOT / "artifacts/all_models/release_build" / release_id
    tmp.mkdir(parents=True, exist_ok=True)

    model_features, temps, infos = {}, {}, {}
    stats_all = None
    for m in MODELS:
        params = A.best_params("pop", m)
        n = int(params["_n_features"])
        cols = rank_list[:n]
        fit_raw, cal_raw = A.split_train(df, valid_start, params.get("_train_years", 8))
        stats = A.fit_stats(fit_raw, rank_list)          # 全列（モデル間で同じ窓なので共通）
        stats_all = stats_all or stats
        fit_df, cal_df = (A.standardize(x, rank_list, stats) for x in (fit_raw, cal_raw))
        (p_cal,), info = A.fit_predict(m, A.model_params(params), fit_df, cal_df, [cal_df], cols)
        (_,), temp = A.calibrated(p_cal, cal_df, [p_cal], [cal_df])
        model_features[m] = cols
        temps[m] = temp
        model = info.pop("model")
        infos[m] = {**{k: v for k, v in info.items()}, "n_fit": len(fit_df), "n_cal": len(cal_df),
                    "train_end": str(fit_raw["race_date"].max().date()), "temperature": temp}
        if m == "lgbm":
            model.save_model(str(tmp / "lgbm_rank.txt"), num_iteration=model.best_iteration)
        elif m == "catboost":
            model.model.save_model(str(tmp / "catboost.onnx"), format="onnx")
            check_catboost_onnx(model, tmp / "catboost.onnx", cal_df, cols)
        elif m == "tabm":
            from nar.models.base import to_batch
            from narops.publish import export_tabm_onnx, verify_onnx

            # GPU で学習したので CPU に移してから書き出す（ONNX は CPU で作る。標準化の統計量も CPU）
            model.cfg.device = "cpu"
            model._fitted().to("cpu")
            export_tabm_onnx(model, len(cols), tmp / "tabm.onnx")
            sample = cal_df.head(2000)
            b = to_batch(sample, cols)
            sizes = sample.groupby("race_id", sort=False).size().tolist()
            infos[m]["onnx_check"] = verify_onnx(model, tmp / "tabm.onnx",
                                                 sample[cols].to_numpy(np.float32), sizes)
        print(m, infos[m], flush=True)

    # 払戻予測モデル（券種ごと）
    import payout_ev as PE

    pay_meta = {"models": {}}
    st = PE.starters_with_pop("2100-01-01")
    end = data_end
    for bt in ("win", "place", "quinella", "exacta", "wide", "trio", "trifecta"):
        pays = PE.payouts(bt)
        tr = PE.training_rows(bt, st, pays, end - pd.DateOffset(years=PE.TRAIN_YEARS), end)
        booster, smear = PE.fit_payout_model(tr, bt)
        booster.save_model(str(tmp / f"payout_{bt}.txt"))
        pay_meta["models"][bt] = {"file": f"payout_{bt}.txt", "smear": smear, "n_train": len(tr)}
        print("payout", bt, len(tr), round(smear, 3), flush=True)
    import roi_search as R
    pay_meta["pop_table"] = R.pop_table_before(R.starters(), "2100-01-01")
    pay_meta["ev_rules"] = EV_RULES
    pay_meta["odds_free_rules"] = ODDS_FREE_RULES
    (tmp / "payout_meta.json").write_text(json.dumps(pay_meta, ensure_ascii=False, indent=1))
    (tmp / "model_features.json").write_text(json.dumps(model_features, ensure_ascii=False, indent=1))

    ens = json.loads((A.OUT / "pop/ensemble.json").read_text())["all_fold_weights"]
    w = {m: ens[m] for m in MODELS}
    tot = sum(w.values())
    w = {m: v / tot for m, v in w.items()}
    oos = {}
    oos_file = A.OUT / "pop/oos_summary.json"
    if oos_file.exists():
        oos = json.loads(oos_file.read_text())

    from narops.publish import build_release

    res = build_release(
        release_id=release_id, out_dir=REL_ROOT, feature_names=list(rank_list),
        dataset_version="all_models_pop_" + str(data_end.date()),
        train_period={"start": str((pd.Timestamp(valid_start) - pd.Timedelta(days=A.EMBARGO_DAYS)
                                    - pd.DateOffset(years=8)).date()), "end": str(data_end.date())},
        oos_metrics=oos.get("ensemble", {}), ensemble_weights={"lgbm": w["lgbm"], "tabm": w["tabm"]},
        lookback_days=180, temperature=1.0, test_results=gate_results(),
        lgbm_booster_path=tmp / "lgbm_rank.txt", tabm_onnx_path=tmp / "tabm.onnx",
        standardizer=stats_all, purpose="production", oos_evaluated_on=str(date.today()),
        model_temperatures={"lgbm": temps["lgbm"], "tabm": temps["tabm"]})
    # CatBoost と払戻モデル・特徴量の割り当てを足し、manifest を作り直す
    for f in ["catboost.onnx", "model_features.json", "payout_meta.json",
              *[f"payout_{bt}.txt" for bt in pay_meta["models"]]]:
        shutil.copy2(tmp / f, res.path / f)
    from narops.model.manifest import Manifest, sha256_file

    man = res.manifest
    man.ensemble_weights = w
    man.model_temperatures = temps
    man.model_sha256 = {p.name: sha256_file(p) for p in sorted(res.path.iterdir()) if p.name != "manifest.json"}
    man.write(res.path / "manifest.json")
    (res.path / "build_info.json").write_text(json.dumps(
        {"models": infos, "valid_start": valid_start, "weights": w}, ensure_ascii=False, indent=1, default=str))
    print("release", res.path, w, temps)
    return 0


def gate_results() -> dict:
    """publish ゲート（RL-01）の証跡。`nar gate-report` と同じ手順（Blocker テスト + 実データ OOS ガード）。"""
    from nar import gate

    payload = gate.build(ROOT / "tests", A.OUT / "pop", run_tests=True)
    if not payload["can_publish"]:
        raise SystemExit(f"publish ゲート不可: {payload}")
    return payload["results"]


def check_catboost_onnx(model, path: Path, df: pd.DataFrame, cols: list[str]) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = df[cols].head(3000).to_numpy(np.float32)
    o = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0]).reshape(len(x), -1).mean(1)
    from catboost import Pool
    ref = model.model.predict(Pool(x))
    err = float(np.max(np.abs(o - ref)))
    if err > 1e-4:
        raise SystemExit(f"CatBoost ONNX の出力が一致しません（最大誤差 {err}）")
    print("catboost onnx ok, max err", err)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
