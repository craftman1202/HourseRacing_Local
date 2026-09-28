"""複勝（3着内／少頭数は2着内）予測専用モデル。

単勝の評価用配布モデル（`artifacts/final_eval_new`、学習 1998-01-01〜2023-08-04）と
**同じ特徴量（23列・同じ標準化統計量）・同じ3アルゴリズム・同じ既定ハイパーパラメータ**で、
目的変数だけを「複勝圏に入ったか」に替えて学習する。

  clogit : 条件付きロジット（l2=1e-3）。複勝圏の k 頭すべてを「選ばれた」として尤度に入れる
  lgbm   : LambdaRank（300 round、既定パラメータ）。ラベルは複勝圏 1 / その他 0
  tabm   : k=4 / hidden=128 / 2層 / 3 epoch / batch_races=256（final_eval_new と同じ）。
           損失はレース内 softmax 交差エントロピーを複勝圏の k 頭について合計したもの

どのモデルもスコアはレース内の「強さ」なので、確率への変換は
  強さ → レース内 softmax → 温度 T → Harville 式で「上位 k 着に入る確率」
で行う。レース内で Σ P(複勝) = k が厳密に成り立つ。温度とアンサンブル重みは
学習期間末尾 90 日（単勝モデルの較正と同じ保留区間）の二値対数損失で決める。

複勝圏の頭数 k: 出走 8 頭以上 = 3、5〜7 頭 = 2、4 頭以下は発売なし（学習・評価から除外）。
実払戻との一致率は 99.3%（2026-09-27 に payout.parquet と突合）。

系統は `PLACE_FAMILY` 環境変数で切り替える（既定は平地 flat）。ばんえいは
`PLACE_FAMILY=banei NAR_CONF_DIR=conf_banei` の両方をセットして呼ぶこと —
前者はこのスクリプトのパス選択、後者は `nar.config`（履歴集計・embargo 導出）の
切り替えで、意味が別なので両方要る。

使い方（learning/ で）:
    PYTHONPATH=src .venv/bin/python scripts/place_model.py fit
    PYTHONPATH=src .venv/bin/python scripts/place_model.py score-oos --unlock-oos --reason "..."

    PLACE_FAMILY=banei NAR_CONF_DIR=conf_banei PYTHONPATH=src .venv/bin/python \
        scripts/place_model.py fit
    PLACE_FAMILY=banei NAR_CONF_DIR=conf_banei PYTHONPATH=src .venv/bin/python \
        scripts/place_model.py score-oos --unlock-oos --reason "..."

`fit` は OOS に一切触れない。`score-oos` は OOS を開封して台帳に記録する（1回だけ、
系統ごとに別のログ — 平地は artifacts/oos_access.log、ばんえいは
artifacts/banei/oos_access.log）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.config import cv_config  # noqa: E402
from nar.eval.splits import OOSGuard  # noqa: E402
from nar.models.base import to_batch  # noqa: E402
from nar.models.clogit import ConditionalLogit  # noqa: E402
from nar.models.lgbm import LgbmRanker  # noqa: E402
from nar.models.tabm import TabM, TabMConfig  # noqa: E402
from nar.train.pipeline import apply_stats, fit_stats, trainable  # noqa: E402

log = logging.getLogger("place_model")

FAMILY = os.environ.get("PLACE_FAMILY", "flat")
if FAMILY not in ("flat", "banei"):
    raise ValueError(f"PLACE_FAMILY は flat/banei のいずれか（受領: {FAMILY!r}）")
if FAMILY == "banei" and os.environ.get("NAR_CONF_DIR") != "conf_banei":
    raise RuntimeError("PLACE_FAMILY=banei には NAR_CONF_DIR=conf_banei も必要です"
                      "（embargo・履歴集計の切り替えが別モジュールにあるため）。")

_PATHS = {
    "flat": dict(gold="data_real/gold/features_noodds/features.parquet",
                win_eval="artifacts/final_eval_new", out="artifacts/place",
                oos_log="artifacts/oos_access.log",
                win_oos_pred="artifacts/oos_predictions.parquet",
                default_win_release="v2026.09.17-C"),
    "banei": dict(gold="data_real/gold/features_noodds_banei/features.parquet",
                 win_eval="artifacts/final_banei_eval_new", out="artifacts/place_banei",
                 oos_log="artifacts/banei/oos_access.log",
                 win_oos_pred="artifacts/banei/oos_predictions.parquet",
                 default_win_release="v2026.09.17-B-banei"),
}[FAMILY]

GOLD = ROOT / _PATHS["gold"]
WIN_EVAL_DIR = ROOT / _PATHS["win_eval"]   # 単勝の評価用モデル（OOS 予測の出所）
OUT = ROOT / _PATHS["out"]
OOS_LOG = ROOT / _PATHS["oos_log"]
WIN_OOS_PRED = ROOT / _PATHS["win_oos_pred"]
DEFAULT_WIN_RELEASE = _PATHS["default_win_release"]
MODELS = ("clogit", "lgbm", "tabm")
HOLDOUT_DAYS = 90
EPS = 1e-12
# final_eval_new / final_banei_eval_new の学習条件（final_meta.json / lgbm_rank.txt の
# ヘッダ / tabm.onnx のパラメータ数から復元）。両系統とも同じ fit-final 経路・同じ
# 当時のバグ（--tabm-batch が効かず既定値 256 で学習）の影響下で作られているので同一値を使う。
# HPO 値は使っていない（当時の fit-final は既定値で学習）。
CLOGIT_L2 = 1e-3
TABM_CFG = dict(k=4, hidden=128, n_layers=2, epochs=3, batch_races=256, seed=0)


# --------------------------------------------------------------------------- ラベル
# 頭数→複勝枠・Harville 変換・合成は運用側と同じ実装を使う（SK-02）
from nar.eval.place import (  # noqa: E402
    harville_topk as harville_place, place_slots, race_log_softmax,
)


def add_place_label(df: pd.DataFrame) -> pd.DataFrame:
    n = df.groupby("race_id")["race_id"].transform("size").to_numpy()
    out = df.assign(n_runners=n, k_place=place_slots(n))
    out = out[out["k_place"] > 0].copy()
    out["is_place"] = (out["finish_pos"] <= out["k_place"]).astype(int)
    return out.reset_index(drop=True)


def to_place(log_strength: np.ndarray, race_ids: np.ndarray, k: np.ndarray,
             t: float = 1.0) -> np.ndarray:
    return harville_place(np.exp(race_log_softmax(log_strength / t, race_ids)), race_ids, k)


def binary_logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def place_metrics(p: np.ndarray, y: np.ndarray, race_ids: np.ndarray,
                  k: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({"p": p, "y": y, "r": race_ids})
    # レース内上位 k 頭に予測した馬の的中率（= 複勝圏の当て率）
    rank = df.groupby("r")["p"].rank(ascending=False, method="first").to_numpy()
    top_k_prec = float(y[rank <= k].mean())
    top1_hit = float(y[rank == 1].mean())
    bins = pd.qcut(p, 10, duplicates="drop")
    cal = df.groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ece = float((abs(cal["p"] - cal["y"]) * cal["n"]).sum() / cal["n"].sum())
    return {"logloss": binary_logloss(p, y), "brier": float(np.mean((p - y) ** 2)),
            "auc": float(roc_auc_score(y, p)), "top1_place_rate": top1_hit,
            "topk_precision": top_k_prec, "ece10": ece, "n_rows": int(len(y)),
            "n_races": int(df["r"].nunique())}


# -------------------------------------------------------------------- データ
def load_frame() -> tuple[pd.DataFrame, list[str], dict]:
    names = json.loads((WIN_EVAL_DIR / "feature_names.json").read_text(encoding="utf-8"))
    stats = json.loads((WIN_EVAL_DIR / "standardizer.json").read_text(encoding="utf-8"))
    cols = ["race_id", "horse_no", "race_date", "finish_pos", "is_win", *names]
    feat = pd.read_parquet(GOLD, columns=cols)
    gold_hash = (GOLD.parent / "content_hash.txt").read_text().strip()
    meta = json.loads((WIN_EVAL_DIR / "final_meta.json").read_text(encoding="utf-8"))
    if gold_hash != meta["gold_content_hash"]:
        raise RuntimeError("gold の内容が単勝評価モデルの学習時と違います")
    return feat, names, stats


# ----------------------------------------------------------------------- 学習
def train_models(fit_df: pd.DataFrame, cal_df: pd.DataFrame, names: list[str], out: Path,
                 export_onnx: bool) -> tuple[dict, dict, dict]:
    """3モデルを学習し、較正区間でのレース内「強さ」（対数スケールの生スコア）を返す。

    export_onnx=True（出荷用）は運用側と同じ読み方ができる形で書き出す:
    `place_clogit_beta.json` / `place_lgbm_rank.txt` / `place_tabm.onnx`。
    このとき TabM の較正区間スコアは運用側 `OnnxScorer` と同じ「member スコアの平均」を使う
    （評価用は member 確率の平均）。温度はスコアの取り方ごとに測らないと合わない。
    """
    import lightgbm as lgb

    from nar.models.lgbm import group_sizes

    out.mkdir(parents=True, exist_ok=True)
    prefix = "place_" if export_onnx else ""
    cal_logs: dict[str, np.ndarray] = {}
    timings = {}

    t0 = time.time()
    cl = ConditionalLogit(l2=CLOGIT_L2).fit(to_batch(fit_df, names, label_col="is_place"))
    (out / f"{prefix}clogit_beta.json").write_text(
        json.dumps({"beta": cl.coefficients()}, indent=2))
    cal_logs["clogit"] = cal_df[names].to_numpy(dtype=float) @ cl.beta
    timings["clogit"] = time.time() - t0
    log.info("clogit 完了 %.0fs", timings["clogit"])

    t0 = time.time()
    ranker = LgbmRanker()           # 単勝と同じ既定パラメータ。ラベル段階だけ 2 値に
    lgbm_params = {**ranker.params, "label_gain": [0.0, 1.0]}
    ds = lgb.Dataset(fit_df[names], label=fit_df["is_place"].to_numpy(),
                     group=group_sizes(fit_df), feature_name=names, free_raw_data=False)
    booster = lgb.train(lgbm_params, ds, num_boost_round=ranker.num_boost_round)
    booster.save_model(str(out / f"{prefix}lgbm_rank.txt"))
    cal_logs["lgbm"] = booster.predict(cal_df[names])
    timings["lgbm"] = time.time() - t0
    log.info("lgbm 完了 %.0fs", timings["lgbm"])

    t0 = time.time()
    import torch

    tm = TabM(TabMConfig(**TABM_CFG)).fit(to_batch(fit_df, names, label_col="is_place"),
                                          to_batch(cal_df, names, label_col="is_place"))
    torch.save({"state": tm.net.state_dict(), "mu": tm._mu, "sd": tm._sd,
                "cfg": TABM_CFG, "features": names}, out / "tabm.pt")
    if export_onnx:
        sys.path.insert(0, str(ROOT.parent / "operation" / "src"))
        import onnxruntime as ort
        from narops.publish import export_tabm_onnx, verify_onnx

        path = out / "place_tabm.onnx"
        export_tabm_onnx(tm, len(names), path)
        head = cal_df.head(4096)
        check = verify_onnx(tm, path, head[names].to_numpy(dtype=np.float32),
                            head.groupby("race_id", sort=False).size().tolist())
        if not check["passed"]:
            raise RuntimeError(f"place_tabm の ONNX 検証に失敗: {check}")
        log.info("place_tabm ONNX 検証: %s", check)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {sess.get_inputs()[0].name:
                              cal_df[names].to_numpy(dtype=np.float32)})[0]
        cal_logs["tabm"] = np.asarray(got).reshape(len(cal_df), -1).mean(axis=1)
    else:
        b = to_batch(cal_df, names, label_col="is_place")
        cal_logs["tabm"] = np.log(np.clip(
            b.flat_predictions(tm.predict_proba(b), len(cal_df)), EPS, 1))
    timings["tabm"] = time.time() - t0
    log.info("tabm 完了 %.0fs", timings["tabm"])
    params = {"clogit_l2": CLOGIT_L2, "lgbm": lgbm_params,
              "lgbm_rounds": ranker.num_boost_round, "tabm": TABM_CFG}
    return cal_logs, params, timings


def calibrate(cal_logs: dict, cal_df: pd.DataFrame):
    """温度（モデルごと）→ アンサンブル重み＋全体温度。いずれも較正区間の二値対数損失。"""
    rid_c, k_c, y_c = (cal_df["race_id"].to_numpy(), cal_df["k_place"].to_numpy(),
                       cal_df["is_place"].to_numpy())
    temps, cal_report = {}, {}
    for m in MODELS:
        res = minimize_scalar(lambda lt: binary_logloss(
            to_place(cal_logs[m], rid_c, k_c, np.exp(lt)), y_c),
            bounds=(np.log(0.2), np.log(5.0)), method="bounded")
        temps[m] = float(np.exp(res.x))
        cal_report[m] = place_metrics(to_place(cal_logs[m], rid_c, k_c, temps[m]), y_c, rid_c, k_c)

    logs = np.column_stack([race_log_softmax(cal_logs[m] / temps[m], rid_c) for m in MODELS])

    def ens_loss(theta):
        w = np.abs(theta[:3]) / np.abs(theta[:3]).sum()
        return binary_logloss(to_place(logs @ w, rid_c, k_c, np.exp(theta[3])), y_c)

    res = minimize(ens_loss, np.array([1 / 3, 1 / 3, 1 / 3, 0.0]), method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-7})
    w = np.abs(res.x[:3]) / np.abs(res.x[:3]).sum()
    ens_t = float(np.exp(res.x[3]))
    cal_report["ensemble"] = place_metrics(to_place(logs @ w, rid_c, k_c, ens_t), y_c, rid_c, k_c)
    cal_report["baseline_uniform"] = place_metrics(
        k_c / cal_df["n_runners"].to_numpy(), y_c, rid_c, k_c)
    return temps, w, ens_t, cal_report


# ------------------------------------------------------------------ fit-prod
def cmd_fit_prod(args) -> int:
    """出荷用。現行の単勝リリースと同じ学習期間・特徴量・標準化統計量で学習する。

    評価は `fit`（評価用、OOS で1回測定済み）で済んでいる。単勝の出荷用モデルと同じく、
    評価後に全データで学習し直したもので、これ自体の OOS 数値は無い。
    """
    rel = ROOT.parent / "operation/data/nar-model/releases" / args.win_release
    manifest = json.loads((rel / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["feature_spec"]["names"])
    stats = json.loads((rel / "standardizer.json").read_text(encoding="utf-8"))
    cols = ["race_id", "horse_no", "race_date", "finish_pos", "is_win", *names]
    feat = pd.read_parquet(GOLD, columns=cols)
    gold_hash = (GOLD.parent / "content_hash.txt").read_text().strip()
    if gold_hash != manifest["dataset_version"]:
        raise RuntimeError("gold の内容が単勝リリースの学習時と違います")
    start = pd.Timestamp(manifest["train_period"]["start"])
    end = pd.Timestamp(manifest["train_period"]["end"])

    d = pd.to_datetime(feat["race_date"])
    window = add_place_label(trainable(feat[(d >= start) & (d <= end)]))
    wd = pd.to_datetime(window["race_date"])
    cut = wd.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    fit_raw, cal_raw = window[wd <= cut], window[wd > cut]
    log.info("出荷用 学習 %s 行 / 較正 %s 行（%s 〜 %s、較正は %s より後）",
             f"{len(fit_raw):,}", f"{len(cal_raw):,}", start.date(), end.date(), cut.date())
    fit_df = apply_stats(fit_raw, names, stats)
    cal_df = apply_stats(cal_raw, names, stats)

    out = OUT / "final_prod"
    cal_logs, params, timings = train_models(fit_df, cal_df, names, out, export_onnx=True)
    temps, w, ens_t, cal_report = calibrate(cal_logs, cal_df)
    meta_out = {
        "purpose": "production", "target": "is_place (k=3 if n>=8, 2 if 5..7)",
        "win_release": args.win_release,
        "train_period": {"start": str(start.date()), "end": str(end.date())},
        "calibration_after": str(cut.date()), "holdout_days": HOLDOUT_DAYS,
        "n_fit_rows": int(len(fit_df)), "n_calibration_rows": int(len(cal_df)),
        "feature_names": names, "params": params,
        "temperatures": temps, "ensemble_weights": dict(zip(MODELS, w.tolist())),
        "ensemble_temperature": ens_t, "calibration_metrics": cal_report,
        "oos_evaluated_by": str((OUT / "oos_place_metrics.json").relative_to(ROOT)) + "（評価用モデル）",
        "timings_sec": timings,
    }
    (out / "place_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2,
                                                    default=float), encoding="utf-8")
    print(pd.DataFrame(cal_report).T.round(4).to_string())
    print(f"温度 {temps} / 重み {dict(zip(MODELS, w.round(3)))} / 全体温度 {ens_t:.3f}")
    return 0


# ----------------------------------------------------------------------- fit
def cmd_fit(args) -> int:
    ccfg = cv_config()
    feat, names, stats = load_frame()
    meta = json.loads((WIN_EVAL_DIR / "final_meta.json").read_text(encoding="utf-8"))
    start = pd.Timestamp(meta["train_period"]["start"])
    end = pd.Timestamp(meta["train_period"]["end"])
    # 学習窓は単勝評価モデルと同一。OOS 開始 − embargo − 1 日と一致することを確かめる
    expect_end = pd.Timestamp(ccfg.oos[0]) - pd.Timedelta(days=ccfg.embargo_days + 1)
    assert end == expect_end, (end, expect_end)

    d = pd.to_datetime(feat["race_date"])
    window = add_place_label(trainable(feat[(d >= start) & (d <= end)]))
    wd = pd.to_datetime(window["race_date"])
    cut = wd.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    fit_raw, cal_raw = window[wd <= cut], window[wd > cut]
    log.info("学習 %s 行 / 較正 %s 行（%s 〜 %s、較正は %s より後）",
             f"{len(fit_raw):,}", f"{len(cal_raw):,}", start.date(), end.date(), cut.date())

    # 標準化の統計量は単勝モデルのものをそのまま使う。同じ行から取り直した値と一致するはず
    # （4頭以下のレースを落とした分だけ僅かにずれるので、一致の確認は緩めに行う）
    re_stats = fit_stats(fit_raw, names)
    drift = max(abs(re_stats[c]["mean"] - stats[c]["mean"]) / stats[c]["std"] for c in names)
    log.info("標準化統計量の差（最大 |Δmean|/sd）: %.4f", drift)
    fit_df = apply_stats(fit_raw, names, stats)
    cal_df = apply_stats(cal_raw, names, stats)

    out = OUT / "final_eval"
    cal_logs, params, timings = train_models(fit_df, cal_df, names, out, export_onnx=False)
    temps, w, ens_t, cal_report = calibrate(cal_logs, cal_df)

    meta_out = {
        "purpose": "evaluation", "target": "is_place (k=3 if n>=8, 2 if 5..7)",
        "train_period": {"start": str(start.date()), "end": str(end.date())},
        "calibration_after": str(cut.date()), "holdout_days": HOLDOUT_DAYS,
        "n_fit_rows": int(len(fit_df)), "n_calibration_rows": int(len(cal_df)),
        "feature_names": names, "standardizer_from": str(WIN_EVAL_DIR.relative_to(ROOT)),
        "standardizer_max_drift_sd": drift,
        "params": params,
        "temperatures": temps, "ensemble_weights": dict(zip(MODELS, w.tolist())),
        "ensemble_temperature": ens_t, "calibration_metrics": cal_report,
        "timings_sec": timings,
    }
    (out / "place_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2,
                                                    default=float), encoding="utf-8")
    print(pd.DataFrame(cal_report).T.round(4).to_string())
    print(f"温度 {temps} / 重み {dict(zip(MODELS, w.round(3)))} / 全体温度 {ens_t:.3f}")
    return 0


# ----------------------------------------------------------------- score OOS
def load_place_models(names: list[str]):
    out = OUT / "final_eval"
    import lightgbm as lgb
    import torch

    beta = json.loads((out / "clogit_beta.json").read_text())["beta"]
    beta = np.asarray([beta[n] for n in names])
    booster = lgb.Booster(model_file=str(out / "lgbm_rank.txt"))
    ck = torch.load(out / "tabm.pt", weights_only=False)
    tm = TabM(TabMConfig(**ck["cfg"]))
    from nar.models.tabm import _TabMNet

    tm.net = _TabMNet(len(names), tm.cfg).to(tm.cfg.device)
    tm.net.load_state_dict(ck["state"])
    tm._mu, tm._sd, tm.feature_names = ck["mu"], ck["sd"], tuple(names)
    return beta, booster, tm


def cmd_score_oos(args) -> int:
    if not args.unlock_oos:
        print("OOS は施錠されています。--unlock-oos を付けて1回だけ実行してください。",
              file=sys.stderr)
        return 2
    ccfg = cv_config()
    meta = json.loads((OUT / "final_eval/place_meta.json").read_text(encoding="utf-8"))
    feat, names, stats = load_frame()
    guard = OOSGuard(ccfg, OOS_LOG)
    guard.unlock(args.reason)

    d = pd.to_datetime(feat["race_date"])
    lo = pd.Timestamp(ccfg.oos[0])
    oos = add_place_label(trainable(feat[d >= lo]))
    df = apply_stats(oos, names, stats)
    rid, k, y = df["race_id"].to_numpy(), df["k_place"].to_numpy(), df["is_place"].to_numpy()

    beta, booster, tm = load_place_models(names)
    raw = {"clogit": df[names].to_numpy(dtype=float) @ beta,
           "lgbm": booster.predict(df[names])}
    b = to_batch(df, names, label_col="is_place")
    raw["tabm"] = np.log(np.clip(b.flat_predictions(tm.predict_proba(b), len(df)), EPS, 1))

    temps, w = meta["temperatures"], np.array([meta["ensemble_weights"][m] for m in MODELS])
    preds = df[["race_id", "horse_no", "race_date", "finish_pos", "is_win",
                "n_runners", "k_place", "is_place"]].copy()
    report = {}
    for m in MODELS:
        preds[f"pp_{m}"] = to_place(raw[m], rid, k, temps[m])
        report[m] = place_metrics(preds[f"pp_{m}"].to_numpy(), y, rid, k)
    logs = np.column_stack([race_log_softmax(raw[m] / temps[m], rid) for m in MODELS])
    preds["pp_ensemble"] = to_place(logs @ w, rid, k, meta["ensemble_temperature"])
    report["ensemble"] = place_metrics(preds["pp_ensemble"].to_numpy(), y, rid, k)
    report["baseline_uniform"] = place_metrics(k / df["n_runners"].to_numpy(), y, rid, k)

    # 比較対象: 単勝モデル（既に開封・保存済みの OOS 予測）を Harville で複勝確率にしたもの
    win = pd.read_parquet(WIN_OOS_PRED,
                          columns=["race_id", "horse_no", "ensemble"])
    preds = preds.merge(win.rename(columns={"ensemble": "pw_ensemble"}),
                        on=["race_id", "horse_no"], how="left")
    ok = preds["pw_ensemble"].notna()
    both = preds[ok].copy()
    both["pw_ensemble"] = both.groupby("race_id")["pw_ensemble"].transform(lambda s: s / s.sum())
    both["pp_from_win"] = harville_place(both["pw_ensemble"].to_numpy(),
                                         both["race_id"].to_numpy(), both["k_place"].to_numpy())
    preds = preds.merge(both[["race_id", "horse_no", "pp_from_win"]],
                        on=["race_id", "horse_no"], how="left")
    sub = preds[ok]
    for col in ("pp_ensemble", "pp_from_win"):
        report[f"{col} (win-joined rows)"] = place_metrics(
            sub[col].to_numpy(), sub["is_place"].to_numpy(), sub["race_id"].to_numpy(),
            sub["k_place"].to_numpy())

    preds.to_parquet(OUT / "oos_place_predictions.parquet", index=False)
    (OUT / "oos_place_metrics.json").write_text(json.dumps(
        {"period": [str(lo.date()), str(pd.to_datetime(df["race_date"]).max().date())],
         "reason": args.reason, "n_win_unmatched": int((~ok).sum()), "metrics": report},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(pd.DataFrame(report).T.round(4).to_string())
    print(f"単勝 OOS 予測と突合できなかった行: {(~ok).sum():,}")
    print(f"→ {OUT / 'oos_place_predictions.parquet'}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    s = sub.add_parser("fit-prod")
    s.add_argument("--win-release", default=DEFAULT_WIN_RELEASE)
    s.set_defaults(func=cmd_fit_prod)
    s = sub.add_parser("score-oos")
    s.add_argument("--unlock-oos", action="store_true")
    s.add_argument("--reason", required=True)
    s.set_defaults(func=cmd_score_oos)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
