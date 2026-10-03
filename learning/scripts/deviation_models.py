"""人気順との乖離を目的変数にした単勝モデル（2026-10-02）。

人気順の派生系統（conf_pop / conf_banei_pop）の gold と、その評価用モデル
（artifacts/final_pop_eval / final_banei_pop_eval）が選んだ特徴量・標準化統計量を使い、
2通りの「乖離」を学習してそれぞれアンサンブルする。

  resid（残差）: 人気だけから作った単勝確率 p_pop を固定オフセットにして
                log π = log p_pop + f(x)、f が人気からの乖離。出力はそのまま単勝確率。
                  clogit（offset 付き条件付きロジット）/ lgbm（LambdaRank、init_score=log p_pop）
                  / tabm（offset_last=True）
  reg（回帰）  : 目的変数 = 着順 − 人気（負ほど人気より走った）。
                  ridge / lgbm（L2 回帰）/ tabm（task="regression"）
                確率への変換は log π = log p_pop − γ·予測値 をレース内 softmax して温度 T。
                γ・T はモデルごとに保留区間のレース NLL で決める。

p_pop は「人気 × 頭数」ごとの勝率表（学習区間だけで集計、頭数の逆数へ収縮）をレース内で
正規化したもの。ベースラインとしても単独で評価する。

学習区間・較正区間は単勝の評価用モデルと同じ（1998-01-01〜2023-08-04、末尾 90 日を較正）。
アンサンブル重みは較正区間の予測で制約付きスタッキング（各モデル温度スケーリング後）。
ハイパーパラメータは既定値（単勝・複勝の評価用モデルと同じ）。

使い方（learning/ で）:
    PLACE_FAMILY=flat_pop NAR_CONF_DIR=conf_pop PYTHONPATH=src .venv/bin/python \\
        scripts/deviation_models.py fit
    PLACE_FAMILY=flat_pop NAR_CONF_DIR=conf_pop PYTHONPATH=src .venv/bin/python \\
        scripts/deviation_models.py score-oos --unlock-oos --reason "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nar.config import cv_config  # noqa: E402
from nar.eval import metrics  # noqa: E402
from nar.eval.calibration import TemperatureScaler  # noqa: E402
from nar.eval.place import race_log_softmax  # noqa: E402
from nar.eval.splits import OOSGuard  # noqa: E402
from nar.models.base import to_batch  # noqa: E402
from nar.models.clogit import ConditionalLogit  # noqa: E402
from nar.models.ensemble import ConstrainedStacker  # noqa: E402
from nar.models.lgbm import LgbmRanker, graded_labels, group_sizes  # noqa: E402
from nar.models.tabm import TabM, TabMConfig  # noqa: E402
from nar.train.pipeline import apply_stats, trainable  # noqa: E402

log = logging.getLogger("deviation_models")

FAMILY = os.environ.get("PLACE_FAMILY", "flat_pop")
_CONF = {"flat_pop": "conf_pop", "banei_pop": "conf_banei_pop"}
if FAMILY not in _CONF:
    raise ValueError(f"PLACE_FAMILY は {tuple(_CONF)} のいずれか（受領: {FAMILY!r}）")
if os.environ.get("NAR_CONF_DIR") != _CONF[FAMILY]:
    raise RuntimeError(f"PLACE_FAMILY={FAMILY} には NAR_CONF_DIR={_CONF[FAMILY]} が必要です")
_P = {
    "flat_pop": dict(gold="data_real/gold/features_noodds_pop/features.parquet",
                     win_eval="artifacts/final_pop_eval", oos_log="artifacts/oos_access.log"),
    "banei_pop": dict(gold="data_real/gold/features_noodds_banei_pop/features.parquet",
                      win_eval="artifacts/final_banei_pop_eval",
                      oos_log="artifacts/banei/oos_access.log"),
}[FAMILY]
GOLD = ROOT / _P["gold"]
WIN_EVAL_DIR = ROOT / _P["win_eval"]
OOS_LOG = ROOT / _P["oos_log"]
OUT = ROOT / f"artifacts/dev_{FAMILY}"
HOLDOUT_DAYS = 90
TABM_CFG = dict(k=4, hidden=128, n_layers=2, epochs=3, batch_races=256, seed=0)
POP_SMOOTH = 20.0          # 勝率表の収縮の強さ（何走ぶんの 1/頭数 を混ぜるか）
MAX_RANK = 16
RESID = ("clogit", "lgbm", "tabm")
REG = ("ridge", "lgbm", "tabm")


# ------------------------------------------------------------------ 人気の事前確率
def pop_inputs(df: pd.DataFrame) -> pd.DataFrame:
    n = df.groupby("race_id")["race_id"].transform("size")
    rank = df["market_pop_rank"].fillna(n).clip(1, MAX_RANK)
    return df.assign(n_runners=n.to_numpy(), pop_rank=rank.astype(int).to_numpy())


def fit_pop_table(df: pd.DataFrame) -> dict:
    g = df.groupby(["pop_rank", "n_runners"])["is_win"].agg(["sum", "count"]).reset_index()
    prior = 1.0 / g["n_runners"]
    g["p"] = (g["sum"] + POP_SMOOTH * prior) / (g["count"] + POP_SMOOTH)
    return {f"{int(r)}_{int(n)}": float(p) for r, n, p in zip(g["pop_rank"], g["n_runners"], g["p"])}


def log_p_pop(df: pd.DataFrame, table: dict) -> np.ndarray:
    raw = np.array([table.get(f"{r}_{n}", 1.0 / n)
                    for r, n in zip(df["pop_rank"], df["n_runners"])])
    return np.log(metrics.normalize_within_race(raw, df["race_id"].to_numpy()))


# ------------------------------------------------------------------ 学習
def fit_resid(fit_df, names):
    x_cols = [*names, "offset"]
    models = {}
    t0 = time.time()
    models["clogit"] = ConditionalLogit(l2=1e-3, offset_last=True).fit(
        to_batch(fit_df, x_cols, label_col="is_win"))
    log.info("resid clogit %.0fs", time.time() - t0)
    import lightgbm as lgb

    t0 = time.time()
    rk = LgbmRanker()
    ds = lgb.Dataset(fit_df[names], label=graded_labels(fit_df["finish_pos"], rk.label_grades),
                     group=group_sizes(fit_df), init_score=fit_df["offset"].to_numpy(),
                     feature_name=list(names), free_raw_data=False)
    models["lgbm"] = lgb.train(rk.params, ds, num_boost_round=rk.num_boost_round)
    log.info("resid lgbm %.0fs", time.time() - t0)
    t0 = time.time()
    models["tabm"] = TabM(TabMConfig(offset_last=True, **TABM_CFG)).fit(
        to_batch(fit_df, x_cols, label_col="is_win"))
    log.info("resid tabm %.0fs", time.time() - t0)
    return models


def score_resid(models, df, names) -> dict[str, np.ndarray]:
    """各モデルの対数強さ（オフセット込み）。"""
    x_cols = [*names, "offset"]
    b = to_batch(df, x_cols, label_col="is_win")
    p = b.flat_predictions(models["clogit"].predict_proba(b), len(df))
    out = {"clogit": np.log(np.clip(p, 1e-12, 1))}
    out["lgbm"] = models["lgbm"].predict(df[names]) + df["offset"].to_numpy()
    out["tabm"] = b.flat_predictions(models["tabm"].predict_scores(b), len(df))
    return out


def fit_reg(fit_df, names):
    from sklearn.linear_model import Ridge
    import lightgbm as lgb

    y = fit_df["dev_target"].to_numpy()
    models = {}
    t0 = time.time()
    models["ridge"] = Ridge(alpha=1.0).fit(fit_df[names].to_numpy(), y)
    log.info("reg ridge %.0fs", time.time() - t0)
    t0 = time.time()
    params = {k: v for k, v in LgbmRanker().params.items()
              if k not in ("objective", "metric", "ndcg_eval_at", "label_gain")}
    params.update(objective="regression", metric="l2")
    models["lgbm"] = lgb.train(params, lgb.Dataset(fit_df[names], label=y,
                                                   feature_name=list(names)),
                               num_boost_round=300)
    log.info("reg lgbm %.0fs", time.time() - t0)
    t0 = time.time()
    models["tabm"] = TabM(TabMConfig(task="regression", **TABM_CFG)).fit(
        to_batch(fit_df, list(names), label_col="dev_target"))
    log.info("reg tabm %.0fs", time.time() - t0)
    return models


def score_reg(models, df, names) -> dict[str, np.ndarray]:
    """各モデルの予測値（着順 − 人気）。"""
    out = {"ridge": models["ridge"].predict(df[names].to_numpy()),
           "lgbm": models["lgbm"].predict(df[names])}
    b = to_batch(df, list(names), label_col="dev_target")
    out["tabm"] = b.flat_predictions(models["tabm"].predict_scores(b), len(df))
    return out


# ------------------------------------------------------------------ 較正・合成
def calibrate_resid(raw: dict, rid, y) -> dict:
    return {m: float(TemperatureScaler().fit(race_log_softmax(raw[m], rid), y, rid).temperature)
            for m in raw}


def calibrate_reg(pred: dict, offset, rid, y) -> dict:
    out = {}
    for m, v in pred.items():
        def nll(th):
            s = (offset - th[0] * v) / np.exp(th[1])
            return metrics.race_nll(np.exp(race_log_softmax(s, rid)), y, rid)
        res = minimize(nll, np.array([0.1, 0.0]), method="Nelder-Mead",
                       options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-7})
        out[m] = {"gamma": float(res.x[0]), "temperature": float(np.exp(res.x[1]))}
    return out


def probs_resid(raw, temps, rid):
    return {m: np.exp(race_log_softmax(raw[m] / temps[m], rid)) for m in raw}


def probs_reg(pred, cal, offset, rid):
    return {m: np.exp(race_log_softmax((offset - cal[m]["gamma"] * pred[m])
                                       / cal[m]["temperature"], rid)) for m in pred}


def stack(probs: dict, y, rid) -> np.ndarray:
    names = list(probs)
    st = ConstrainedStacker(names).fit(pd.DataFrame(probs), y, rid)
    return dict(zip(names, st.weights.tolist()))


def blend(probs: dict, w: dict, rid) -> np.ndarray:
    st = ConstrainedStacker(list(w))
    st.weights = np.array([w[m] for m in w])
    return st.predict_proba(pd.DataFrame({m: probs[m] for m in w}), rid)


# ------------------------------------------------------------------ データ
def load():
    names = json.loads((WIN_EVAL_DIR / "feature_names.json").read_text(encoding="utf-8"))
    stats = json.loads((WIN_EVAL_DIR / "standardizer.json").read_text(encoding="utf-8"))
    meta = json.loads((WIN_EVAL_DIR / "final_meta.json").read_text(encoding="utf-8"))
    cols = ["race_id", "horse_no", "race_date", "finish_pos", "is_win", "market_pop_rank",
            *[c for c in names if c != "market_pop_rank"]]
    feat = pd.read_parquet(GOLD, columns=cols)
    if (GOLD.parent / "content_hash.txt").read_text().strip() != meta["gold_content_hash"]:
        raise RuntimeError("gold の内容が評価用モデルの学習時と違います")
    return feat, names, stats, meta


def prepare(raw: pd.DataFrame, names, stats, table) -> pd.DataFrame:
    df = pop_inputs(raw)
    df["dev_target"] = df["finish_pos"] - df["pop_rank"]
    keep = df[["pop_rank", "n_runners", "dev_target", "market_pop_rank"]]
    out = apply_stats(df, names, stats)          # 行順は race_id, horse_no で並べ直される
    key = df.set_index(["race_id", "horse_no"])
    idx = pd.MultiIndex.from_arrays([out["race_id"], out["horse_no"]])
    for c in ("pop_rank", "n_runners", "dev_target"):
        out[c] = key[c].reindex(idx).to_numpy()
    del keep
    if table is not None:
        out["offset"] = log_p_pop(out, table)
    return out


def summary(p, df) -> dict:
    return metrics.summary(p, df["is_win"].to_numpy(), df["finish_pos"].to_numpy(),
                           df["race_id"].to_numpy())


# ------------------------------------------------------------------ コマンド
def cmd_fit(args) -> int:
    feat, names, stats, meta = load()
    start, end = (pd.Timestamp(meta["train_period"][k]) for k in ("start", "end"))
    d = pd.to_datetime(feat["race_date"])
    window = trainable(feat[(d >= start) & (d <= end)])
    wd = pd.to_datetime(window["race_date"])
    cut = wd.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    fit_raw, cal_raw = window[wd <= cut], window[wd > cut]
    table = fit_pop_table(pop_inputs(fit_raw))
    fit_df = prepare(fit_raw, names, stats, table)
    cal_df = prepare(cal_raw, names, stats, table)
    log.info("学習 %s 行 / 較正 %s 行（%s〜%s）", f"{len(fit_df):,}", f"{len(cal_df):,}",
             start.date(), end.date())
    rid, y = cal_df["race_id"].to_numpy(), cal_df["is_win"].to_numpy()

    resid_models = fit_resid(fit_df, names)
    reg_models = fit_reg(fit_df, names)
    raw_r = score_resid(resid_models, cal_df, names)
    pred_g = score_reg(reg_models, cal_df, names)
    temps = calibrate_resid(raw_r, rid, y)
    regcal = calibrate_reg(pred_g, cal_df["offset"].to_numpy(), rid, y)
    pr = probs_resid(raw_r, temps, rid)
    pg = probs_reg(pred_g, regcal, cal_df["offset"].to_numpy(), rid)
    w_r, w_g = stack(pr, y, rid), stack(pg, y, rid)

    report = {"p_pop": summary(np.exp(cal_df["offset"].to_numpy()), cal_df)}
    for m in RESID:
        report[f"resid_{m}"] = summary(pr[m], cal_df)
    report["resid_ensemble"] = summary(blend(pr, w_r, rid), cal_df)
    for m in REG:
        report[f"reg_{m}"] = summary(pg[m], cal_df)
    report["reg_ensemble"] = summary(blend(pg, w_g, rid), cal_df)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "models.pkl", "wb") as f:
        pickle.dump({"resid": resid_models, "reg": reg_models}, f)
    (OUT / "dev_meta.json").write_text(json.dumps({
        "family": FAMILY, "features": names, "train_period": meta["train_period"],
        "calibration_after": str(cut.date()), "pop_table": table,
        "resid_temperatures": temps, "resid_weights": w_r,
        "reg_calibration": regcal, "reg_weights": w_g,
        "holdout_metrics": report,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(pd.DataFrame(report).T[["race_nll", "top1", "top3", "ece"]].round(4).to_string())
    print(f"resid 重み {w_r}\nreg 重み {w_g}\nreg 較正 {regcal}")
    return 0


def cmd_score_oos(args) -> int:
    if not args.unlock_oos:
        print("--unlock-oos を付けて1回だけ実行してください。", file=sys.stderr)
        return 2
    ccfg = cv_config()
    meta = json.loads((OUT / "dev_meta.json").read_text(encoding="utf-8"))
    with open(OUT / "models.pkl", "rb") as f:
        models = pickle.load(f)
    feat, names, stats, _ = load()
    OOSGuard(ccfg, OOS_LOG).unlock(args.reason)
    lo = pd.Timestamp(ccfg.oos[0])
    df = prepare(trainable(feat[pd.to_datetime(feat["race_date"]) >= lo]), names, stats,
                 meta["pop_table"])
    rid = df["race_id"].to_numpy()
    pr = probs_resid(score_resid(models["resid"], df, names), meta["resid_temperatures"], rid)
    pg = probs_reg(score_reg(models["reg"], df, names), meta["reg_calibration"],
                   df["offset"].to_numpy(), rid)

    base = df[["race_id", "horse_no", "race_date", "is_win", "finish_pos"]].copy()
    report = {"p_pop": summary(np.exp(df["offset"].to_numpy()), df)}
    for tag, probs, w in (("resid", pr, meta["resid_weights"]), ("reg", pg, meta["reg_weights"])):
        out = base.copy()
        for m, p in probs.items():
            out[m] = p
            report[f"{tag}_{m}"] = summary(p, df)
        out["ensemble"] = blend(probs, w, rid)
        report[f"{tag}_ensemble"] = summary(out["ensemble"].to_numpy(), df)
        out.to_parquet(OUT / f"oos_win_{tag}.parquet", index=False)
    pd.DataFrame(base.assign(ensemble=np.exp(df["offset"].to_numpy()))).to_parquet(
        OUT / "oos_win_pop.parquet", index=False)
    (OUT / "oos_metrics.json").write_text(json.dumps(
        {"reason": args.reason, "metrics": report}, ensure_ascii=False, indent=2,
        default=float), encoding="utf-8")
    print(pd.DataFrame(report).T[["race_nll", "top1", "top3", "ece"]].round(4).to_string())
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit").set_defaults(func=cmd_fit)
    s = sub.add_parser("score-oos")
    s.add_argument("--unlock-oos", action="store_true")
    s.add_argument("--reason", required=True)
    s.set_defaults(func=cmd_score_oos)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
