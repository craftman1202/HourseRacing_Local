"""全モデルの walk-forward 学習と OOS 評価。

fold ごとに、特徴量選択 → HPO → 各モデル学習 → レース内正規化 → 温度スケーリング
→ OOF 予測の蓄積、という順で回す。アンサンブルの重みは OOF のみから推定する。

OOS はここでは触らない。`evaluate_oos()` を明示的に呼んだときだけ開封する。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CVConfig, FeatureConfig
from ..eval import guards, metrics
from ..eval.calibration import TemperatureScaler
from ..eval.splits import Fold, OOSGuard, make_folds, make_oos_fold, validate_folds
from ..features.builder import asof_features
from ..models import baselines
from ..models.base import to_batch
from ..models.clogit import ConditionalLogit
from ..models.ensemble import ConstrainedStacker, log_average, simple_average
from ..models.lgbm import LgbmRanker

log = logging.getLogger(__name__)

MODEL_ORDER = ("clogit", "lgbm", "tabm", "bayes")

# 最後にベイズを走らせたときの収束診断。レポートがここから読む。
LAST_BAYES_DIAGNOSTICS: dict = {}


@dataclass
class RunConfig:
    models: tuple[str, ...] = MODEL_ORDER
    hpo_trials: dict[str, int] = field(default_factory=dict)
    do_selection: bool = True
    n_null_runs: int = 15
    calibrate: bool = True
    bayes_max_races: int = 4000     # NUTS/SVI に載せる直近レース数の上限
    bayes_method: str = "svi"
    tabm_epochs: int = 60
    # CPU 実行では TabM の幅がそのまま実行時間になる（実測 200k 行で
    # hidden=256/3層/k=8 が 67 秒/エポック、hidden=128/2層/k=4 が 10 秒/エポック）。
    # 幅を実行ごとに明示して、レポートに残せるようにする。
    tabm_width: dict = field(default_factory=dict)
    seed: int = 0


@dataclass
class FoldOutput:
    fold: int
    n_train: int
    n_valid: int
    selected_features: list[str]
    predictions: pd.DataFrame          # race_id, horse_no, is_win, finish_pos, <model>...
    metrics: pd.DataFrame
    hpo: dict[str, dict]
    timings: dict[str, float]
    temperatures: dict[str, float]


def fit_stats(feat: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    """補完値と標準化の統計量を取り出す。

    配布物に固めて推論側へ渡すために要る。推論は1レース分しか手元に無いので、
    その場で計算すると「レース内での z 値」という全く別の量になる。
    """
    stats: dict[str, dict[str, float]] = {}
    for c in cols:
        s = pd.to_numeric(feat[c], errors="coerce")
        med = s.median()
        filled = s.fillna(0.0 if pd.isna(med) else med)
        sd = filled.std()
        stats[c] = {
            "median": 0.0 if pd.isna(med) else float(med),
            "mean": float(filled.mean()),
            "std": float(sd) if sd and sd > 0 else 1.0,
        }
    return stats


def apply_stats(feat: pd.DataFrame, cols: list[str],
                stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """凍結した統計量で補完・標準化する。"""
    out = feat.copy()
    for c in cols:
        st = stats.get(c) or {"median": 0.0, "mean": 0.0, "std": 1.0}
        s = pd.to_numeric(out[c], errors="coerce").fillna(st["median"])
        out[c] = (s - st["mean"]) / (st["std"] or 1.0)
    return out.sort_values(["race_id", "horse_no"]).reset_index(drop=True)


def prepare(feat: pd.DataFrame, cols: list[str],
            stats: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """欠損補完と標準化。

    stats を渡さない場合、統計量は**渡されたフレーム内**でのみ計算する。
    呼び出し側が train と valid を別々に通すことで、valid の統計が train に
    混ざらない。

    stats を渡すと、その凍結した値を使う。配布モデルの推論はこちらを使う。
    推論時は1レース分しか無いため、その場で計算すると「レース内での z 値」に
    なってしまい、学習時と全く別の入力をモデルに渡すことになる。
    """
    if stats is not None:
        return apply_stats(feat, cols, stats)
    return apply_stats(feat, cols, fit_stats(feat, cols))


def _fit_predict_clogit(params, train, valid, cols) -> np.ndarray:
    model = ConditionalLogit(l2=params.get("l2", 1e-3), l1=params.get("l1", 0.0))
    model.fit(to_batch(train, cols))
    b = to_batch(valid, cols)
    return b.flat_predictions(model.predict_proba(b), len(valid))


def _fit_predict_lgbm(params, train, valid, cols) -> np.ndarray:
    p = dict(params)
    rounds = p.pop("_num_boost_round", 300)
    return LgbmRanker(p, num_boost_round=rounds).fit(train, cols).predict_proba(valid)


def _fit_predict_tabm(params, train, valid, cols, epochs: int = 60) -> np.ndarray:
    from ..models.tabm import TabM, TabMConfig

    cfg = TabMConfig(epochs=epochs, **{k: v for k, v in params.items()
                                       if k in TabMConfig.__annotations__})
    model = TabM(cfg).fit(to_batch(train, cols), to_batch(valid, cols))
    b = to_batch(valid, cols)
    return b.flat_predictions(model.predict_proba(b), len(valid))


def _fit_predict_bayes(train, valid, cols, cfg: RunConfig) -> np.ndarray:
    from ..models.bayes import (
        BayesConfig, HierarchicalPlackettLuce, build_arrays, restrict_likelihood_to,
    )

    # 全期間フル推論は現実的でないので直近窓に絞る（設計書 §8.4）。
    # prepare() は race_id 順に並べるが、race_id は競馬場コード始まりなので
    # 時系列順ではない。そのまま tail を取ると「直近」ではなく「場コードが大きい
    # レース」を取ってしまうので、必ず start_ts で並べ直してから切る。
    cutoff_races = (train[["race_id", "start_ts"]].drop_duplicates()
                    .sort_values("start_ts").tail(cfg.bayes_max_races)["race_id"])
    tr = train[train["race_id"].isin(set(cutoff_races))]
    combined = pd.concat([tr, valid], ignore_index=True)
    cov = [c for c in ("draw_rel", "field_size", "log_prize") if c in cols]
    # インデックス空間は train+valid で共有する（valid の馬の事後を引くため）が、
    # 尤度は train のレースだけに効かせる。ここを分けないと答えを見て答える。
    arr = build_arrays(combined, cov)
    fit_arr = restrict_likelihood_to(arr, set(tr["race_id"]))

    model = HierarchicalPlackettLuce(BayesConfig(seed=cfg.seed))
    (model.fit_nuts if cfg.bayes_method == "nuts" else model.fit_svi)(fit_arr)

    global LAST_BAYES_DIAGNOSTICS
    LAST_BAYES_DIAGNOSTICS = {"method": cfg.bayes_method, "n_train_rows": len(tr),
                              **model.cfg.diagnostics}

    p_all = model.flat_predictions(arr, len(combined))
    return p_all[len(tr):]


def run_fold(
    fold: Fold, feat: pd.DataFrame, cols: list[str], cfg: RunConfig,
    ccfg: CVConfig,
) -> FoldOutput:
    tr_mask, va_mask = fold.mask(feat["race_date"])
    train_raw, valid_raw = feat[tr_mask].copy(), feat[va_mask].copy()

    timings: dict[str, float] = {}
    selected = list(cols)
    if cfg.do_selection:
        from ..features.selection import select

        t0 = time.time()
        # 選択は fold の学習期間内でのみ実行する（CV-10）
        sel = select(prepare(train_raw, cols), cols, seed=cfg.seed + fold.index,
                     n_null_runs=cfg.n_null_runs)
        selected = sel.selected
        timings["selection"] = time.time() - t0
        log.info("fold %d: 特徴量 %d → %d", fold.index, len(cols), len(selected))

    train, valid = prepare(train_raw, selected), prepare(valid_raw, selected)
    keep = ["race_id", "horse_no", "race_date", "is_win", "finish_pos"]
    # オッズと人気は予測に使わないが、トラックB の突合と人気帯別 ECE に要る。
    # ここで持ち出さないと下流が silver を引き直す羽目になる。
    keep += [c for c in ("odds_win", "popularity") if c in valid.columns]
    preds = valid[keep].copy()
    rid = valid["race_id"].to_numpy()
    y = valid["is_win"].to_numpy()

    hpo_results: dict[str, dict] = {}
    temps: dict[str, float] = {}

    for name in cfg.models:
        t0 = time.time()
        try:
            params = _tune(name, fold, feat, selected, cfg, ccfg, hpo_results)
            if name == "clogit":
                p = _fit_predict_clogit(params, train, valid, selected)
            elif name == "lgbm":
                p = _fit_predict_lgbm(params, train, valid, selected)
            elif name == "tabm":
                p = _fit_predict_tabm({**cfg.tabm_width, **params}, train, valid,
                                      selected, cfg.tabm_epochs)
            elif name == "bayes":
                p = _fit_predict_bayes(train, valid, selected, cfg)
            else:
                raise ValueError(f"未知のモデル: {name}")
        except Exception as exc:  # noqa: BLE001
            # 1モデルの失敗で fold 全体を落とさない。落ちたことは記録に残す
            log.exception("fold %d: %s が失敗しました: %s", fold.index, name, exc)
            timings[name] = time.time() - t0
            continue

        p = metrics.normalize_within_race(np.nan_to_num(p, nan=1e-9), rid)
        if cfg.calibrate:
            # 温度は valid でのみ最適化する（CA-02）。同じ valid で評価もするので
            # 較正の効果は楽観的に出る。OOS では train 側 fold の温度を使う。
            scaler = TemperatureScaler().fit(np.log(np.clip(p, 1e-12, 1)), y, rid)
            temps[name] = scaler.temperature
            p = scaler.transform(np.log(np.clip(p, 1e-12, 1)), rid)
        preds[name] = p
        timings[name] = time.time() - t0

    # ベースライン
    preds["baseline_uniform"] = baselines.uniform(valid)
    if "odds_win" in valid.columns:
        preds["baseline_market"] = baselines.market(valid)

    rows = []
    for name in [c for c in preds.columns if c not in keep]:
        m = metrics.summary(preds[name].to_numpy(), y, valid["finish_pos"].to_numpy(), rid)
        rows.append({"fold": fold.index, "model": name, **m})

    return FoldOutput(
        fold=fold.index, n_train=int(tr_mask.sum()), n_valid=int(va_mask.sum()),
        selected_features=selected, predictions=preds, metrics=pd.DataFrame(rows),
        hpo=hpo_results, timings=timings, temperatures=temps,
    )


def _tune(name, fold, feat, cols, cfg: RunConfig, ccfg: CVConfig, sink: dict) -> dict:
    """HPO。試行数 0 なら既定パラメータで走らせる。"""
    from . import hpo as hpo_mod

    n = cfg.hpo_trials.get(name, 0)
    if n <= 0 or name not in hpo_mod.SUGGESTERS:
        return _defaults(name)

    fn = {
        "clogit": _fit_predict_clogit,
        "lgbm": _fit_predict_lgbm,
        "tabm": lambda p, tr, va, c: _fit_predict_tabm(
            {**cfg.tabm_width, **p}, tr, va, c, cfg.tabm_epochs),
    }[name]

    # prepare は hpo.run 側で1度だけ適用する。ここで二重に掛けると、
    # 目的関数が見る valid と予測の行順がずれる。
    res = hpo_mod.run(name, fold, feat, cols, fn, n_trials=n,
                      embargo_days=ccfg.embargo_days, n_inner=ccfg.inner_n_folds,
                      seed=cfg.seed, prepare_fn=prepare)
    sink[name] = {"best_params": res.best_params, "best_value": res.best_value,
                  "n_trials": res.n_trials}
    return res.best_params


def _defaults(name: str) -> dict:
    return {"clogit": {"l2": 1e-3}, "lgbm": {"_num_boost_round": 300, "_label_grades": 3},
            "tabm": {}, "bayes": {}}.get(name, {})


def build_oof(folds: list[FoldOutput]) -> pd.DataFrame:
    """OOF 予測。各行はちょうど1つの fold から来る（EN-02 の前提）。"""
    frames = []
    for f in folds:
        p = f.predictions.copy()
        p["fold"] = f.fold
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def fit_ensemble(oof: pd.DataFrame, model_names: list[str]) -> ConstrainedStacker:
    usable = [m for m in model_names if m in oof.columns and oof[m].notna().all()]
    st = ConstrainedStacker(usable)
    st.fit(oof[usable], oof["is_win"].to_numpy(), oof["race_id"].to_numpy(),
           fold_ids=oof["fold"].to_numpy())
    return st


def ensemble_metrics(oof: pd.DataFrame, st: ConstrainedStacker) -> pd.DataFrame:
    rid = oof["race_id"].to_numpy()
    y = oof["is_win"].to_numpy()
    pos = oof["finish_pos"].to_numpy()
    cols = st.model_names
    variants = {
        "ensemble_stacked": st.predict_proba(oof[cols], rid),
        "ensemble_simple_avg": simple_average(oof[cols], rid),
        "ensemble_log_avg": log_average(oof[cols], rid),
    }
    rows = [{"model": k, **metrics.summary(v, y, pos, rid)} for k, v in variants.items()]
    return pd.DataFrame(rows)


def evaluate_oos(
    feat: pd.DataFrame, cols: list[str], cfg: RunConfig, ccfg: CVConfig,
    stacker: ConstrainedStacker, artifacts: Path, reason: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[guards.Tripwire]]:
    """OOS 最終評価。ここで初めて開封する。"""
    guard = OOSGuard(ccfg, artifacts / "oos_access.log")
    guard.unlock(reason)

    oos = make_oos_fold(ccfg, pd.to_datetime(feat["race_date"]).max().date())
    out = run_fold(oos, feat, cols, cfg, ccfg)

    valid = out.predictions
    rid = valid["race_id"].to_numpy()
    have = [m for m in stacker.model_names if m in valid.columns]
    if len(have) == len(stacker.model_names):
        valid["ensemble_stacked"] = stacker.predict_proba(valid[stacker.model_names], rid)

    meta_cols = {"race_id", "horse_no", "race_date", "is_win", "finish_pos",
                 "fold", "odds_win", "popularity"}
    rows = []
    for name in [c for c in valid.columns if c not in meta_cols]:
        rows.append({"model": name, **metrics.summary(
            valid[name].to_numpy(), valid["is_win"].to_numpy(),
            valid["finish_pos"].to_numpy(), rid)})
    summary = pd.DataFrame(rows).sort_values("race_nll")

    best = summary.iloc[0]
    tw = guards.check(oos_top1=float(best["top1"]), oos_nll=float(best["race_nll"]))
    return summary, valid, tw


def trainable(feat: pd.DataFrame, race_col: str = "race_id",
              pos_col: str = "finish_pos", win_col: str = "is_win") -> pd.DataFrame:
    """ラベルが定義できる行だけを残す。

    実データには3種類の「勝者のいないレース」がある。どれも学習には使えない。
      1. 取消・除外で走らなかった馬（着順が NULL）。
         これを「4着以下」として入れると、走っていない馬を負けとして学習し、
         レース内 softmax の分母にも入ってしまう。
      2. 中止・未実施で1行も着順が無いレース（実データで 2,227 レース）。
      3. 1着同着（実データで 604 レース）。離散選択の尤度は「1レース1勝者」を
         前提にしており、同着の扱いは定義されていない。0.14% なので落とす。
    """
    ran = feat[pos_col].notna() if pos_col in feat.columns else pd.Series(True, index=feat.index)
    out = feat[ran]
    wins = out.groupby(race_col)[win_col].transform("sum")
    keep = out[wins == 1]
    log.info("学習対象: %s 行 / %s レース（除外: 未出走 %s 行、勝者が1頭でないレース %s）",
             f"{len(keep):,}", f"{keep[race_col].nunique():,}",
             f"{int((~ran).sum()):,}",
             f"{out.loc[wins != 1, race_col].nunique():,}")
    return keep.reset_index(drop=True)


def run_walkforward(
    feat: pd.DataFrame, fcfg: FeatureConfig, ccfg: CVConfig, cfg: RunConfig,
) -> list[FoldOutput]:
    feat = trainable(feat)
    cols = [c for c in asof_features(fcfg) if c in feat.columns]
    folds = make_folds(ccfg)
    races = feat[["race_id", "race_date"]].drop_duplicates()
    usable = [f for f in folds if _has_data(f, races)]
    validate_folds(usable, races)

    out = []
    for f in usable:
        log.info("=== fold %d: train %s〜%s / valid %s〜%s ===",
                 f.index, f.train_start, f.train_end, f.valid_start, f.valid_end)
        out.append(run_fold(f, feat, cols, cfg, ccfg))
    return out


def _has_data(fold: Fold, races: pd.DataFrame) -> bool:
    tr, va = fold.mask(races["race_date"])
    return bool(tr.sum() > 100 and va.sum() > 100)
