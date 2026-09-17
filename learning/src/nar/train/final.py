"""本番用モデルの最終学習と配布物の書き出し。

walk-forward はモデルの**評価**であって、本番に出すモデルそのものは作らない。
fold ごとのモデルは学習期間が短く、最新の数年を見ていない。本番には
「OOS 境界の直前までの全データ」で学習し直したモデルを出す。

較正温度と特徴量選択は、この最終学習期間の内側だけで決める。walk-forward の
どこかの fold で決めた値を持ち込むと、その fold の valid が本番モデルに漏れる。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CVConfig, FeatureConfig
from ..eval import metrics
from ..eval.calibration import TemperatureScaler
from ..models.base import to_batch
from ..models.clogit import ConditionalLogit
from ..models.lgbm import LgbmRanker
from .pipeline import apply_stats, fit_stats, trainable

log = logging.getLogger(__name__)

# 較正温度を測るための保留期間。学習期間の末尾をここに使う。
# OOS には一切触れない。
HOLDOUT_DAYS = 90


@dataclass
class FinalArtifacts:
    out_dir: Path
    feature_names: list[str]
    train_period: dict[str, str]
    temperatures: dict[str, float]
    written: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def training_window(ccfg: CVConfig, fcfg: FeatureConfig,
                    through: pd.Timestamp | str | None = None
                    ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """学習に使ってよい期間。

    既定の終端は OOS 開始日から embargo 日を引いた日。embargo を引かないと、
    履歴系特徴量が OOS 期間の情報を含む（設計書 §CV）。**評価用**のモデルは
    必ずこれを使う。

    through を渡すとその日で打ち切る。OOS での評価が済んだあと、**出荷用**の
    モデルを入手できる全データで学習し直すのに使う。保留評価で性能を測り、
    それから全データで作り直すのは標準的な手順で、embargo は評価を守るための
    ものだから、後ろに検証区間を持たない出荷モデルには要らない。
    ここを既定にはしない。既定で全期間を使うと、評価用と出荷用の区別が消える。
    """
    start = pd.Timestamp(ccfg.train_start)
    if through is not None:
        end = pd.Timestamp(through)
    else:
        end = pd.Timestamp(ccfg.oos[0]) - pd.Timedelta(days=ccfg.embargo_days + 1)
    if end <= start:
        raise ValueError(
            f"学習期間が空です（{start.date()} 〜 {end.date()}）。"
            "embargo が長すぎるか OOS 開始が早すぎます。")
    return start, end


def fit_and_export(
    feat: pd.DataFrame, cols: list[str], ccfg: CVConfig, fcfg: FeatureConfig,
    out_dir: str | Path, gold_hash: str | None = None,
    through: str | None = None,
    models: tuple[str, ...] = ("clogit", "lgbm", "tabm"),
    seed: int = 0, tabm_epochs: int = 3, tabm_width: dict | None = None,
    do_selection: bool = True, n_null_runs: int = 5,
    holdout_days: int = HOLDOUT_DAYS,
    hpo_params: dict[str, dict] | None = None,
) -> FinalArtifacts:
    """本番モデルを学習し、配布物一式を書き出す。

    `hpo_params`（2026-09-17 追加）: `{"clogit": {...}, "lgbm": {...}, "tabm": {...}}`。
    `nar learn` の nested HPO（`train/hpo.py`）が見つけた値をここで初めて配布物に
    反映させる。**これを渡さない限り、fit-final は HPO の結果と無関係にハードコードされた
    既定値（clogit l2=1e-3、lgbm の LightGBM 既定、tabm は `tabm_width`/`tabm_epochs`
    のみ）で学習する** — `nar learn` で何時間かけて探索しても、配布モデルはその恩恵を
    一切受けない状態だった。値は `_fit_one` がモデルごとの構築式にそのままマージする
    （キーが無ければ元の既定値を使うので、`hpo_params=None` は完全に後方互換）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    start, end = training_window(ccfg, fcfg, through)
    dates = pd.to_datetime(feat["race_date"])
    window = feat[(dates >= start) & (dates <= end)]
    window = trainable(window)
    if window.empty:
        raise ValueError(f"{start.date()} 〜 {end.date()} に学習対象がありません")

    # 較正温度は学習期間の末尾で測る。同じ行で学習も較正もすると温度が 1 に張り付く。
    cut = pd.to_datetime(window["race_date"]).max() - pd.Timedelta(days=holdout_days)
    wd = pd.to_datetime(window["race_date"])
    fit_raw, cal_raw = window[wd <= cut], window[wd > cut]
    log.info("最終学習: %s 〜 %s / 学習 %s 行・較正 %s 行",
             start.date(), end.date(), f"{len(fit_raw):,}", f"{len(cal_raw):,}")
    if cal_raw.empty:
        raise ValueError("較正用の保留期間が空です。holdout_days を短くしてください。")

    selected = list(cols)
    if do_selection:
        from ..features.selection import select

        sel = select(apply_stats(fit_raw, cols, fit_stats(fit_raw, cols)), cols,
                     seed=seed, n_null_runs=n_null_runs)
        selected = sel.selected
        log.info("最終学習: 特徴量 %d → %d", len(cols), len(selected))

    # 補完値と標準化の統計量は**学習に使った行だけ**から取り、配布物に固める。
    # 推論側は1レース分しか手元に無いので、その場で計算すると
    # 「レース内での z 値」という別の量になり、モデルは学習時と違う入力を見る。
    stats = fit_stats(fit_raw, selected)
    fit_df = apply_stats(fit_raw, selected, stats)
    cal_df = apply_stats(cal_raw, selected, stats)
    art = FinalArtifacts(out, selected,
                         {"start": str(start.date()), "end": str(end.date())}, {})

    hpo_params = hpo_params or {}
    for name in models:
        t0 = time.time()
        try:
            p = _fit_one(name, fit_df, cal_df, selected, out, art,
                         seed=seed, tabm_epochs=tabm_epochs,
                         tabm_width=tabm_width or {},
                         hpo=hpo_params.get(name, {}))
        except Exception as exc:  # noqa: BLE001
            log.exception("最終学習: %s が失敗しました: %s", name, exc)
            art.notes.append(f"{name}: 失敗（{exc}）")
            continue
        rid = cal_df["race_id"].to_numpy()
        y = cal_df["is_win"].to_numpy()
        p = metrics.normalize_within_race(np.nan_to_num(p, nan=1e-9), rid)
        scaler = TemperatureScaler().fit(np.log(np.clip(p, 1e-12, 1)), y, rid)
        art.temperatures[name] = float(scaler.temperature)
        log.info("最終学習: %s 完了（%.1f 秒、温度 %.4f）",
                 name, time.time() - t0, scaler.temperature)

    (out / "feature_names.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "standardizer.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "final_meta.json").write_text(json.dumps({
        # 学習に使った gold の内容ハッシュ。取り込み時の dataset_version とは
        # 別物で、「このモデルが何を見たか」を一意に指すのはこちら。
        # `nar features` が書き出した値を読む。ここで数百万行を CSV 化して
        # 計算し直すと、それだけで数十分かかる。
        "gold_content_hash": gold_hash or "",
        "train_period": art.train_period,
        # 評価用（OOS 境界で打ち切り）か、出荷用（全データ）かを残す。
        # manifest の oos_metrics は評価用モデルで測った数字なので、
        # どちらの版かが分からないと数字の意味が読めない。
        "purpose": "production" if through else "evaluation",
        # 出荷用のとき、OOS を測ったのは別の（境界で打ち切った）モデル。
        # どの学習期間で測った数字かを持たせておく。
        "oos_evaluated_on": (str(pd.Timestamp(ccfg.oos[0])
                                 - pd.Timedelta(days=ccfg.embargo_days + 1))[:10]
                             if through else None),
        "holdout_days": holdout_days,
        "n_fit_rows": int(len(fit_raw)), "n_calibration_rows": int(len(cal_raw)),
        "temperatures": art.temperatures,
        "feature_names": selected,
        "standardizer": stats,
        "hpo_params": hpo_params,
        "notes": art.notes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    art.written.extend(["feature_names.json", "standardizer.json", "final_meta.json"])
    return art


def _fit_one(name, fit_df, cal_df, cols, out: Path, art: FinalArtifacts,
             seed: int, tabm_epochs: int, tabm_width: dict,
             hpo: dict | None = None) -> np.ndarray:
    """`hpo` は `nar learn` の nested HPO が見つけた値（無ければ空 dict = 既定値のまま）。"""
    hpo = hpo or {}
    if name == "clogit":
        model = ConditionalLogit(l2=hpo.get("l2", 1e-3),
                                 l1=hpo.get("l1", 0.0)).fit(to_batch(fit_df, cols))
        (out / "clogit_beta.json").write_text(
            json.dumps({"beta": model.coefficients()}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        art.written.append("clogit_beta.json")
        b = to_batch(cal_df, cols)
        return b.flat_predictions(model.predict_proba(b), len(cal_df))

    if name == "lgbm":
        # HPO 側（train/hpo.py::suggest_lgbm）と同じ unpack 規約。
        # `_num_boost_round`/`_label_grades` は LightGBM のネイティブパラメータではない。
        p = dict(hpo)
        rounds = p.pop("_num_boost_round", 300)
        ranker = LgbmRanker(p, num_boost_round=rounds).fit(fit_df, cols)
        ranker.booster.save_model(str(out / "lgbm_rank.txt"))
        art.written.append("lgbm_rank.txt")
        return ranker.predict_proba(cal_df)

    if name == "tabm":
        from ..models.tabm import TabM, TabMConfig

        # tabm_width（fit-final の --tabm-k 等）を既定にしつつ、HPO が見つけた値で
        # 上書きする。HPO は width（k/hidden/n_layers）そのものも探索するので、
        # ここが一致して初めて「HPO が選んだアーキテクチャがそのまま配布される」。
        merged = {**tabm_width, **hpo}
        cfg = TabMConfig(epochs=tabm_epochs,
                         **{k: v for k, v in merged.items()
                            if k in TabMConfig.__annotations__})
        model = TabM(cfg).fit(to_batch(fit_df, cols), to_batch(cal_df, cols))
        _export_tabm(model, len(cols), out, art, cal_df, cols)
        b = to_batch(cal_df, cols)
        return b.flat_predictions(model.predict_proba(b), len(cal_df))

    raise ValueError(f"最終学習の対象外のモデルです: {name}")


def _export_tabm(model, n_features: int, out: Path, art: FinalArtifacts,
                 cal_df: pd.DataFrame, cols: list[str]) -> None:
    """ONNX 化して、元モデルと出力が一致することを確認してから残す（MP-07）。

    一致しなければ配布物に含めない。壊れた ONNX を出すくらいなら、
    そのモデルを外して残りで運用するほうが安全。
    """
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "operation" / "src"))
        from narops.publish import export_tabm_onnx, verify_onnx
    except Exception as exc:  # noqa: BLE001
        art.notes.append(f"tabm: ONNX 変換をスキップ（{exc}）")
        return

    path = out / "tabm.onnx"
    try:
        export_tabm_onnx(model, n_features, path)
        # 検証はレースごとの並びで行う。1着が入れ替わらないことまで見ないと、
        # 微小な数値差が順位を変えていても気付けない（MP-07）。
        # 検証用の行列は **パディングしていない** 実データから作る。
        # to_batch はレースを最大頭数の矩形に揃えるので、その x を平坦化すると
        # 行数が実際の出走数の合計と合わず、レース単位の突き合わせが崩れる。
        head = cal_df.head(4096)
        sizes = head.groupby("race_id", sort=False).size().tolist()
        sample = head[cols].to_numpy(dtype=float)
        check = verify_onnx(model, path, sample.astype(np.float32), sizes)
        if not check["passed"]:
            raise ValueError(f"最大差 {check['max_abs_diff']:.3e} / "
                             f"Top-1 一致 {check['top1_match']:.1%}")
    except Exception as exc:  # noqa: BLE001
        art.notes.append(f"tabm: ONNX 変換または検証に失敗（{exc}）。配布物から外します")
        # 重みを別ファイルに書き出す形式があるので、本体だけ消すと孤児が残る
        for stray in out.glob(path.name + "*"):
            stray.unlink(missing_ok=True)
        return
    art.written.extend(sorted(f.name for f in out.glob(path.name + "*")))
