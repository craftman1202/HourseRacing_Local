"""人気との乖離（回帰）モデルの確率変換（2026-10-05）。

学習側（scripts/deviation_models.py）と運用側（narops.runtime / inference）が同じ関数を
呼ぶ（SK-02）。

  p_pop: 人気 × 頭数ごとの勝率表（学習区間で集計・頭数の逆数へ収縮）をレース内で正規化
  回帰モデルの予測 pred = 着順 − 人気 の期待値（負ほど人気より走る）
  log π = (log p_pop − γ · pred) / T をレース内 softmax。γ・T はモデルごと、
  複数モデルは制約付き重み w で log π を幾何平均（ConstrainedStacker と同じ合成）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import normalize_within_race
from .place import race_log_softmax

MAX_RANK = 16


def pop_rank_inputs(rank: np.ndarray, n_runners: np.ndarray) -> np.ndarray:
    """人気が無い馬（取消・オッズ未発表）は最下位扱い。1〜MAX_RANK に丸める。"""
    n = np.asarray(n_runners, dtype=float)
    r = np.asarray(rank, dtype=float)
    r = np.where(np.isfinite(r) & (r > 0), r, n)
    return np.clip(r, 1, MAX_RANK).astype(int)


def log_p_pop(rank: np.ndarray, n_runners: np.ndarray, race_ids: np.ndarray,
              table: dict[str, float]) -> np.ndarray:
    """人気だけから作った単勝確率（の対数）。表に無い組は 1/頭数。"""
    r = pop_rank_inputs(rank, n_runners)
    n = np.asarray(n_runners, dtype=int)
    raw = np.array([table.get(f"{ri}_{ni}", 1.0 / ni) for ri, ni in zip(r, n)])
    return np.log(normalize_within_race(raw, np.asarray(race_ids)))


def reg_probability(offset: np.ndarray, preds: dict[str, np.ndarray],
                    calibration: dict[str, dict[str, float]], weights: dict[str, float],
                    race_ids: np.ndarray) -> np.ndarray:
    """回帰予測 → 単勝確率。重み 0 のモデルは無くてよい。"""
    rid = np.asarray(race_ids)
    names = [m for m, w in weights.items() if w > 0]
    missing = [m for m in names if m not in preds]
    if missing:
        raise KeyError(f"乖離モデルの出力が足りません: {missing}")
    total = sum(weights[m] for m in names)
    log_mix = np.zeros(len(rid))
    for m in names:
        c = calibration[m]
        s = (np.asarray(offset, float) - c["gamma"] * np.asarray(preds[m], float)) / c["temperature"]
        log_mix += (weights[m] / total) * race_log_softmax(s, rid)
    p = np.exp(log_mix)
    return normalize_within_race(p, rid)


def popularity_from_odds(odds_win: pd.Series) -> pd.Series:
    """単勝オッズから人気順を作る（1 = 最低オッズ）。同オッズは同順位（min）。

    運用時は月次ファイルの「人気」（確定時点）が無いので、締切前のオッズから作る。
    学習に使った確定人気とは時点が違う（学習・バックテストは楽観側）。
    """
    o = pd.to_numeric(odds_win, errors="coerce")
    return o.where(o > 0).rank(method="min")
