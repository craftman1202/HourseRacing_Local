"""トラックB（オッズ使用）。

この設計の最大の弱点。オッズは2026年3月以降しか存在せず、walk-forward 5-fold を
組むと 1 fold が1か月程度になり統計的検出力が絶望的に不足する。

そこで独立モデルではなく、トラックA の出力を**固定オフセット**とする残差モデルにする。

    logit(π_i) = log p_i^A + η·logit(q_i^market) + v_i'α

推定するのは η と数個の α だけ。数千レースでも安定する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..eval.metrics import EPS, race_softmax

MAX_ESTIMATED_PARAMS = 10


@dataclass
class TrackBReport:
    n_races: int
    eta: float
    alpha: dict[str, float]
    n_estimated_params: int
    underpowered: bool
    notes: list[str]


class ResidualOddsModel:
    """トラックA を固定オフセットに載せた残差モデル。"""

    def __init__(self, odds_features: list[str] | None = None) -> None:
        self.odds_features = list(odds_features or [])
        if 1 + len(self.odds_features) > MAX_ESTIMATED_PARAMS:
            raise ValueError(
                f"推定パラメータが {1 + len(self.odds_features)} 個。"
                f"{MAX_ESTIMATED_PARAMS} 個以下に抑えてください（TB-03）。"
                "数千レースで多数のパラメータを推定すれば確実に過学習します。"
            )
        self.eta: float = 0.0
        self.alpha: np.ndarray = np.zeros(len(self.odds_features))

    def _logits(self, p_a: np.ndarray, q_market: np.ndarray, v: np.ndarray,
                eta: float, alpha: np.ndarray) -> np.ndarray:
        # log p^A の係数は 1.0 に固定。学習対象パラメータに含めない（TB-02）
        offset = np.log(np.clip(p_a, EPS, 1.0))
        q = np.clip(q_market, EPS, 1 - EPS)
        return offset + eta * np.log(q / (1 - q)) + (v @ alpha if v.shape[1] else 0.0)

    def fit(self, df: pd.DataFrame, p_a: np.ndarray, q_market: np.ndarray,
            race_col: str = "race_id", label_col: str = "is_win") -> "ResidualOddsModel":
        rid = df[race_col].to_numpy()
        y = df[label_col].to_numpy(float)
        v = df[self.odds_features].to_numpy(float) if self.odds_features else np.zeros((len(df), 0))

        def objective(params: np.ndarray) -> float:
            p = race_softmax(self._logits(p_a, q_market, v, params[0], params[1:]), rid)
            return float(-np.log(np.clip(p[y == 1], EPS, 1.0)).mean())

        res = minimize(objective, np.zeros(1 + v.shape[1]), method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9})
        self.eta, self.alpha = float(res.x[0]), res.x[1:]
        return self

    def predict_proba(self, df: pd.DataFrame, p_a: np.ndarray, q_market: np.ndarray,
                      race_col: str = "race_id") -> np.ndarray:
        v = df[self.odds_features].to_numpy(float) if self.odds_features else np.zeros((len(df), 0))
        return race_softmax(
            self._logits(p_a, q_market, v, self.eta, self.alpha), df[race_col].to_numpy())

    def n_estimated_params(self) -> int:
        return 1 + len(self.odds_features)

    def report(self, df: pd.DataFrame, race_col: str = "race_id") -> TrackBReport:
        """TB-01/05: 検出力不足と参考値扱いの注記を自動で付ける。"""
        n_races = int(df[race_col].nunique())
        notes: list[str] = []
        underpowered = n_races < 5000
        if underpowered:
            notes.append(
                f"**統計的検出力不足**: オッズ利用可能レースが {n_races:,} 件（5,000件未満）。"
                "この規模で特徴量選択や HPO を回せば確実に過学習します。")
        notes.append("**参考値**: トラックB の結果は 2年分のオッズが蓄積するまで信頼できません。")
        notes.append(
            "本格運用の判断はオッズが2年分（2028年春）貯まってから行うのが誠実な判断です。")
        return TrackBReport(
            n_races=n_races, eta=self.eta,
            alpha=dict(zip(self.odds_features, self.alpha)),
            n_estimated_params=self.n_estimated_params(),
            underpowered=underpowered, notes=notes,
        )


def market_implied(df: pd.DataFrame, odds_col: str = "odds_win",
                   race_col: str = "race_id", takeout: float = 0.20) -> np.ndarray:
    """市場暗黙確率。レース内で正規化する（Σq=1）。"""
    raw = (1.0 - takeout) / df[odds_col].to_numpy(float)
    total = pd.Series(raw).groupby(df[race_col].to_numpy()).transform("sum").to_numpy()
    return raw / total


def snapshot_gaps(captured_days: pd.Series, max_gap_days: int = 3) -> pd.DataFrame:
    """日次オッズ蓄積の欠測検出（TB-07）。

    締切直前オッズの時系列は月次ファイルには無い情報で、今から貯めないと
    永久に手に入らない。欠測が連続したら気付ける状態にしておく。
    """
    days = pd.to_datetime(pd.Series(captured_days)).dt.normalize().drop_duplicates().sort_values()
    gaps = days.diff().dt.days
    hit = gaps >= max_gap_days
    return pd.DataFrame({
        "gap_start": days.shift(1)[hit].dt.date,
        "gap_end": days[hit].dt.date,
        "gap_days": gaps[hit].astype(int),
    }).reset_index(drop=True)


def align_to_odds_period(*frames: pd.DataFrame, odds_col: str = "odds_win") -> list[pd.DataFrame]:
    """TB-06: 比較評価をオッズ利用可能な同一レース集合に揃える。

    トラックA を全期間、トラックB を6か月で評価して優劣を論じるのは無意味。
    """
    base = frames[0]
    usable = set(base.loc[base[odds_col].notna(), "race_id"])
    for f in frames[1:]:
        usable &= set(f.loc[f[odds_col].notna(), "race_id"]) if odds_col in f else set(f["race_id"])
    return [f[f["race_id"].isin(usable)].copy() for f in frames]
