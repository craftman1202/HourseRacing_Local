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


# α1（log p^A の係数）を自由推定するときに 1 へ引き戻す Ridge の強さ。
# 数千レースで自由度を1つ増やす代償を抑えるための事前分布に相当する（§13.1-4）。
ALPHA1_RIDGE = 50.0


@dataclass
class TrackBReport:
    n_races: int
    eta: float
    alpha: dict[str, float]
    n_estimated_params: int
    underpowered: bool
    notes: list[str]
    alpha1: float = 1.0
    free_offset_coef: bool = False


class ResidualOddsModel:
    """トラックA を固定オフセットに載せた残差モデル。

    `free_offset_coef=True` にすると Benter (1994) の市場混合モデル
    （Research.md §2.1）そのものになり、トラックA 側ロジットの係数 α1 も推定する。

        logit(π_i) = α1·log p_i^A + η·logit(q_i^market) + v_i'α

    既定は α1 = 1 固定（TB-02）。自由化はモデル比較のための**診断**として使う
    位置づけで、α1 が 1 から離れていればトラックA 自体の較正がずれている
    シグナルになる。自由度が増えるぶん過学習しやすいので、1 へ引き戻す
    Ridge を必ず掛ける（設計書 §10.2 / §13.1-4）。
    """

    def __init__(self, odds_features: list[str] | None = None,
                 free_offset_coef: bool = False,
                 alpha1_ridge: float = ALPHA1_RIDGE) -> None:
        self.odds_features = list(odds_features or [])
        self.free_offset_coef = bool(free_offset_coef)
        self.alpha1_ridge = float(alpha1_ridge)
        if self.n_estimated_params() > MAX_ESTIMATED_PARAMS:
            raise ValueError(
                f"推定パラメータが {self.n_estimated_params()} 個。"
                f"{MAX_ESTIMATED_PARAMS} 個以下に抑えてください（TB-03）。"
                "数千レースで多数のパラメータを推定すれば確実に過学習します。"
            )
        self.eta: float = 0.0
        self.alpha: np.ndarray = np.zeros(len(self.odds_features))
        self.alpha1: float = 1.0

    def _logits(self, p_a: np.ndarray, q_market: np.ndarray, v: np.ndarray,
                eta: float, alpha: np.ndarray, alpha1: float = 1.0) -> np.ndarray:
        # 既定では log p^A の係数は 1.0 固定＝学習対象に含めない（TB-02）。
        offset = alpha1 * np.log(np.clip(p_a, EPS, 1.0))
        q = np.clip(q_market, EPS, 1 - EPS)
        return offset + eta * np.log(q / (1 - q)) + (v @ alpha if v.shape[1] else 0.0)

    def fit(self, df: pd.DataFrame, p_a: np.ndarray, q_market: np.ndarray,
            race_col: str = "race_id", label_col: str = "is_win") -> "ResidualOddsModel":
        rid = df[race_col].to_numpy()
        y = df[label_col].to_numpy(float)
        v = df[self.odds_features].to_numpy(float) if self.odds_features else np.zeros((len(df), 0))
        free = self.free_offset_coef

        def objective(params: np.ndarray) -> float:
            a1 = float(params[-1]) if free else 1.0
            p = race_softmax(
                self._logits(p_a, q_market, v, params[0], params[1:1 + v.shape[1]], a1), rid)
            nll = float(-np.log(np.clip(p[y == 1], EPS, 1.0)).mean())
            # α1 を 1 から引き離すのは、それに見合う尤度の改善があるときだけ。
            return nll + (self.alpha1_ridge * (a1 - 1.0) ** 2 / max(len(df), 1) if free else 0.0)

        x0 = np.zeros(1 + v.shape[1] + (1 if free else 0))
        if free:
            x0[-1] = 1.0
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-9})
        self.eta = float(res.x[0])
        self.alpha = res.x[1:1 + v.shape[1]]
        self.alpha1 = float(res.x[-1]) if free else 1.0
        return self

    def predict_proba(self, df: pd.DataFrame, p_a: np.ndarray, q_market: np.ndarray,
                      race_col: str = "race_id") -> np.ndarray:
        v = df[self.odds_features].to_numpy(float) if self.odds_features else np.zeros((len(df), 0))
        return race_softmax(
            self._logits(p_a, q_market, v, self.eta, self.alpha, self.alpha1),
            df[race_col].to_numpy())

    def n_estimated_params(self) -> int:
        return 1 + len(self.odds_features) + (1 if self.free_offset_coef else 0)

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
        if self.free_offset_coef:
            notes.append(
                f"**α1 = {self.alpha1:.3f}**（Benter 混合モデルのトラックA 係数、"
                f"1.0 へ Ridge 収縮）。1 から大きく離れる場合はトラックA の較正ずれを疑う。")
        return TrackBReport(
            n_races=n_races, eta=self.eta,
            alpha=dict(zip(self.odds_features, self.alpha)),
            n_estimated_params=self.n_estimated_params(),
            underpowered=underpowered, notes=notes,
            alpha1=self.alpha1, free_offset_coef=self.free_offset_coef,
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
