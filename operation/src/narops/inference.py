"""推論の実体。

順序は「各モデル予測 → レース内 softmax 正規化 → 温度スケーリング →
アンサンブル重み付き幾何平均 → 期待値 → 自己インパクト補正 → Kelly」。

出力確率のレース内総和が 1.0 ± 1e-9 でなければ例外を投げて配信を止める。
**間違った推論を配信するより、届かないほうがまし**（設計書 §3.3 / IN-01）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .clock import Clock, minutes_until, to_utc
from .config import OpsConfig
from .errors import (
    InsufficientData, NormalizationError, RaceExpired, ZeroFillForbidden,
)
from .model.manifest import Manifest, verify_no_version_mix
from .shared import (
    NOMINAL_TAKEOUT, ConstrainedStacker, effective_odds, harville_place_probability,
    kelly_fraction, normalize_within_race, race_softmax,
)

TOL = 1e-9


class ScoreModel(Protocol):
    name: str

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """レース内で比較可能な生スコア（確率でなくてよい）。"""


@dataclass
class InferenceResult:
    race_id: str
    model_release: str
    track_used: str            # "A" / "A+B"
    frame: pd.DataFrame        # horse_no, p_win, p_market, ev, ev_adjusted, kelly, stake
    per_model: dict[str, np.ndarray]
    computed_at: datetime
    notes: list[str] = field(default_factory=list)

    def bet_candidates(self, min_ev: float) -> pd.DataFrame:
        return self.frame[(self.frame["ev_adjusted"] >= min_ev)
                          & (self.frame["stake_yen"] > 0)].copy()


# ---------------------------------------------------------------------- IN-03
def assert_within_window(start_ts: datetime, clock: Clock, cfg: OpsConfig) -> float:
    """発走 −13 分 ±60 秒の窓で呼ばれているか。

    発走後の呼び出しは実行しない。遅れた推論を配信するのは害しかない。
    クロックずれは保守側（早める方向）に倒す（DR-05）。
    """
    remaining = minutes_until(start_ts, clock)
    if remaining <= 0:
        raise RaceExpired(
            f"発走時刻を過ぎています（残り {remaining:.1f} 分）。推論しません。")
    lead, tol = cfg.infer_lead_minutes, cfg.infer_lead_tolerance_sec / 60.0
    if remaining > lead + tol:
        raise RaceExpired(
            f"発走まで {remaining:.1f} 分あり、推論窓（{lead}±{tol * 60:.0f}秒）より早すぎます。")
    return remaining


# ---------------------------------------------------------------------- IN-05
def assert_no_zero_fill(features: pd.DataFrame, market_columns: list[str]) -> None:
    """オッズ欠損をゼロ・平均で埋めた入力を拒否する。

    埋めて推論を続けると学習時の分布から外れる。トラックA 単独へ縮退するのが
    正しい挙動であり、埋めるのは禁止。
    """
    for col in market_columns:
        if col not in features.columns:
            continue
        s = features[col]
        if s.isna().any():
            raise ZeroFillForbidden(
                f"{col} に欠損があります。トラックB は実行せず A 単独へ縮退してください。")
        if (s == 0).any():
            raise ZeroFillForbidden(
                f"{col} にゼロ埋めの痕跡があります（オッズ 0 は存在しません）。")


# ---------------------------------------------------------------------- IN-01
def assert_normalized(p: np.ndarray, race_ids: np.ndarray) -> None:
    sums = pd.Series(p).groupby(race_ids).sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=TOL):
        worst = float((sums - 1.0).abs().max())
        raise NormalizationError(
            f"レース内確率の総和が 1 になりません（最大誤差 {worst:.3e}）。"
            "結果を書かずに中止します。")


def blend(per_model: dict[str, np.ndarray], weights: dict[str, float],
          race_ids: np.ndarray) -> np.ndarray:
    """log π ∝ Σ w_m log p^(m)。学習側と同じ制約付きスタッキング（IN-06）。"""
    names = [n for n in weights if n in per_model]
    if not names:
        raise InsufficientData("アンサンブルに使えるモデル出力がありません")
    w = np.array([weights[n] for n in names], dtype=float)
    if abs(w.sum() - 1.0) > 1e-9 or (w < 0).any():
        raise NormalizationError(f"アンサンブル重みが非負かつ総和1ではありません: {dict(zip(names, w))}")
    stacker = ConstrainedStacker(names)
    stacker.weights = w
    return stacker.predict_proba(pd.DataFrame({n: per_model[n] for n in names}), race_ids)


def market_implied(odds: np.ndarray, race_ids: np.ndarray, takeout: float) -> np.ndarray:
    raw = (1.0 - takeout) / np.asarray(odds, dtype=float)
    return normalize_within_race(raw, race_ids)


@dataclass
class PoolSizeModel:
    """プール総額の推定（設計書 §3.3）。

    当日ファイルのオッズからは逆算できないので、場×クラス×曜日の中央値テーブルを
    使う。**保守的に（小さめに）見積もる**：プールを小さく見るほど自己インパクトが
    大きく出て、期待値を過大評価しない側に倒れる。
    """

    table: dict[tuple[int, int, int], float] = field(default_factory=dict)
    default_yen: float = 1_000_000.0
    safety_factor: float = 0.7

    def estimate(self, baba_code: int, class_level: int, weekday: int) -> float:
        base = self.table.get((baba_code, class_level, weekday), self.default_yen)
        return base * self.safety_factor


def run_inference(
    *,
    race_id: str,
    features: pd.DataFrame,
    models: dict[str, ScoreModel],
    manifests: list[Manifest],
    weights: dict[str, float],
    temperature: float,
    clock: Clock,
    cfg: OpsConfig,
    odds: pd.Series | None = None,
    pool_model: PoolSizeModel | None = None,
    baba_code: int = 20,
    class_level: int = 1,
    day_budget_remaining: int | None = None,
    takeout: float = NOMINAL_TAKEOUT["単勝"],
) -> InferenceResult:
    """1レース分の推論。"""
    verify_no_version_mix(manifests)            # MP-06
    if features.empty:
        raise InsufficientData(f"{race_id}: 特徴量が空です")

    race_ids = np.full(len(features), race_id, dtype=object)
    notes: list[str] = []

    per_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        scores = np.asarray(model.score(features), dtype=float)
        if scores.shape[0] != len(features):
            raise InsufficientData(f"{name}: 予測 {scores.shape[0]} 行が頭数 {len(features)} と不一致")
        per_model[name] = race_softmax(scores, race_ids)

    p = blend(per_model, weights, race_ids)

    # IN-07: 温度スケーリング。順位を変えない変換であることは学習側で検証済み
    if temperature and temperature > 0:
        p = race_softmax(np.log(np.clip(p, 1e-12, 1.0)) / temperature, race_ids)

    assert_normalized(p, race_ids)              # IN-01

    # 複勝（上位3着以内）確率。単勝モデルしか無いので Harville 式で近似する
    # （単勝確率だけから求める、着差分布などを要求しない標準的な近似）。
    # 通知で「単勝ダメでも複勝は堅い」を読めるようにするのが目的。
    p_top3 = harville_place_probability(p, race_ids, k=3)

    out = pd.DataFrame({
        "horse_no": features["horse_no"].to_numpy(),
        "p_win": p,
        "p_top3": p_top3,
    })

    # IN-04: オッズが揃っているときだけトラックB（市場情報つき）を使う
    track_used = "A"
    if odds is not None and len(odds) == len(features) and not pd.isna(odds).any():
        if (np.asarray(odds, dtype=float) <= 0).any():
            raise ZeroFillForbidden("オッズに 0 以下の値があります（ゼロ埋めの疑い）")
        q = market_implied(odds.to_numpy(), race_ids, takeout)
        out["p_market"] = q
        out["odds_win"] = np.asarray(odds, dtype=float)
        track_used = "A+B"
    else:
        out["p_market"] = np.nan
        out["odds_win"] = np.nan
        notes.append("オッズ未取得のためトラックA 単独で算出（ゼロ埋めはしない）")

    out = _economics(out, cfg, pool_model, baba_code, class_level, clock,
                     takeout, day_budget_remaining)
    return InferenceResult(race_id, manifests[0].model_id, track_used, out,
                           per_model, clock.now(), notes)


def _economics(out: pd.DataFrame, cfg: OpsConfig, pool_model: PoolSizeModel | None,
               baba_code: int, class_level: int, clock: Clock, takeout: float,
               day_budget_remaining: int | None) -> pd.DataFrame:
    """期待値・自己インパクト補正・Kelly。すべて学習側の実装を共有する。"""
    if out["odds_win"].isna().all():
        for c in ("ev", "ev_adjusted", "kelly", "stake_yen", "pool_yen", "stake_hint_yen"):
            out[c] = np.nan if c not in ("stake_yen", "stake_hint_yen") else 0
        out["stake_yen"] = 0
        out["stake_hint_yen"] = 0
        return out

    o = out["odds_win"].to_numpy(dtype=float)
    p = out["p_win"].to_numpy(dtype=float)
    out["ev"] = p * o

    pool = (pool_model or PoolSizeModel()).estimate(
        baba_code, class_level, clock.now().weekday())
    out["pool_yen"] = pool

    # 1/4 Kelly を暫定額として、その額での自己インパクト補正後オッズで再評価する。
    # 補正前の期待値で賭け額を決めてから補正すると、常に過大な額になる。
    provisional = np.clip(kelly_fraction(p, o, cap=cfg.kelly_fraction), 0, None)
    # 100円単位に丸める前の生の Kelly 額。単勝の最低賭け金は100円なので、
    # これを下回る額は実際には賭けられない。だからといって0円と同じ扱いで
    # 通知から消すと、「なぜ EV が良いのに推奨が空欄なのか」が読めない。
    # 賭けられない理由（単位未満）と賭ける理由が無い（EV不足）を区別できるよう、
    # 丸め前の額を stake_hint_yen として別に残す（DC-05 と同じ「表示は必ず
    # 計算値と丸め規則込みで一致させる」を、実額と参考額の2列に分けて満たす）。
    raw_stake = np.minimum(provisional * cfg.max_bet_per_race, cfg.max_bet_per_race)
    stake = np.where(raw_stake < 100, 0.0, np.floor(raw_stake / 100) * 100)

    o_eff = np.array([
        effective_odds(oi, bi, pool, takeout) if bi > 0 else oi
        for oi, bi in zip(o, stake)
    ])
    out["odds_effective"] = o_eff
    out["ev_adjusted"] = p * o_eff

    # 補正後に期待値が閾値を割った買い目は落とす
    passes = out["ev_adjusted"].to_numpy() >= cfg.discord_min_ev
    stake = np.where(passes, stake, 0.0)
    raw_stake = np.where(passes, raw_stake, 0.0)
    out["kelly"] = provisional

    if day_budget_remaining is not None:
        stake = _fit_budget(stake, min(day_budget_remaining, cfg.max_bet_per_day))
    total = stake.sum()
    if total > cfg.max_bet_per_race * len(stake):
        stake = _fit_budget(stake, cfg.max_bet_per_race * len(stake))
    out["stake_yen"] = stake.astype(int)
    # 参考額は「EV は基準を満たすが、Kelly 額が最低賭け金（100円）に届かない」
    # ときだけ実額と別の値になる。budget 按分の対象外（実額ではないので
    # 予算を消費しない）。
    out["stake_hint_yen"] = np.floor(raw_stake).astype(int)
    return out


def _fit_budget(stake: np.ndarray, budget: float) -> np.ndarray:
    """予算に収まるよう按分して 100 円単位に丸める（上限は絶対に超えない）。"""
    total = stake.sum()
    if budget <= 0:
        return np.zeros_like(stake)
    if total <= budget:
        return stake
    scaled = np.floor(stake * (budget / total) / 100) * 100
    return scaled
