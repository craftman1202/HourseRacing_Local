"""推論の実体。

順序は「各モデル予測 → レース内 softmax 正規化 → モデルごとの温度スケーリング
（学習側 walk-forward と同じ順序、2026-09-17 修正） → アンサンブル重み付き
幾何平均 → 期待値 → 自己インパクト補正 → Kelly」。manifest にモデルごとの
温度が無い旧リリースだけ、合成後に1回だけ温度をかける旧経路にフォールバック
する（Design_LogicFlow.md §5-3）。

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
    NOMINAL_TAKEOUT, ConstrainedStacker, effective_odds, estimated_place_odds,
    harville_place_probability, kelly_fraction, max_ev_bets, normalize_within_race,
    place_probability, place_slots, race_softmax,
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
        # 「EV の高い方を1点」経路では賭ける券種の EV が ev_bet に入る（複勝を買う馬は
        # 単勝の EV が閾値未満でありうる）。賭け金を決めた時点で閾値判定は済んでいる。
        if "ev_bet" in self.frame.columns:
            return self.frame[self.frame["stake_yen"] > 0].copy()
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
    model_temperatures: dict[str, float] | None = None,
    place: Any = None,
    place_odds: pd.DataFrame | None = None,
    strategy: Any = None,
) -> InferenceResult:
    """1レース分の推論。

    `place`（runtime.PlaceModels）と `strategy`（config.StrategyConfig）が両方あるときは
    複勝専用モデルで P(複勝) を出し、「単勝・複勝の EV の高い方を1点」で賭け金を決める。
    `place_odds` は features と同じ行順の pl_min / pl_max。無ければ単勝だけで判定する。
    """
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

    # IN-07: 温度スケーリング。
    # 学習側（train/pipeline.py::run_fold）は「モデルごとに温度をかけてから
    # アンサンブル重みを推定する」順序で較正を検証している。旧実装はここが
    # 逆で、モデルを重み付き合成した後に1個の温度（重み最大モデルのもの）を
    # かけていた（Design_LogicFlow.md §5-3、学習と推論の不一致）。
    # manifest に model_temperatures（モデルごとの温度）があれば学習と同じ
    # 順序で個別に較正する。旧リリース（この dict が空）は挙動を変えない
    # ため、従来どおり合成後に1回だけ掛ける経路にフォールバックする。
    model_temperatures = model_temperatures or {}
    if model_temperatures:
        for name in list(per_model):
            t = model_temperatures.get(name)
            if t and t > 0:
                per_model[name] = race_softmax(
                    np.log(np.clip(per_model[name], 1e-12, 1.0)) / t, race_ids)
        p = blend(per_model, weights, race_ids)
    else:
        p = blend(per_model, weights, race_ids)
        if temperature and temperature > 0:
            p = race_softmax(np.log(np.clip(p, 1e-12, 1.0)) / temperature, race_ids)

    assert_normalized(p, race_ids)              # IN-01

    # 複勝（上位3着以内）確率。単勝モデルしか無いので Harville 式で近似する
    # （単勝確率だけから求める、着差分布などを要求しない標準的な近似）。
    # 通知で「単勝ダメでも複勝は堅い」を読めるようにするのが目的。
    use_place = place is not None and strategy is not None
    if use_place:
        k = place_slots(np.full(len(features), len(features)))
        place_scores = {name: np.asarray(m.score(features), dtype=float)
                        for name, m in place.models.items()}
        p_top3 = place_probability(place_scores, place.temperatures, place.weights,
                                   place.ensemble_temperature, race_ids, k)
        if not np.isclose(p_top3.sum(), k[0], atol=1e-6):
            raise NormalizationError(
                f"複勝確率の総和 {p_top3.sum():.6f} が複勝枠 {k[0]} と一致しません")
    else:
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

    if use_place:
        out["p_place"] = p_top3
        if (place_odds is not None and len(place_odds) == len(features)
                and {"pl_min", "pl_max"} <= set(place_odds.columns)):
            out["pl_min"] = place_odds["pl_min"].to_numpy(dtype=float)
            out["pl_max"] = place_odds["pl_max"].to_numpy(dtype=float)
        else:
            out["pl_min"] = np.nan
            out["pl_max"] = np.nan
            notes.append("複勝オッズ未取得のため単勝だけで判定")
        out = _economics_max_ev(out, strategy, race_ids, cfg, day_budget_remaining)
    else:
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

    # 上限は「1レースの合計」に掛ける。設計書 §3.3 と Discord の「買い目合計」
    # 表示はどちらもレース単位の合計を指しており、config の `max_per_race` も
    # その意味で書かれている。
    #
    # 以前は `max_bet_per_race * 頭数` を上限にしていた。ただし**現状これは
    # 到達しない条件**だった: 単勝のみ・Σp=1 なら Σf* < 1 なので、レース合計は
    # 0.25（Kelly 係数）× max_bet_per_race = ¥750 が上界で、¥3,000 にすら届かない
    # （実測でも最大 ¥600）。つまり実害の出ていた不具合ではなく、意味を持たない
    # 上限式だった。合計に掛け直すのは、式が名前と設計の意味に一致していないと
    # 次の変更で事故になるため。
    #
    # 実際に効き始めるのは Design_Operation.md §3.5 の同時ポートフォリオ最適化を
    # 入れたときで、1レースに複数券種の買い目が並ぶと Σf* は 1 を超えうる。
    #
    # レース合計 → 1日残枠 の順に掛ける。_fit_budget は縮める方向にしか働かない
    # ので、この順序なら両方の上限を必ず満たす。
    stake = _fit_budget(stake, cfg.max_bet_per_race)
    if day_budget_remaining is not None:
        stake = _fit_budget(stake, min(day_budget_remaining, cfg.max_bet_per_day))
    out["stake_yen"] = stake.astype(int)
    # 参考額は「EV は基準を満たすが、Kelly 額が最低賭け金（100円）に届かない」
    # ときだけ実額と別の値になる。budget 按分の対象外（実額ではないので
    # 予算を消費しない）。
    out["stake_hint_yen"] = np.floor(raw_stake).astype(int)
    return out


def _economics_max_ev(out: pd.DataFrame, strategy, race_ids: np.ndarray, cfg: OpsConfig,
                      day_budget_remaining: int | None) -> pd.DataFrame:
    """単勝・複勝の EV の高い方を1点（判定と金額は nar.eval.place.max_ev_bets の単一実装）。

    自己インパクト補正はしない（戦略の検証が補正なしのため）。ev_adjusted には
    単勝 EV をそのまま入れる — prediction テーブルの列の意味（単勝の期待値）を変えない。
    1日の上限（betting.max_per_day）だけは従来どおり掛ける。
    """
    out["pl_est"] = estimated_place_odds(out["pl_min"], out["pl_max"],
                                         strategy.place_odds_alpha)
    out["pool_yen"] = np.nan
    out["odds_effective"] = out["odds_win"]
    out["stake_hint_yen"] = 0
    if out["odds_win"].isna().all():
        out["ev"] = np.nan
        out["ev_adjusted"] = np.nan
        out["ev_place"] = out["p_place"] * out["pl_est"]
        out["ev_bet"] = np.nan
        out["bet_type"] = None
        out["kelly"] = 0.0
        out["stake_yen"] = 0
        return out

    bets = max_ev_bets(race_ids, out["p_win"], out["odds_win"], out["p_place"], out["pl_est"],
                       min_ev=strategy.min_ev, min_prob=strategy.min_prob,
                       budget_win=strategy.budget_win_per_race,
                       budget_place=strategy.budget_place_per_race,
                       kelly_scale=strategy.kelly_scale)
    out["ev"] = bets["ev_win"].to_numpy()
    out["ev_adjusted"] = bets["ev_win"].to_numpy()
    out["ev_place"] = bets["ev_place"].to_numpy()
    out["kelly"] = bets["kelly"].to_numpy()
    stake = bets["stake_yen"].to_numpy(dtype=float)
    if day_budget_remaining is not None:
        stake = _fit_budget(stake, min(day_budget_remaining, cfg.max_bet_per_day))
    out["stake_yen"] = stake.astype(int)
    out["bet_type"] = np.where(out["stake_yen"] > 0, bets["bet_type"].to_numpy(), None)
    out["ev_bet"] = np.where(out["stake_yen"] > 0, bets["ev_chosen"].to_numpy(), np.nan)
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
