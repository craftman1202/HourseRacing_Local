"""統計指標。

主指標はレース内 NLL。後段の期待値計算が確率の較正精度に直接依存するため、
Top-1 精度や NDCG を主指標にすると較正が改善しないまま順位だけ整う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _codes(race_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """レース ID を 0..n-1 の整数コードにする。

    レースごとに boolean マスクを作ると O(レース数 × 行数) になり、実データ規模
    （数十万行 × 数万レース）で評価が学習より遅くなる。整数コード + bincount /
    np.maximum.at で線形に落とす。
    """
    codes, uniques = pd.factorize(np.asarray(race_ids), sort=False)
    return codes.astype(np.intp), len(uniques)


def race_softmax(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """レース内 softmax。総和が厳密に 1 になる（MD-10 / EN-05）。"""
    scores = np.asarray(scores, dtype=float)
    codes, n = _codes(race_ids)
    peak = np.full(n, -np.inf)
    np.maximum.at(peak, codes, scores)
    e = np.exp(scores - peak[codes])
    total = np.bincount(codes, weights=e, minlength=n)
    return e / total[codes]


def normalize_within_race(p: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    codes, n = _codes(race_ids)
    total = np.bincount(codes, weights=p, minlength=n)
    size = np.bincount(codes, minlength=n)
    # 総和 0 のレースは一様分布に倒す（全馬の予測が 0 になるのは実装異常だが、
    # ここで NaN を返すと下流の NLL が黙って壊れる）
    safe = np.where(total > 0, total, 1.0)
    out = p / safe[codes]
    degenerate = total[codes] <= 0
    return np.where(degenerate, 1.0 / size[codes], out)


def race_nll(p: np.ndarray, y: np.ndarray, race_ids: np.ndarray) -> float:
    """レースごとに勝ち馬の -log p を取り、レース単位で平均する。

    行単位で平均すると頭数の多いレースが軽くなるので、必ずレース単位で割る。
    """
    df = pd.DataFrame({"p": p, "y": y, "r": race_ids})
    win = df[df["y"] == 1]
    return float(-np.log(np.clip(win["p"].to_numpy(), EPS, 1.0)).mean())


def uniform_nll(race_ids: np.ndarray) -> float:
    """一様分布ベースライン = 平均 ln N（EV-01）。頭数ごとの ln N をレース平均する。"""
    n = pd.Series(race_ids).value_counts()
    return float(np.log(n.to_numpy()).mean())


def multiclass_brier(p: np.ndarray, y: np.ndarray, race_ids: np.ndarray) -> float:
    df = pd.DataFrame({"p": p, "y": y, "r": race_ids})
    return float(df.groupby("r").apply(
        lambda g: ((g["p"] - g["y"]) ** 2).sum(), include_groups=False
    ).mean())


def top_k_accuracy(p: np.ndarray, finish_pos: np.ndarray, race_ids: np.ndarray, k: int = 1) -> float:
    df = pd.DataFrame({"p": p, "pos": finish_pos, "r": race_ids})

    def hit(g: pd.DataFrame) -> float:
        picked = set(g.nlargest(k, "p").index)
        actual = set(g[g["pos"] <= k].index)
        return float(len(picked & actual) > 0) if k == 1 else len(picked & actual) / k

    return float(df.groupby("r").apply(hit, include_groups=False).mean())


def ndcg_at_k(p: np.ndarray, finish_pos: np.ndarray, race_ids: np.ndarray, k: int = 3) -> float:
    df = pd.DataFrame({"p": p, "pos": finish_pos, "r": race_ids})

    def score(g: pd.DataFrame) -> float:
        rel = np.clip(4 - g["pos"].to_numpy(), 0, 3).astype(float)
        order = np.argsort(-g["p"].to_numpy())
        disc = 1.0 / np.log2(np.arange(2, min(k, len(g)) + 2))
        dcg = (rel[order][:k] * disc).sum()
        idcg = (np.sort(rel)[::-1][:k] * disc).sum()
        return dcg / idcg if idcg > 0 else 0.0

    return float(df.groupby("r").apply(score, include_groups=False).mean())


def spearman_within_race(p: np.ndarray, finish_pos: np.ndarray, race_ids: np.ndarray) -> float:
    from scipy.stats import spearmanr

    df = pd.DataFrame({"p": p, "pos": finish_pos, "r": race_ids})

    def rho(g: pd.DataFrame) -> float:
        if len(g) < 3 or g["p"].nunique() < 2:
            return np.nan
        return spearmanr(-g["p"], g["pos"]).statistic

    return float(df.groupby("r").apply(rho, include_groups=False).dropna().mean())


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """10分位ビンの ECE。等幅ではなく等頻度で切る（低確率帯に大半が集まるため）。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def ece_by_popularity_band(
    p: np.ndarray, y: np.ndarray, popularity: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    """人気帯別 ECE（CA-05）。

    全体 ECE が良好でも穴馬帯だけ過大評価というパターンは実運用の損失に直結する。
    """
    pop = np.asarray(popularity)
    bands = {"1-3": pop <= 3, "4-7": (pop >= 4) & (pop <= 7), "8+": pop >= 8}
    return {
        name: expected_calibration_error(p[m], y[m], n_bins)
        for name, m in bands.items() if m.sum() > n_bins
    }


def summary(
    p: np.ndarray, y: np.ndarray, finish_pos: np.ndarray, race_ids: np.ndarray,
    popularity: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        "race_nll": race_nll(p, y, race_ids),
        "uniform_nll": uniform_nll(race_ids),
        "brier": multiclass_brier(p, y, race_ids),
        "top1": top_k_accuracy(p, finish_pos, race_ids, 1),
        "top3": top_k_accuracy(p, finish_pos, race_ids, 3),
        "ndcg@3": ndcg_at_k(p, finish_pos, race_ids, 3),
        "spearman": spearman_within_race(p, finish_pos, race_ids),
        "ece": expected_calibration_error(p, y),
    }
    if popularity is not None:
        out.update({f"ece_pop_{k}": v for k, v in ece_by_popularity_band(p, y, popularity).items()})
    return out


def block_bootstrap_ci(
    values: pd.Series, blocks: pd.Series, n_iter: int = 10_000,
    alpha: float = 0.05, seed: int = 0, statistic=np.mean,
) -> tuple[float, float, float]:
    """開催日をブロック単位とするブロックブートストラップ。

    レース間は同日内で相関するので i.i.d. リサンプルは CI を過小に出す。
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values.to_numpy(), "b": blocks.to_numpy()})
    groups = [g["v"].to_numpy() for _, g in df.groupby("b", sort=False)]
    n = len(groups)
    stats = np.empty(n_iter)
    for i in range(n_iter):
        pick = rng.integers(0, n, size=n)
        stats[i] = statistic(np.concatenate([groups[j] for j in pick]))
    point = float(statistic(df["v"].to_numpy()))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH 補正。年×場×指標で数百の比較を同時に見るため必須。"""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty(n)
    out[order] = adj
    return out
