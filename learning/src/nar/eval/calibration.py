"""キャリブレーション。

順位を変えない変換であること（CA-06）と、確率総和が保たれること（CA-01/03）を
不変条件として持つ。等張回帰は総和制約を壊すので、必ずレース内で再正規化する。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import EPS, normalize_within_race, race_nll


class TemperatureScaler:
    """単一パラメータ T。valid 分割でのみ最適化する（CA-02）。"""

    def __init__(self) -> None:
        self.temperature: float | None = None

    def fit(self, log_p: np.ndarray, y: np.ndarray, race_ids: np.ndarray) -> "TemperatureScaler":
        def objective(log_t: float) -> float:
            p = self._apply(log_p, race_ids, float(np.exp(log_t)))
            return race_nll(p, y, race_ids)

        res = minimize_scalar(objective, bounds=(np.log(0.2), np.log(5.0)), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, log_p: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError("fit() を valid 分割で先に呼んでください")
        return self._apply(log_p, race_ids, self.temperature)

    @staticmethod
    def _apply(log_p: np.ndarray, race_ids: np.ndarray, t: float) -> np.ndarray:
        from .metrics import race_softmax

        return race_softmax(np.asarray(log_p, dtype=float) / t, race_ids)


class IsotonicRaceCalibrator:
    """等張回帰 + レース内再正規化。単調なので Top-1 は変わらない。"""

    def __init__(self) -> None:
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicRaceCalibrator":
        self._iso.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, p: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
        return normalize_within_race(self._iso.predict(np.asarray(p, dtype=float)), race_ids)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Reliability Diagram 用の (予測平均, 実測平均, 件数)。等頻度ビン。"""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows
