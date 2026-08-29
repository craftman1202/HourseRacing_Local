"""正則化付き条件付きロジット（McFadden discrete choice）。

レース内で確率の総和が1になる構造を仮定に組み込むため、「1レース1勝者」という
制約と数学的に整合する。他モデルのサニティチェックの基準線。

statsmodels の条件付きロジットは正則化が弱いので、対数尤度と解析勾配を直接書いて
L-BFGS で最適化する。解析勾配は MD-03 で数値微分と突き合わせて検証している。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .base import RaceBatch, masked_softmax


class ConditionalLogit:
    def __init__(self, l2: float = 1e-4, l1: float = 0.0, fit_intercept: bool = False) -> None:
        if fit_intercept:
            # レース内 softmax では定数項が完全に打ち消えるので推定不能
            raise ValueError("条件付きロジットに切片は含められません（レース内で相殺されます）")
        self.l2 = l2
        self.l1 = l1
        self.beta: np.ndarray | None = None
        self.feature_names: tuple[str, ...] = ()
        self.result_ = None

    def _nll_and_grad(self, beta: np.ndarray, b: RaceBatch) -> tuple[float, np.ndarray]:
        scores = np.einsum("rnf,f->rn", b.x, beta)
        p = masked_softmax(scores, b.mask)
        chosen = b.y * b.mask
        nll = -np.sum(chosen * np.log(np.clip(p, 1e-300, None))) / b.n_races
        # ∂/∂β = Σ_r (E_p[x] - x_chosen)
        resid = (p - chosen) * b.mask
        grad = np.einsum("rn,rnf->f", resid, b.x) / b.n_races

        nll += self.l2 * float(beta @ beta)
        grad += 2.0 * self.l2 * beta
        if self.l1 > 0:
            # L-BFGS-B は非平滑点を扱えないので、|β| を平滑近似する
            smooth = np.sqrt(beta**2 + 1e-12)
            nll += self.l1 * float(smooth.sum())
            grad += self.l1 * beta / smooth
        return float(nll), grad

    def fit(self, batch: RaceBatch, max_iter: int = 500) -> "ConditionalLogit":
        n_feat = batch.x.shape[2]
        self.feature_names = batch.feature_names
        self.result_ = minimize(
            self._nll_and_grad, np.zeros(n_feat), args=(batch,),
            jac=True, method="L-BFGS-B", options={"maxiter": max_iter},
        )
        self.beta = self.result_.x
        return self

    def predict_proba(self, batch: RaceBatch) -> np.ndarray:
        """(n_races, max_n)。各行の有効馬にわたる総和が厳密に 1（MD-01）。"""
        if self.beta is None:
            raise RuntimeError("fit() を先に呼んでください")
        return masked_softmax(np.einsum("rnf,f->rn", batch.x, self.beta), batch.mask)

    def standard_errors(self, batch: RaceBatch) -> np.ndarray:
        """観測情報行列の逆行列から SE を出す（MD-04 の被覆率検証に使う）。"""
        if self.beta is None:
            raise RuntimeError("fit() を先に呼んでください")
        p = self.predict_proba(batch)
        n_feat = batch.x.shape[2]
        info = np.zeros((n_feat, n_feat))
        for r in range(batch.n_races):
            m = batch.mask[r]
            xr, pr = batch.x[r, m], p[r, m]
            xbar = pr @ xr
            centered = xr - xbar
            info += (centered * pr[:, None]).T @ centered
        return np.sqrt(np.diag(np.linalg.pinv(info)))

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.beta))
