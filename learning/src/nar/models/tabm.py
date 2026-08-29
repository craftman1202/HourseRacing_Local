"""TabM（ICLR 2025）＋ listwise 損失。

パラメータ効率的アンサンブル。1つのMLPが k 個のMLPのアンサンブルを模倣する。
共有バックボーンに member ごとの rank-1 アダプタ（BatchEnsemble 型）を掛けることで、
k 個ぶんのパラメータを持たずに Deep Ensemble 相当の多様性とキャリブレーションを得る。

損失は素の二値交差エントロピーではなく、レース単位の softmax 交差エントロピー
（= Plackett-Luce の1位項）に置き換える。頭数可変はマスク付き softmax で扱う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval.metrics import EPS
from .base import RaceBatch

NEG_INF = -1e30


def masked_log_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """パディング位置に確率を漏らさない log_softmax。

    -inf ではなく大きな負の有限値を使う。-inf は全マスク行で NaN を生み、
    その NaN が勾配経由で全パラメータに伝播する。
    """
    return torch.log_softmax(scores.masked_fill(~mask, NEG_INF), dim=-1)


def plackett_luce_first_term(
    log_p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """レース単位の負の対数尤度。

    -log( exp(μ_win) / Σ_j exp(μ_j) ) を勝ち馬について取り、レース単位で平均する。
    これは Plackett-Luce 尤度の1位項に厳密に一致する（MD-14）。
    """
    win = (y * mask).to(log_p.dtype)
    per_race = -(log_p * win).sum(dim=-1)
    return per_race.mean()


@dataclass
class TabMConfig:
    k: int = 8                  # アンサンブル member 数
    hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_races: int = 256
    patience: int = 10
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    history: list[dict] = field(default_factory=list)


def _random_sign(*shape: int) -> torch.Tensor:
    return torch.randint(0, 2, shape).float() * 2.0 - 1.0


class _BatchEnsembleLinear(nn.Module):
    """共有 weight に member ごとの rank-1 スケーリングを掛ける層。

    出力 = ((x * r_i) W + b) * s_i + b_i。W が k 個ぶん複製されないのが要点で、
    パラメータ数は共有 MLP + O(k·d) にとどまる。
    """

    def __init__(self, d_in: int, d_out: int, k: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        # r/s を 1 で初期化すると全 member が同一の関数から始まり、勾配も同一に
        # なるため最後まで分岐しない（member 分散が 1e-6 に張り付く）。
        # BatchEnsemble の標準どおりランダム符号で対称性を破る。
        self.r = nn.Parameter(_random_sign(k, d_in))
        self.s = nn.Parameter(_random_sign(k, d_out))
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (k, n, d_in)
        return self.linear(x * self.r[:, None, :]) * self.s[:, None, :] + self.bias[:, None, :]


class _TabMNet(nn.Module):
    def __init__(self, d_in: int, cfg: TabMConfig) -> None:
        super().__init__()
        self.k = cfg.k
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(cfg.n_layers):
            layers.append(_BatchEnsembleLinear(d, cfg.hidden, cfg.k))
            d = cfg.hidden
        self.blocks = nn.ModuleList(layers)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Parameter(torch.zeros(cfg.k, cfg.hidden))
        self.head_bias = nn.Parameter(torch.zeros(cfg.k))
        nn.init.normal_(self.head, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n, d_in) → member ごとのスコア (k, n)。"""
        h = x.unsqueeze(0).expand(self.k, -1, -1)
        for block in self.blocks:
            h = self.dropout(F.relu(block(h)))
        return (h * self.head[:, None, :]).sum(-1) + self.head_bias[:, None]


class TabM:
    def __init__(self, cfg: TabMConfig | None = None) -> None:
        self.cfg = cfg or TabMConfig()
        self.net: _TabMNet | None = None
        self.feature_names: tuple[str, ...] = ()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    # ------------------------------------------------------------------ 内部処理
    def _fitted(self) -> _TabMNet:
        if self.net is None:
            raise RuntimeError("fit() を先に呼んでください")
        return self.net

    def _tensors(self, batch: RaceBatch, fit_scaler: bool = False):
        x = batch.x.reshape(-1, batch.x.shape[2])
        if fit_scaler:
            valid = batch.mask.reshape(-1)
            self._mu = x[valid].mean(axis=0)
            sd = x[valid].std(axis=0)
            sd[sd < 1e-8] = 1.0
            self._sd = sd
        if self._mu is None or self._sd is None:
            raise RuntimeError("標準化の統計量が未設定です。fit() を先に呼んでください")
        x = (x - self._mu) / self._sd
        # パディング位置は 0 に固定する。標準化後の値が残ると、パディング不変性
        # （MD-12）は softmax マスクだけに依存することになり、検証が弱くなる。
        x = x.reshape(batch.x.shape)
        x[~batch.mask] = 0.0
        dev = self.cfg.device
        return (
            torch.as_tensor(x, dtype=torch.float32, device=dev),
            torch.as_tensor(batch.y, dtype=torch.float32, device=dev),
            torch.as_tensor(batch.mask, device=dev),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        n_races, max_n, d = x.shape
        scores = self._fitted()(x.reshape(-1, d))        # (k, n_races*max_n)
        return scores.reshape(self.cfg.k, n_races, max_n)

    # ---------------------------------------------------------------------- API
    def fit(self, train: RaceBatch, valid: RaceBatch | None = None) -> "TabM":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self.feature_names = train.feature_names
        self.net = _TabMNet(train.x.shape[2], cfg).to(cfg.device)

        xt, yt, mt = self._tensors(train, fit_scaler=True)
        xv = yv = mv = None
        if valid is not None:
            xv, yv, mv = self._tensors(valid)

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        gen = torch.Generator().manual_seed(cfg.seed)

        best, best_state, bad = float("inf"), None, 0
        cfg.history = []
        for epoch in range(cfg.epochs):
            self.net.train()
            perm = torch.randperm(train.n_races, generator=gen)
            total = 0.0
            for i in range(0, train.n_races, cfg.batch_races):
                idx = perm[i:i + cfg.batch_races].to(cfg.device)
                log_p = masked_log_softmax(self._forward(xt[idx]), mt[idx])
                # member ごとに損失を取って平均する。予測を先に平均すると
                # member 間の多様性が損失に効かず、アンサンブルの意味が消える。
                loss = plackett_luce_first_term(log_p, yt[idx].expand(cfg.k, -1, -1), mt[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                total += float(loss.detach()) * len(idx)
            sched.step()

            train_loss = total / train.n_races
            row = {"epoch": epoch, "train_nll": train_loss}
            if xv is not None:
                row["valid_nll"] = self._eval_nll(xv, yv, mv)
                if row["valid_nll"] < best - 1e-5:
                    best, bad = row["valid_nll"], 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            cfg.history.append(row)
            if bad >= cfg.patience:
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval_nll(self, x, y, mask) -> float:
        self._fitted().eval()
        log_p = masked_log_softmax(self._forward(x), mask)
        # 評価は member 平均の予測に対して行う（推論時に返すものと一致させる）
        p = log_p.exp().mean(dim=0).clamp_min(EPS)
        return float(-(p.log() * (y * mask)).sum(dim=-1).mean())

    @torch.no_grad()
    def predict_members(self, batch: RaceBatch) -> np.ndarray:
        """(k, n_races, max_n)。member ごとの確率（MD-13 の検証に使う）。"""
        self._fitted().eval()
        x, _, mask = self._tensors(batch)
        return masked_log_softmax(self._forward(x), mask).exp().cpu().numpy()

    def predict_proba(self, batch: RaceBatch) -> np.ndarray:
        """member 平均。レース内総和は member ごとに 1 なので平均後も 1。"""
        p = self.predict_members(batch).mean(axis=0)
        p[~batch.mask] = 0.0
        return p

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.cfg.history)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._fitted().parameters())
