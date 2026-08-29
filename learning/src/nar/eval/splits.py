"""purged walk-forward 分割と OOS ロック。

分割の単位はレースであり、同一レースが train と valid に分かれることは無い。
同一開催日をまたがせないことも保証する（同一日の馬は結果が相互依存するため）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..config import CVConfig
from ..errors import OOSAccessError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    embargo_days: int
    name: str = "fold"

    def mask(self, dates: pd.Series) -> tuple[pd.Series, pd.Series]:
        d = pd.to_datetime(dates).dt.date
        train = (d >= self.train_start) & (d <= self.train_end)
        valid = (d >= self.valid_start) & (d <= self.valid_end)
        return train, valid


def make_folds(cfg: CVConfig, data_end: date | None = None) -> list[Fold]:
    """embargo は cfg.embargo_days（= features.max_lookback_days）から引く。

    履歴系特徴量が学習期間末尾の情報を含むため、embargo を最大ルックバック窓と
    等しくしないと train の末尾が valid の特徴量に滲む（CV-03/04）。
    """
    folds: list[Fold] = []
    for i, (vs, ve) in enumerate(cfg.folds, start=1):
        train_end = vs - timedelta(days=cfg.embargo_days + 1)
        train_start = (
            cfg.train_start if cfg.mode == "expanding"
            else max(cfg.train_start, train_end - timedelta(days=365 * cfg.rolling_window_years))
        )
        if train_start >= train_end:
            raise ValueError(
                f"fold {i}: embargo {cfg.embargo_days} 日が長すぎて学習期間が消えました。"
            )
        folds.append(Fold(i, train_start, train_end, vs, ve, cfg.embargo_days))
    return folds


def make_oos_fold(cfg: CVConfig, data_end: date | None = None) -> Fold:
    vs, ve = cfg.oos
    ve = ve or data_end or date.today()
    train_end = vs - timedelta(days=cfg.embargo_days + 1)
    return Fold(0, cfg.train_start, train_end, vs, ve, cfg.embargo_days, name="oos")


def inner_folds(outer: Fold, n: int, embargo_days: int) -> list[Fold]:
    """ネストHPO の内側。外側 train 期間内に完全に収まる（CV-09）。"""
    span = (outer.train_end - outer.train_start).days
    step = span // (n + 1)
    if step <= embargo_days:
        raise ValueError(
            f"内側 {n}-fold には期間が足りません（1区間 {step} 日 <= embargo {embargo_days} 日）。"
        )
    out: list[Fold] = []
    for k in range(1, n + 1):
        v_start = outer.train_start + timedelta(days=step * k)
        v_end = min(v_start + timedelta(days=step - 1), outer.train_end)
        t_end = v_start - timedelta(days=embargo_days + 1)
        out.append(Fold(k, outer.train_start, t_end, v_start, v_end, embargo_days, "inner"))
    return out


class OOSGuard:
    """OOS を覗いた時点で全評価が無効になる。読み出しをコードレベルで塞ぐ（CV-07/08）。"""

    def __init__(self, cfg: CVConfig, ledger_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self._unlocked = False
        self.ledger_path = Path(ledger_path) if ledger_path else None

    def unlock(self, reason: str) -> None:
        self._unlocked = True
        self._record(reason)

    def _record(self, reason: str) -> None:
        entry = {"unlocked_at": pd.Timestamp.utcnow().isoformat(), "reason": reason}
        log.warning("OOS を開封しました: %s", reason)
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def access_count(self) -> int:
        if not self.ledger_path or not self.ledger_path.exists():
            return 0
        return sum(1 for line in self.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip())

    def filter(self, df: pd.DataFrame, date_col: str = "race_date") -> pd.DataFrame:
        """OOS 期間を落とした DataFrame を返す。開封済みならそのまま通す。"""
        if self._unlocked or not self.cfg.oos_locked:
            return df
        start, end = self.cfg.oos
        d = pd.to_datetime(df[date_col]).dt.date
        hit = (d >= start) & ((d <= end) if end else True)
        if hit.any():
            return df[~hit].copy()
        return df

    def assert_readable(self, df: pd.DataFrame, date_col: str = "race_date") -> None:
        if self._unlocked or not self.cfg.oos_locked:
            return
        start, end = self.cfg.oos
        d = pd.to_datetime(df[date_col]).dt.date
        hit = (d >= start) & ((d <= end) if end else True)
        if hit.any():
            raise OOSAccessError(
                f"OOS 期間（{start} 以降）のデータ {int(hit.sum())} 行に触れようとしました。"
                "最終評価スクリプトから --unlock-oos 付きでのみ読めます。"
            )


def validate_folds(folds: list[Fold], races: pd.DataFrame) -> None:
    """CV-01/02/03/05/06 をまとめて検証する。分割は作ったら必ずこれを通す。"""
    prev_train: set[str] | None = None
    for f in folds:
        tr, va = f.mask(races["race_date"])
        tr_ids, va_ids = set(races.loc[tr, "race_id"]), set(races.loc[va, "race_id"])
        if tr_ids & va_ids:
            raise ValueError(f"fold {f.index}: race_id が train/valid に重複（{len(tr_ids & va_ids)} 件）")

        d = pd.to_datetime(races["race_date"]).dt.date
        tr_days, va_days = set(d[tr]), set(d[va])
        if tr_days & va_days:
            raise ValueError(f"fold {f.index}: 同一開催日が train/valid にまたがっています")
        if tr_days and va_days:
            gap = (min(va_days) - max(tr_days)).days
            if gap < f.embargo_days:
                raise ValueError(
                    f"fold {f.index}: embargo が {gap} 日しかありません（要 {f.embargo_days} 日）"
                )
            if max(tr_days) >= min(va_days):
                raise ValueError(f"fold {f.index}: train が valid より後の日付を含みます")
        if prev_train is not None and f.name == "fold" and not prev_train <= tr_ids:
            raise ValueError(f"fold {f.index}: 拡張窓なのに前 fold の train を包含していません")
        prev_train = tr_ids
