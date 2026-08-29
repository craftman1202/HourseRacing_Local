"""運用モード。

設計書 §9 の段階計画をコードで強制する。モードを上げるのは人間の判断であり、
コードから自動で上がることはない。

    shadow : 推論して DB に貯めるだけ。**ベット候補も Discord 配信も生成しない**
    paper  : Discord へ推奨を流すが、ベットは仮想（is_paper=True）
    live   : 実投票を前提とした運用

既定は shadow。第1段階（2週間）で適時性・カバレッジ・skew の3点だけを検証する、
というのが設計の指示であり、いきなり配信を始める経路を作らない。

モードとは別に「配信停止」がある。skew 不一致や RF ガード発火は、モードが
paper/live でも配信を止める。良すぎる結果を信じて資金を投じるのが最も高額な失敗。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"

    @property
    def emits_bets(self) -> bool:
        """ベット候補を DB に書くか。shadow では書かない（RL-02 と同じ思想）。"""
        return self is not Mode.SHADOW

    @property
    def delivers(self) -> bool:
        """Discord へ推奨を配信するか。"""
        return self is not Mode.SHADOW

    @property
    def is_paper(self) -> bool:
        """仮想ベットか。live 以外はすべて仮想。"""
        return self is not Mode.LIVE


class ModelProvenance(str, Enum):
    """モデルが何で学習されたか。

    合成データで学習したモデルで実投票するのは、意味のない金額を賭けることになる。
    live へ上げる条件にこれを含める。
    """

    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass(frozen=True)
class OperatingState:
    mode: Mode
    provenance: ModelProvenance
    delivery_blocked: bool = False
    blocked_reason: str = ""

    @classmethod
    def from_env(cls) -> "OperatingState":
        """環境変数からモードを読む。

        変数名は `NAROPS_MODE`。他の設定（NAROPS_MODEL_ROOT など）と揃えてある。
        以前は `OPS_MODE` を読んでいたが、デプロイ側は `NAROPS_MODE` を渡して
        おり、**モード指定がまったく効いていなかった**（常に shadow）。
        黙って片方を無視すると同じ事故が起きるので、両方あって食い違う場合は
        止める。
        """
        new = os.environ.get("NAROPS_MODE")
        old = os.environ.get("OPS_MODE")
        if new and old and new.lower() != old.lower():
            raise RuntimeError(
                f"NAROPS_MODE={new} と OPS_MODE={old} が食い違っています。"
                "どちらが効くか読めない状態で運用しないでください。")
        return cls(
            mode=Mode((new or old or "shadow").lower()),
            provenance=ModelProvenance(
                os.environ.get("MODEL_PROVENANCE", "synthetic").lower()),
        )

    def validate(self) -> None:
        """モードとモデル素性の整合。

        合成データ学習のモデルで live に上げようとしたら起動を止める。
        「気づいたら本番で賭けていた」を作らない。
        """
        if self.mode is Mode.LIVE and self.provenance is ModelProvenance.SYNTHETIC:
            raise RuntimeError(
                "合成データで学習したモデルでは live モードにできません。"
                "実データで再学習し MODEL_PROVENANCE=real にしてください。")

    def can_emit_bets(self) -> bool:
        return self.mode.emits_bets and not self.delivery_blocked

    def can_deliver(self) -> bool:
        return self.mode.delivers and not self.delivery_blocked

    def blocked(self, reason: str) -> "OperatingState":
        return OperatingState(self.mode, self.provenance, True, reason)

    def describe(self) -> str:
        base = f"mode={self.mode.value} provenance={self.provenance.value}"
        return f"{base} 配信停止（{self.blocked_reason}）" if self.delivery_blocked else base
