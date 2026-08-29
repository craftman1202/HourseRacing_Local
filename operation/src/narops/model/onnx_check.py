"""ONNX 変換後の等価性検証（MP-07）。

TabM を ONNX にするのは Cloud Run から PyTorch（数百MB）を排除して
コールドスタートを短縮するため。変換後は必ず PyTorch 版との出力一致を
検証してから公開する。

最大絶対差だけでは不十分で、**Top-1 の一致率も見る**。差が小さくても順位が
入れ替われば買い目が変わり、それは別のモデルを配信しているのと同じ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_ABS_DIFF = 1e-5


@dataclass
class EquivalenceResult:
    max_abs_diff: float
    top1_match: float
    n_races: int
    tolerance: float = MAX_ABS_DIFF

    @property
    def passed(self) -> bool:
        return self.max_abs_diff < self.tolerance and self.top1_match == 1.0

    def describe(self) -> str:
        return (f"最大絶対差 {self.max_abs_diff:.3e}（許容 {self.tolerance:.0e}） / "
                f"Top-1 一致率 {self.top1_match * 100:.2f}% / {self.n_races} レース")


def compare_outputs(original: np.ndarray, converted: np.ndarray,
                    race_sizes: list[int], tolerance: float = MAX_ABS_DIFF
                    ) -> EquivalenceResult:
    """元モデルと変換後の出力を突き合わせる。

    race_sizes はレースごとの頭数。Top-1 はレース単位で比較する。
    """
    a = np.asarray(original, dtype=float).reshape(len(original), -1)
    b = np.asarray(converted, dtype=float).reshape(len(converted), -1)
    if a.shape != b.shape:
        raise ValueError(f"出力の形が違います: {a.shape} vs {b.shape}")
    if sum(race_sizes) != len(a):
        raise ValueError(f"race_sizes の合計 {sum(race_sizes)} が行数 {len(a)} と不一致")

    max_diff = float(np.max(np.abs(a - b))) if a.size else 0.0

    matches = 0
    offset = 0
    for n in race_sizes:
        sa, sb = a[offset:offset + n, 0], b[offset:offset + n, 0]
        matches += int(np.argmax(sa) == np.argmax(sb))
        offset += n
    top1 = matches / len(race_sizes) if race_sizes else 1.0
    return EquivalenceResult(max_diff, top1, len(race_sizes), tolerance)


def assert_equivalent(original: np.ndarray, converted: np.ndarray,
                      race_sizes: list[int]) -> EquivalenceResult:
    from ..errors import ArtifactIntegrityError

    res = compare_outputs(original, converted, race_sizes)
    if not res.passed:
        raise ArtifactIntegrityError(
            f"ONNX 変換後の出力が元モデルと一致しません。{res.describe()}")
    return res
