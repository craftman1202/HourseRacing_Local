"""学習側パッケージ（`nar`）への唯一の入口。

SK-02 は「学習時と推論時が同一モジュール・同一関数を呼ぶ／重複実装の存在で失敗」を
要求する。運用側で to_prerace / 期待値計算 / Kelly / 自己インパクト補正を書き直すと、
Web と Discord に出る数値が学習時の定義とズレる — それが最悪のバグになる。

そこで運用側からは**再実装を一切持たず**、ここを経由して学習側の実体を借りる。
このモジュールが `nar.*` を re-export する唯一の場所であり、テストは
「narops の他モジュールが nar を直接 import していないこと」ではなく
「同一関数オブジェクトを指していること」で同一性を検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 学習側は隣のディレクトリにある。editable install していない環境でも
# 動くようにパスを通す。二重に通さないよう存在チェックしてから追加する。
_LEARNING_SRC = Path(__file__).resolve().parents[3] / "learning" / "src"
if _LEARNING_SRC.is_dir() and str(_LEARNING_SRC) not in sys.path:
    sys.path.insert(0, str(_LEARNING_SRC))

from nar.eval import metrics as _metrics  # noqa: E402
from nar.eval.calibration import TemperatureScaler  # noqa: E402
from nar.eval.economic import (  # noqa: E402
    NOMINAL_TAKEOUT, effective_odds, implied_takeout, inverse_odds_sum, kelly_fraction,
)
from nar.features.builder import ASOF_FEATURES  # noqa: E402
from nar.features.shrinkage import shrink  # noqa: E402
from nar.models.ensemble import ConstrainedStacker  # noqa: E402
from nar.transform.prerace import (  # noqa: E402
    CUMULATIVE_RECORD_COLS, POST_RACE_ALL, assert_prerace, to_prerace,
)

race_softmax = _metrics.race_softmax
normalize_within_race = _metrics.normalize_within_race
race_nll = _metrics.race_nll
expected_calibration_error = _metrics.expected_calibration_error
summary_metrics = _metrics.summary

__all__ = [
    "ASOF_FEATURES", "CUMULATIVE_RECORD_COLS", "POST_RACE_ALL", "NOMINAL_TAKEOUT",
    "ConstrainedStacker", "TemperatureScaler",
    "assert_prerace", "to_prerace", "effective_odds", "kelly_fraction",
    "implied_takeout", "inverse_odds_sum", "shrink",
    "race_softmax", "normalize_within_race", "race_nll",
    "expected_calibration_error", "summary_metrics",
    "learning_package_root",
]


def learning_package_root() -> Path:
    """学習側パッケージの実体パス。SK-02 の呼び出しグラフ検証で使う。"""
    import nar

    return Path(nar.__file__).resolve().parent
