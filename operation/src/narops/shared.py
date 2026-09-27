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
from nar.eval.place import (  # noqa: E402
    estimated_place_odds, max_ev_bets, place_probability, place_slots,
)
from nar.features.builder import ASOF_FEATURES  # noqa: E402
from nar.features.shrinkage import shrink  # noqa: E402
from nar.models.ensemble import ConstrainedStacker  # noqa: E402
from nar.transform.prerace import (  # noqa: E402
    CUMULATIVE_RECORD_COLS, POST_RACE_ALL, assert_prerace, to_prerace,
)

race_softmax = _metrics.race_softmax
harville_place_probability = _metrics.harville_place_probability
normalize_within_race = _metrics.normalize_within_race
race_nll = _metrics.race_nll
expected_calibration_error = _metrics.expected_calibration_error
summary_metrics = _metrics.summary

__all__ = [
    "ASOF_FEATURES", "CUMULATIVE_RECORD_COLS", "POST_RACE_ALL", "NOMINAL_TAKEOUT",
    "ConstrainedStacker", "TemperatureScaler",
    "assert_prerace", "to_prerace", "effective_odds", "kelly_fraction",
    "implied_takeout", "inverse_odds_sum", "shrink",
    "race_softmax", "normalize_within_race", "race_nll", "harville_place_probability",
    "expected_calibration_error", "summary_metrics",
    "estimated_place_odds", "max_ev_bets", "place_probability", "place_slots",
    "learning_package_root", "learning_conf_dir", "feature_config_for",
    "track_names",
]


# 系統ごとの学習設定ディレクトリ。ばんえいは特徴量集合も対象場も別なので、
# ここを取り違えると manifest に別競技の lookback_days が載る。
_CONF_DIRS = {
    "flat": "conf",
    "banei": "conf_banei",
}


_TRACK_NAMES: dict[int, str] | None = None


def track_names() -> dict[int, str]:
    """baba_code → 競馬場名。学習側の TrackMaster（名前→コード）の逆引き。

    Discord 通知・API の両方が同じ表示名を出す必要があるので、ここを唯一の
    実装にする（api_app と service で別々に持つと、片方だけ場名を直して
    もう片方が古いままになる）。廃止場は KNOWN_BABA_CODES に含まれないので、
    未知コードは呼び出し側で `場{code}` にフォールバックさせる — 実在しない
    名前をでっち上げるより、コードのまま出すほうが安全。
    """
    global _TRACK_NAMES
    if _TRACK_NAMES is None:
        from nar.transform.keys import TrackMaster

        _TRACK_NAMES = {code: name for name, code in TrackMaster().mapping.items()}
    return _TRACK_NAMES


def learning_conf_dir(family: str = "flat") -> Path:
    """`learning/conf*` の実体パス。

    `nar.config.feature_config()` を引数なしで呼ぶと、プロセス全体の
    `NAR_CONF_DIR`（既定は平地）が効く。運用側は1プロセスで両系統を扱うので、
    必ず系統を明示して読む。
    """
    if family not in _CONF_DIRS:
        raise ValueError(f"未知のモデル系統: {family!r}（{tuple(_CONF_DIRS)} のいずれか）")
    return _LEARNING_SRC.parent / _CONF_DIRS[family]


def feature_config_for(family: str = "flat"):
    """系統に対応する FeatureConfig。"""
    from nar.config import feature_config

    return feature_config(str(learning_conf_dir(family)))


def learning_package_root() -> Path:
    """学習側パッケージの実体パス。SK-02 の呼び出しグラフ検証で使う。"""
    import nar

    return Path(nar.__file__).resolve().parent
