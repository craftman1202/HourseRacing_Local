"""検証設計。"""

from .splits import Fold, OOSGuard, make_folds, make_oos_fold, validate_folds

__all__ = ["Fold", "make_folds", "make_oos_fold", "validate_folds", "OOSGuard"]
