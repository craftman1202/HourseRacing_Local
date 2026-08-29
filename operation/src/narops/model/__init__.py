"""モデル配布の契約。"""

from .manifest import FeatureSpec, Manifest, verify_artifacts, verify_feature_spec
from .registry import ModelRegistry, Release

__all__ = ["Manifest", "FeatureSpec", "verify_artifacts", "verify_feature_spec",
           "ModelRegistry", "Release"]
