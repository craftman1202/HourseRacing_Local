"""運用側で止めるべき失敗の型。

「静かに間違った金額を賭ける」経路に直結するものは、握りつぶさずに例外にする。
"""


class OpsError(Exception):
    pass


class ArtifactIntegrityError(OpsError):
    """配布物の SHA-256 不一致・manifest 不備（MP-02 / RL-04）。"""


class FeatureSpecMismatch(OpsError):
    """推論が作る特徴量が feature_spec と一致しない（MP-03）。"""


class VersionMixError(OpsError):
    """1推論内で dataset_version の異なるモデルを混ぜた（MP-06）。"""


class StaleDataError(OpsError):
    """確定層が前日分を反映していない。fail-closed で推論しない（DB-04）。"""


class AsOfViolation(OpsError):
    """発走時刻以降のレコードが特徴量に混入した（DB-03）。"""


class SkewError(OpsError):
    """学習時と推論時の特徴量が一致しない（SK-01 / SK-03）。"""


class ZeroFillForbidden(OpsError):
    """オッズ欠損をゼロ・平均で埋めようとした（IN-05）。"""


class RaceExpired(OpsError):
    """発走時刻を過ぎたレースに対する推論要求（IN-03）。"""


class InsufficientData(OpsError):
    """出馬表未確定・枠順未定など、推論してはいけない状態（IN-09）。"""


class NormalizationError(OpsError):
    """レース内確率の総和が 1 でない（IN-01）。配信するより届かないほうがまし` 。"""


class PartitionFilterRequired(OpsError):
    """パーティションフィルタなしのクエリ（DB-09 / CO-01）。"""


class BytesBilledUnbounded(OpsError):
    """maximum_bytes_billed 未設定のジョブ（CO-01）。"""


class SecretLeak(OpsError):
    """秘密情報がログ・レスポンスに現れた（DC-01）。"""


class PublishGateFailed(OpsError):
    """学習側 Blocker が GREEN でないまま publish しようとした（RL-01）。"""


class PromotionRejected(OpsError):
    """シャドー指標が現行版に届かないまま昇格しようとした（RL-03）。"""
