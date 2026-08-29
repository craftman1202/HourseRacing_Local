"""パイプラインを止めるべき失敗の型。

silent failure を作らないため、握りつぶし可能な例外は定義しない。
"""


class NarError(Exception):
    pass


class SchemaDriftError(NarError):
    """ZIP 内 CSV のスキーマが既知値と一致しない（SG-02/03）。silver 昇格を止める。"""


class UnknownTrackError(NarError):
    """競馬場マスタに無い名称。黙って NULL にせず停止する（TR-03）。"""


class LeakageError(NarError):
    """post-race 列が pre-race スキーマに残っている（LK-04/08）。"""


class OOSAccessError(NarError):
    """--unlock-oos なしで OOS 期間を読もうとした（CV-07）。"""


class ContentTypeError(NarError):
    """HTML エラーページが 200 で返ってきた等（IG-07）。"""


class TooGoodToBeTrueError(NarError):
    """性能が良すぎる。成果ではなくバグの徴候として扱う（RF-01..08）。"""
