"""ZIP 展開と CP932 デコード。

bronze を silver から分けている理由がここにある。展開とデコードだけを済ませた層が
あると「NAR の仕様が変わったのか自分のパーサがバグったのか」を即座に切り分けられる。
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass

log = logging.getLogger(__name__)

# 実測（2026-07 の月次ファイル）では **UTF-8 BOM 付き**。設計書は CP932 を第一候補と
# しているが、cp932 は UTF-8 バイト列を例外なく「復号」してしまい、文字化けしたまま
# 気付けない。BOM を先に見て確定させ、無い場合だけ候補を順に試す。
DEFAULT_CANDIDATES = ("utf-8-sig", "cp932")
_BOM_CODECS = ((b"\xef\xbb\xbf", "utf-8-sig"),
               (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))


@dataclass
class DecodedCsv:
    name: str
    text: str
    codec: str
    replaced: bool


def inner_names(zip_bytes: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return sorted(n for n in zf.namelist() if not n.endswith("/"))


def decode(raw: bytes, candidates: tuple[str, ...] = DEFAULT_CANDIDATES) -> tuple[str, str, bool]:
    """BOM 判定 → 候補順 → replace フォールバック。

    BOM を最優先にするのが要点。候補順だけに頼ると、cp932 が UTF-8 バイト列を
    例外なく復号してしまい（文字化けするだけ）、間違ったまま先へ進む。

    1998年前後のファイルには外字や機種依存文字が混入している。ここで例外を投げると
    30年分のバックフィルが1バイトで止まるので、最後は警告を出して通す（IG-10）。
    """
    for bom, codec in _BOM_CODECS:
        if raw.startswith(bom):
            try:
                return raw.decode(codec), codec, False
            except UnicodeDecodeError:
                break
    for codec in candidates:
        try:
            return raw.decode(codec), codec, False
        except UnicodeDecodeError:
            continue
    log.warning("デコード不能なバイトを replace で通しました（%d bytes）", len(raw))
    return raw.decode("cp932", errors="replace"), "cp932/replace", True


def extract(zip_bytes: bytes, candidates: tuple[str, ...] = DEFAULT_CANDIDATES) -> list[DecodedCsv]:
    out: list[DecodedCsv] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            text, codec, replaced = decode(zf.read(name), candidates)
            # ZIP エントリ名自体も CP932。ここを取り違えると file_key が化ける
            display = name
            try:
                display = name.encode("cp437").decode("cp932")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            out.append(DecodedCsv(display, text, codec, replaced))
    return out
