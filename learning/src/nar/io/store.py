"""ストレージ抽象。

パスは常に fsspec 互換 URI で扱う。DATA_ROOT が file:// でも gs:// でも
呼び出し側のコードは変わらない（設計書 §1 / RP-06）。
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import IO, Iterator

import fsspec

_LAYERS = ("raw", "bronze", "silver", "gold", "meta")


class Store:
    def __init__(self, root: str | None = None) -> None:
        root = root or os.environ.get("DATA_ROOT") or "file://./data"
        if "://" not in root:
            root = f"file://{os.path.abspath(root)}"
        self.root = root.rstrip("/")
        self.fs, self._base = fsspec.core.url_to_fs(self.root)

    def uri(self, *parts: str) -> str:
        return "/".join([self.root, *(p.strip("/") for p in parts)])

    def path(self, *parts: str) -> str:
        return "/".join([self._base.rstrip("/"), *(p.strip("/") for p in parts)])

    def exists(self, *parts: str) -> bool:
        return self.fs.exists(self.path(*parts))

    def ls(self, *parts: str) -> list[str]:
        p = self.path(*parts)
        return sorted(self.fs.find(p)) if self.fs.exists(p) else []

    def read_bytes(self, *parts: str) -> bytes:
        with self.fs.open(self.path(*parts), "rb") as f:
            return f.read()

    def write_atomic(self, data: bytes, *parts: str) -> str:
        """テンポラリに書いてから rename。中断しても不完全ファイルが残らない（IG-08）。"""
        target = self.path(*parts)
        tmp = f"{target}.tmp"
        parent = target.rsplit("/", 1)[0]
        self.fs.makedirs(parent, exist_ok=True)
        try:
            with self.fs.open(tmp, "wb") as f:
                f.write(data)
            self.fs.mv(tmp, target)
        except BaseException:
            if self.fs.exists(tmp):
                self.fs.rm(tmp)
            raise
        return target

    @contextmanager
    def open(self, *parts: str, mode: str = "rb") -> Iterator[IO[bytes]]:
        if "w" in mode:
            self.fs.makedirs(self.path(*parts).rsplit("/", 1)[0], exist_ok=True)
        with self.fs.open(self.path(*parts), mode) as f:
            yield f

    def ensure_layout(self) -> None:
        for layer in _LAYERS:
            self.fs.makedirs(self.path(layer), exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(fs, path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with fs.open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
