"""リリースの保管と current ポインタ。

`current` の切り替えは原子的に行い、読み取り中の推論は旧版で完走させる（MP-05）。
ロールバックはポインタを戻すだけなので即座に効く（MP-08）。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..errors import ArtifactIntegrityError, PromotionRejected
from .manifest import Manifest, verify_artifacts

CURRENT = "current.json"
AUDIT = "promotions.jsonl"


@dataclass
class Release:
    release_id: str
    path: Path
    manifest: Manifest


class ModelRegistry:
    """ローカル FS 上のリリース保管庫。

    GCS へ移すときは Path 操作を fsspec に差し替えるだけで済むよう、
    ディレクトリ構造は設計書の `gs://nar-model/` と同一にしてある。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        (self.root / "releases").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- 参照
    def releases(self) -> list[str]:
        d = self.root / "releases"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []

    def load(self, release_id: str, verify: bool = True) -> Release:
        path = self.root / "releases" / release_id
        if not path.is_dir():
            raise ArtifactIntegrityError(f"リリース {release_id} が存在しません")
        manifest = Manifest.read(path / "manifest.json")
        if verify:
            verify_artifacts(path, manifest)
        return Release(release_id, path, manifest)

    # -------------------------------------------------------------- MP-05
    def current_id(self) -> str | None:
        p = self.root / CURRENT
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8")).get("release_id")

    def current(self, verify: bool = True) -> Release:
        rid = self.current_id()
        if rid is None:
            raise ArtifactIntegrityError("current ポインタが設定されていません")
        return self.load(rid, verify=verify)

    def set_current(self, release_id: str, actor: str = "system", reason: str = "") -> None:
        """原子的にポインタを差し替える。

        テンポラリに書いてから os.replace で置換する。書き込み途中の
        ポインタを推論が読むと、存在しないリリースを指した状態になる。
        """
        if release_id not in self.releases():
            raise ArtifactIntegrityError(
                f"{release_id} は releases 配下に実在しません。current は実在する"
                "ディレクトリのみを指せます。")
        self.load(release_id, verify=True)      # 壊れた版へは切り替えない

        previous = self.current_id()
        target = self.root / CURRENT
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "release_id": release_id,
            "switched_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

        with (self.root / AUDIT).open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "at": datetime.now().isoformat(timespec="seconds"),
                "actor": actor, "from": previous, "to": release_id, "reason": reason,
            }, ensure_ascii=False) + "\n")

    def audit_log(self) -> list[dict]:
        p = self.root / AUDIT
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    # -------------------------------------------------------------- 保存
    def publish(self, release_id: str, src_dir: str | Path) -> Release:
        dst = self.root / "releases" / release_id
        if dst.exists():
            raise ArtifactIntegrityError(
                f"{release_id} は既に存在します。リリースは不変にしてください。")
        shutil.copytree(src_dir, dst)
        return self.load(release_id, verify=True)

    def rollback(self, actor: str = "system") -> str:
        """直前のリリースへ戻す（MP-08）。"""
        history = [e for e in self.audit_log() if e.get("from")]
        if not history:
            raise PromotionRejected("戻せる直前リリースの記録がありません")
        previous = history[-1]["from"]
        self.set_current(previous, actor=actor, reason="rollback")
        return previous
