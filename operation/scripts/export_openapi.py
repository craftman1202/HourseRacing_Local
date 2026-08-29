"""nar-api の OpenAPI スキーマを JSON に書き出す。

Web 側の型生成（openapi-typescript）の入力になる。契約（narops.api.build_app）
だけを見ればよいので、実際の BigQuery/GCS 接続は不要 — services を空のまま
build_app() を呼び、ルートを実行せずスキーマだけ取り出す。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from narops.api import build_app  # noqa: E402


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2] / "web" / "openapi.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    app = build_app()
    out_path.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
