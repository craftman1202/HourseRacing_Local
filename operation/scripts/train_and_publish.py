"""学習成果物から配布用リリースを作る（3ステップの薄いラッパ）。

中身は持たない。実装は次の3つに分かれていて、それぞれ単体で実行・テストできる。

  1. `nar gate-report`   学習側 Blocker の結果を実際のテスト実行から集める
  2. `nar fit-final`     OOS 直前までの全データで最終モデルを学習し、成果物を書く
  3. `narops publish-release`  成果物と証跡から manifest 付きリリースを組み立てる

以前このスクリプトは同じ処理を独自に持っていたが、較正温度を 1.0 に固定したまま
配布する、存在しない `test_gate.json` を読んでゲートが常に落ちる、gold ではなく
silver から特徴量を作り直すため評価したものと別の行列になる、といった食い違いが
あった。実装を1か所に寄せて、ここは順番を保証するだけにする。

実行:
    PYTHONPATH=src:../learning/src python scripts/train_and_publish.py \
        --release-id v2026.08.27-A
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LEARNING = Path(__file__).resolve().parents[2] / "learning"
OPERATION = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> int:
    print(f"\n$ (cd {cwd.name} && {' '.join(cmd)})", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--release-id", required=True)
    p.add_argument("--data-root", default=f"file://{LEARNING / 'data_real'}")
    p.add_argument("--artifacts", default=str(LEARNING / "artifacts"))
    p.add_argument("--final-dir", default=str(LEARNING / "artifacts" / "final"))
    p.add_argument("--model-root", default=str(OPERATION / "data" / "nar-model"))
    p.add_argument("--tabm-epochs", type=int, default=3)
    p.add_argument("--skip-gate", action="store_true",
                   help="既存の gate_report.json を使う（テストを再実行しない）")
    p.add_argument("--stage", action="store_true", help="登録せず組み立てだけ")
    args = p.parse_args()

    py = sys.executable
    env_learning = [py, "-m", "nar.cli", "--data-root", args.data_root,
                    "--artifacts", args.artifacts]

    gate = env_learning + ["gate-report"] + (["--no-run"] if args.skip_gate else [])
    if run(gate, LEARNING) != 0:
        print("Blocker が GREEN ではありません。publish を中止します。", file=sys.stderr)
        return 1

    fit = env_learning + ["fit-final", "--out", args.final_dir,
                          "--tabm-epochs", str(args.tabm_epochs)]
    if run(fit, LEARNING) != 0:
        return 1

    publish = [py, "-m", "narops.cli", "--model-root", args.model_root,
               "publish-release", "--release-id", args.release_id,
               "--final-dir", args.final_dir, "--artifacts", args.artifacts]
    if args.stage:
        publish.append("--stage")
    return run(publish, OPERATION)


if __name__ == "__main__":
    raise SystemExit(main())
