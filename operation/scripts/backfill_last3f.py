"""確定層に `last3f`（上がり3F）を後から埋める一度きりの移行スクリプト。

## なぜ専用スクリプトが要るか

`h_pace_bal_last3`（過去走のペースバランス）を特徴量に足したことで、確定層に
`last3f` が必要になった。列の追加自体は `bq update` で済むが、**既存 400 万行の
値は NULL のまま**で、そのまま新モデルを本番に出すと学習時は値があり推論時だけ
全行 NULL という最悪の train-serving skew になる。

通常の `narops load-history` は使えない。`db/merge.py::merge_final` の更新経路は
変更行を `(race_id, horse_no)` のタプル列挙で DELETE する実装で、「1日あたり
数千行の訂正」を前提にしている。今回は 260 万行が変更対象になるため、
BigQuery のクエリ長上限を確実に超える。

そこでステージング表への load → 1本の UPDATE で埋める。BigQuery のロードジョブは
無課金、UPDATE のスキャンは確定層 1.1GB + ステージング 0.1GB 程度で無料枠に収まる。

## 使い方

    cd operation
    export PYTHONPATH=src:../learning/src
    python scripts/backfill_last3f.py --silver ../learning/data_real/silver   # 計画のみ
    python scripts/backfill_last3f.py --silver ../learning/data_real/silver --apply

`--apply` を付けない限り何も書かない。既定は計画表示だけ。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

STAGING = "_last3f_backfill"


def run(cmd: list[str], *, dry: bool) -> int:
    print(("[dry] " if dry else "[run] ") + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)


def build_payload(silver: Path) -> pd.DataFrame:
    """silver から (race_id, horse_no, race_date, last3f) だけを取り出す。

    `last3f` は silver では文字列（空文字が欠測）。BigQuery の FLOAT64 に入れる
    ので、ここで必ず数値化する。DuckDB は文字列を黙って受けるが BigQuery は
    型で落ちるため、ローカルだけ通って本番で落ちる典型パターンになる。
    """
    entry = pd.read_parquet(silver / "entry.parquet",
                            columns=["race_id", "horse_no", "race_date", "last3f"])
    entry["last3f"] = pd.to_numeric(entry["last3f"], errors="coerce")
    out = entry[entry["last3f"].notna()].copy()
    # 明らかな入力異常は入れない。学習側 features/builder.py の
    # LAST3F_MIN_SEC / LAST3F_MAX_SEC と同じ範囲で切る（定義をずらさない）。
    from nar.features.builder import LAST3F_MAX_SEC, LAST3F_MIN_SEC

    out = out[out["last3f"].between(LAST3F_MIN_SEC, LAST3F_MAX_SEC)]
    out["race_date"] = pd.to_datetime(out["race_date"]).dt.date
    out["horse_no"] = out["horse_no"].astype("int64")
    return out.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="確定層の last3f を後追いで埋める")
    p.add_argument("--silver", required=True, help="learning 側 silver ディレクトリ")
    p.add_argument("--project", default="sample-335613")
    p.add_argument("--dataset", default="nar_ops")
    p.add_argument("--apply", action="store_true", help="実際に書き込む")
    args = p.parse_args(argv)

    dry = not args.apply
    silver = Path(args.silver)
    payload = build_payload(silver)
    span = (payload["race_date"].min(), payload["race_date"].max())
    print(f"投入対象 {len(payload):,} 行（{span[0]} 〜 {span[1]}）")
    if payload.empty:
        print("埋める値がありません。silver に last3f が入っているか確認してください。",
              file=sys.stderr)
        return 1

    table = f"{args.project}:{args.dataset}.{STAGING}"
    fq = f"`{args.project}.{args.dataset}`"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "last3f.parquet"
        if not dry:
            payload.to_parquet(path, index=False)
        print(f"ステージング {path}（{len(payload):,} 行）")

        # 1) ステージング表へロード（ロードジョブは無課金）
        rc = run(["bq", f"--project_id={args.project}", "load", "--replace",
                  "--source_format=PARQUET", table, str(path)], dry=dry)
        if rc != 0:
            return rc

        # 2) 1本の UPDATE で確定層へ反映する。
        #    確定層は require_partition_filter=true なので、UPDATE 側にも
        #    パーティション列の条件が要る（無いとクエリ自体が拒否される）。
        sql = (
            f"UPDATE {fq}.entry_result_final t "
            f"SET last3f = s.last3f "
            f"FROM {fq}.{STAGING} s "
            f"WHERE t.race_id = s.race_id AND t.horse_no = s.horse_no "
            f"AND t.race_date = s.race_date "
            f"AND t.race_date BETWEEN '{span[0]}' AND '{span[1]}'"
        )
        rc = run(["bq", f"--project_id={args.project}", "query",
                  "--use_legacy_sql=false", sql], dry=dry)
        if rc != 0:
            return rc

        # 3) ステージング表は残さない。次回以降の混乱の元になる。
        rc = run(["bq", f"--project_id={args.project}", "rm", "-f", "-t", table],
                 dry=dry)
        if rc != 0:
            return rc

    print("完了。`narops health` と skew 検証で確認してください。" if not dry
          else "計画のみ。実行するには --apply を付けてください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
