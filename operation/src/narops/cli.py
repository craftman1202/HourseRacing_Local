"""運用 CLI。

FastAPI のエンドポイントと同じ処理を手元から叩けるようにする。障害対応時に
「Web が落ちているので何もできない」状態を作らないため。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from datetime import date

import pandas as pd

from .clock import SystemClock, business_date
from .config import OpsConfig, SecretResolver, redact
from .db import schema
from .db.backend import Warehouse
from .db.freshness import check as check_freshness
from .model.registry import ModelRegistry
from .monitoring import MonitorConfig, check_model_freshness, should_block_delivery

log = logging.getLogger("narops")


def _warehouse(args) -> Warehouse:
    """`NAROPS_BACKEND=bigquery` なら実 BigQuery に繋ぐ。

    これが無いと `--db` に何を渡してもローカル DuckDB ファイルを作るだけで、
    本番相手に「実運用の前提」コマンドを打ったつもりが実は何も届いていない、
    という事故になる（2026-08-29 に実際に起きた — `--db bigquery` は
    "bigquery" という名前のローカルファイルを作っただけで、本番 BigQuery には
    一切書き込まれていなかった）。`api_app.py::_default_warehouse` と同じ
    環境変数規約に揃える。
    """
    cfg = OpsConfig.load()
    if os.environ.get("NAROPS_BACKEND") == "bigquery":
        from .gcp import BigQueryWarehouse

        log.info("NAROPS_BACKEND=bigquery: 本番 BigQuery (%s.%s) に接続します",
                 cfg.project, cfg.raw["gcp"]["resources"]["bq_dataset"])
        return BigQueryWarehouse(project=cfg.project,
                                 dataset=cfg.raw["gcp"]["resources"]["bq_dataset"],
                                 location=cfg.region, max_bytes_billed=cfg.max_bytes_billed)
    wh = Warehouse(args.db, max_bytes_billed=cfg.max_bytes_billed)
    schema.create_all(wh)
    return wh


def cmd_load_history(args) -> int:
    """学習側の silver（月次ファイル由来）を確定層に投入する。

    運用系が実データで動くための前提。これが無いと確定層が空のままで、
    鮮度ゲート（DB-04）が毎回 fail-closed になる。

    実処理は `refresh.load_history()` に委譲する（自動日次更新ジョブと共有する
    ため、`narops.refresh` にリファクタ済み — 挙動は変えていない）。
    """
    from .refresh import load_history

    entry_path = Path(args.silver) / "entry.parquet"
    race_path = Path(args.silver) / "race.parquet"
    for f in (entry_path, race_path):
        if not f.exists():
            print(f"{f} がありません。`nar silver` を先に実行してください。",
                  file=sys.stderr)
            return 1

    entry = pd.read_parquet(entry_path)
    race = pd.read_parquet(race_path)
    since = pd.Timestamp(args.since).date() if args.since else None
    if since is not None:
        span_min = pd.to_datetime(entry["race_date"]).min()
        if pd.notna(span_min) and span_min > pd.Timestamp(since) - pd.Timedelta(days=365):
            print(f"警告: 入力の開始が {span_min.date()} で、--since {args.since} に対し"
                  "履歴が短すぎます。速度指数の基準統計量が揃いません。",
                  file=sys.stderr)

    wh = _warehouse(args)
    result = load_history(wh, entry, race, since=since, source_sha256=args.source_sha256)
    if result.n_pending:
        print(f"着順未確定の行 {result.n_pending:,} 件を含めて投入します（履歴の整合のため）")
    for year, res in result.per_year:
        print(f"  {year}: 追加 {res.inserted:,} / 更新 {res.updated:,} / "
              f"不変 {res.unchanged:,}", flush=True)
    print(f"確定層に {result.total_merged:,} 行を反映しました。{result.freshness.describe()}")
    wh.close()
    return 0


def _predictions_by(args, release_id: str) -> int:
    """そのリリースが実際に出した推論の件数。

    「まだ使われていない」ことの判定はポインタの有無ではなく、推論の実績で行う。
    ポインタの有無で判定すると、運用中のモデルを黙って差し替える穴になる。
    """
    wh = _warehouse(args)
    try:
        # prediction は race_date でパーティションされている。条件を付けないと
        # BigQuery が実行を拒否する。初回導入の判定なので、システムの稼働開始
        # より前から見れば十分。
        row = wh.query(
            "SELECT COUNT(*) AS n FROM prediction "
            "WHERE race_date >= ? AND model_release = ?",
            [date(2020, 1, 1), release_id], allow_full_scan=True)
        return int(row["n"].iloc[0]) if len(row) else 0
    finally:
        wh.close()


def cmd_upload_release(args) -> int:
    """ローカルのリリースを GCS に上げる。

    Cloud Run はローカルディスクを持たない。配布物は GCS から取る。
    `current` の差し替えは既定では行わない。上げることと切り替えることは
    別の操作で、まとめると「上げたつもりが本番が切り替わっていた」が起きる。
    """
    from .gcp import GcsModelRegistry
    from .publish import upload_release

    # バケット名は infra/services.json が唯一の情報源。OpsConfig にも書くと
    # 二重管理になり、片方だけ直した状態で本番が別のバケットを見る。
    # コンテナからも使えるよう、場所は環境変数で差し替えられるようにする。
    infra_path = Path(os.environ.get(
        "NAROPS_INFRA_FILE",
        Path(__file__).resolve().parents[2] / "infra" / "services.json"))
    if not infra_path.exists():
        print(f"{infra_path} がありません。NAROPS_INFRA_FILE で指定してください。",
              file=sys.stderr)
        return 1
    infra = json.loads(infra_path.read_text(encoding="utf-8"))
    project = infra["project"]
    bucket = infra["gcs"]["model"]
    local = Path(args.model_root) / "releases" / args.release_id
    if not local.is_dir():
        print(f"{local} がありません。先に `publish-release` を実行してください。",
              file=sys.stderr)
        return 1

    uploaded = upload_release(local, bucket, project, args.release_id)
    print(f"{len(uploaded)} 個を {bucket} へ上げました:")
    for name in uploaded:
        print(f"  {name}")

    if not args.set_current:
        print("current は変更していません。切り替えは --set-current か "
              "`narops promote`。")
        return 0

    reg = GcsModelRegistry(bucket=bucket, project=project, family=args.family)
    existing = reg.current_id()
    if existing is not None and existing != args.release_id:
        print(f"GCS 側の current は既に {existing} です。入れ替えは "
              "`narops promote` を使ってください。", file=sys.stderr)
        return 1
    reg.set_current(args.release_id, actor="cli", reason="初回導入")
    print(f"GCS 側 current = {args.release_id}")
    return 0


def cmd_bootstrap_current(args) -> int:
    """最初の1本だけ current に置く。

    `promote` は「現行モデルより良いこと」を昇格条件にしている（RL-03）。
    比較対象が存在しない初回にはその判定が使えない。だからといって
    `promote` の条件を緩めると、以後の入れ替えでも同じ抜け道が開く。

    この経路は **current が未設定のときだけ** 通る。既にあれば必ず `promote`
    を使わせる。置いた直後は shadow で動かすこと（ベット候補も配信も出ない）。
    """
    from .model.registry import ModelRegistry

    reg = ModelRegistry(args.model_root, family=args.family)
    existing = reg.current_id()
    if existing is not None:
        used = _predictions_by(args, existing)
        if not (args.replace_unused and used == 0):
            print(f"current は既に {existing} です（推論 {used:,} 件）。"
                  "入れ替えは `narops promote` を使ってください"
                  "（シャドー実績の比較が必要）。", file=sys.stderr)
            return 1
        print(f"{existing} は推論を1件も出していないため、初回設定の訂正として"
              "差し替えます。")
    if not args.confirm:
        print(f"{args.release_id} を current にします。--confirm を付けてください。",
              file=sys.stderr)
        return 2

    reg.set_current(args.release_id, actor=args.actor, reason="初回導入")
    print(f"current = {args.release_id}")
    print("モード: shadow のまま運用してください。ベット候補も配信も出ません。")
    print("live へ上げるには 14 日分のシャドー実績を貯めて `narops promote`。")
    return 0


def cmd_publish_release(args) -> int:
    """学習側の成果物をリリースにまとめる。

    manifest に入れる値はすべて学習側の出力から取る。手で書くと、配布物と
    manifest が食い違ったまま本番に出る。
    """
    import json

    from .model.registry import ModelRegistry
    from .publish import build_release
    from .shared import feature_config_for

    final = Path(args.final_dir)
    art = Path(args.artifacts)
    meta_path = final / "final_meta.json"
    gate_path = art / "gate_report.json"
    for f in (meta_path, gate_path):
        if not f.exists():
            print(f"{f} がありません。`nar fit-final` と `nar gate-report` を"
                  "先に実行してください。", file=sys.stderr)
            return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    # OOS の数字は「OOS 境界で打ち切った評価用モデル」で測ったもの。出荷用は
    # そのあと全データで作り直すので、同一のモデルではない。標準的な手順だが、
    # manifest を読む人が数字の出どころを取り違えないよう注記を残す。
    purpose = meta.get("purpose", "evaluation")
    evaluated_on = meta.get("oos_evaluated_on")
    if args.eval_final_dir:
        eval_meta = Path(args.eval_final_dir) / "final_meta.json"
        if eval_meta.exists():
            evaluated_on = json.loads(
                eval_meta.read_text(encoding="utf-8"))["train_period"]["end"]
    if purpose == "production":
        if not evaluated_on:
            print("出荷用ですが、OOS を測ったモデルの学習期間末が不明です。"
                  "--eval-final-dir を指定してください。", file=sys.stderr)
            return 1
        print(f"出荷用（全データで再学習）。OOS の数値は {evaluated_on} までで"
              "学習した評価用モデルで測った値です。")

    weights_path = art / "ensemble_weights.csv"
    weights: dict[str, float] = {}
    if weights_path.exists():
        w = pd.read_csv(weights_path)
        name_col = w.columns[0]
        weights = {str(r[name_col]): float(r["weight"]) for _, r in w.iterrows()}

    # OOS はアンサンブル（実際に配るもの）の数字を載せる。モデル単体の
    # いちばん良い数字を載せると、配布物の性能を過大に見せることになる。
    oos_path = art / "oos_metrics.json"
    oos_metrics: dict[str, float] = {}
    if oos_path.exists():
        payload = json.loads(oos_path.read_text(encoding="utf-8"))
        block = payload.get("metrics", payload)
        chosen = block.get("ensemble") if isinstance(block, dict) else None
        if not isinstance(chosen, dict):
            chosen = {k: v for k, v in payload.items()
                      if isinstance(v, (int, float))}
        oos_metrics = {k: float(v) for k, v in chosen.items()
                       if isinstance(v, (int, float))}
        fired = [t["id"] for t in payload.get("tripwires", []) if t.get("fired")]
        if fired:
            print(f"OOS ガードが発火しています: {fired}。publish しません。",
                  file=sys.stderr)
            return 1

    beta_path = final / "clogit_beta.json"
    beta = (json.loads(beta_path.read_text(encoding="utf-8"))["beta"]
            if beta_path.exists() else None)
    lgbm = final / "lgbm_rank.txt"
    onnx = final / "tabm.onnx"

    # 較正温度はモデルごとに測っているが、manifest は1つしか持たない。
    # 重みが最大のモデルの温度を採る。0 重みのモデルの温度を採ると、
    # 実際に効いているモデルと違う較正が入る。
    temps = meta.get("temperatures", {})
    lead = max(weights, key=weights.get) if weights else None
    temperature = float(temps.get(lead, next(iter(temps.values()), 1.0)))

    result = build_release(
        release_id=args.release_id,
        out_dir=Path(args.model_root) / "staging",
        feature_names=meta["feature_names"],
        dataset_version=args.dataset_version or meta.get("gold_content_hash", ""),
        train_period=meta["train_period"],
        oos_metrics=oos_metrics,
        ensemble_weights=weights,
        # 特徴量の最長ルックバックは学習側の設定が唯一の情報源。運用側で
        # 別に持つと MP-04 が「manifest のほうが短い」で起動を止める。
        lookback_days=int(feature_config_for(args.family).max_lookback_days),
        temperature=temperature,
        test_results=gate["results"],
        track=args.track,
        git_commit=args.git_commit,
        oos_evaluated_on=evaluated_on,
        purpose=purpose,
        clogit_beta=beta,
        lgbm_booster_path=lgbm if lgbm.exists() else None,
        tabm_onnx_path=onnx if onnx.exists() else None,
        standardizer=meta.get("standardizer"),
    )
    print(f"リリースを組み立てました: {result.path}")
    print(f"  重み: {result.manifest.ensemble_weights}")
    print(f"  温度: {temperature:.4f}（{lead} 基準）")
    for n in result.notes:
        print(f"  注意: {n}")
    if args.stage:
        print("--stage 指定のため登録はしません。")
        return 0

    reg = ModelRegistry(args.model_root, family=args.family)
    reg.publish(args.release_id, result.path)
    print(f"レジストリに登録しました: {args.release_id}")
    print("current への切り替えは `narops promote` で行ってください（二段確認）。")
    return 0


def cmd_init(args) -> int:
    wh = _warehouse(args)
    print(f"DWH を初期化しました: {args.db}")
    wh.close()
    return 0


def cmd_health(args) -> int:
    """鮮度・モデル・配信可否をまとめて出す。"""
    cfg = OpsConfig.load()
    clock = SystemClock()
    wh = _warehouse(args)
    try:
        fresh = check_freshness(wh, clock)
        print(f"DB 鮮度      : {'OK' if fresh.is_fresh else 'STALE'}  {fresh.describe()}")

        if args.model_root:
            reg = ModelRegistry(args.model_root, family=args.family)
            rid = reg.current_id()
            print(f"current      : {rid}")
            if rid:
                m = reg.load(rid).manifest
                trained = pd.Timestamp(m.train_period["end"]).date()
                alert = check_model_freshness(
                    trained, business_date(clock.now()),
                    MonitorConfig(model_stale_days=cfg.model_stale_days))
                print(f"学習期間末   : {trained}  {'⚠ ' + alert.message if alert else 'OK'}")
                print(f"dataset_ver  : {m.dataset_version}")
        return 0 if fresh.is_fresh else 1
    finally:
        wh.close()


def cmd_verify_release(args) -> int:
    """配布物の完全性と feature_spec を検証する（MP-02/03）。"""
    reg = ModelRegistry(args.model_root, family=args.family)
    rid = args.release or reg.current_id()
    if not rid:
        print("current ポインタが未設定です", file=sys.stderr)
        return 2
    rel = reg.load(rid, verify=True)
    print(f"{rid}: 配布物の SHA-256 一致 OK")
    print(f"  dataset_version : {rel.manifest.dataset_version}")
    print(f"  lookback_days   : {rel.manifest.lookback_days}")
    print(f"  OOS             : {rel.manifest.oos_metrics}")
    return 0


def cmd_promote(args) -> int:
    from .release import ShadowMetrics, promote

    reg = ModelRegistry(args.model_root, family=args.family)
    shadow = ShadowMetrics(args.shadow_days, args.shadow_nll, args.shadow_ece, 0.0)
    prod = ShadowMetrics(30, args.production_nll, args.production_ece, 0.0)
    promote(reg, args.release, shadow, prod, actor=args.actor, confirmed=args.confirm)
    print(f"current を {args.release} に切り替えました（実行者 {args.actor}）")
    return 0


def cmd_rollback(args) -> int:
    reg = ModelRegistry(args.model_root, family=args.family)
    previous = reg.rollback(actor=args.actor)
    print(f"current を {previous} へ戻しました")
    return 0


def cmd_skew_check(args) -> int:
    """SK-01: 前日分の snapshot と再計算値を突き合わせる。"""
    from .features import load_snapshot
    from .monitoring import check_skew
    from .skew import compare

    cfg = OpsConfig.load()
    wh = _warehouse(args)
    try:
        day = pd.Timestamp(args.date).date() if args.date else business_date(
            SystemClock().now()) - pd.Timedelta(days=1).to_pytimedelta()
        snap = load_snapshot(wh, day, max_bytes_billed=cfg.max_bytes_billed)
        if snap.empty:
            print(f"{day}: snapshot がありません（推論が走っていない可能性）")
            return 1
        recomputed = pd.read_parquet(args.recomputed) if args.recomputed else snap
        report = compare(snap, recomputed, cfg.tolerated_skew_columns, day)
        print(report.summary())
        alert = check_skew(report)
        if alert:
            print(f"[{alert.severity}] {alert.message}", file=sys.stderr)
            return 1
        return 0
    finally:
        wh.close()


def cmd_diff_features(args) -> int:
    """ランブック用: 該当レースの学習時計算と推論時計算を並べる。"""
    from .features import load_snapshot

    cfg = OpsConfig.load()
    wh = _warehouse(args)
    try:
        snap = load_snapshot(wh, pd.Timestamp(args.date).date(),
                             max_bytes_billed=cfg.max_bytes_billed)
        rows = snap[snap["race_id"] == args.race_id]
        if rows.empty:
            print(f"{args.race_id}: snapshot がありません")
            return 1
        print(rows.T.to_string())
        return 0
    finally:
        wh.close()


def cmd_estimate_cost(args) -> int:
    from .cost import CostEstimate, TARGET_MONTHLY_USD, default_usage

    est = CostEstimate(default_usage(args.odds_per_day))
    total = est.compute()
    print(json.dumps(est.breakdown, ensure_ascii=False, indent=2))
    print(f"月額見積: ${total}（目標 ${TARGET_MONTHLY_USD}）")
    return 0 if total <= TARGET_MONTHLY_USD else 1


def cmd_check_secrets(args) -> int:
    """DC-02: 必要な秘密がすべて解決できるか。値は絶対に表示しない。"""
    r = SecretResolver(args.env)
    keys = ["DISCORD_WEBHOOK_PREDICTION", "DISCORD_WEBHOOK_ALERT",
            "DISCORD_WEBHOOK_DAILY"]
    missing = []
    for k in keys:
        try:
            value = r.get(k)
            print(f"{k}: 解決 OK（{redact(value)}）")
        except RuntimeError:
            missing.append(k)
            print(f"{k}: **未設定**", file=sys.stderr)
    return 1 if missing else 0


def cmd_deploy(args) -> int:
    """既定は plan。--apply を明示したときだけ作成する。"""
    from .deploy import apply_plan, build_plan

    plan = build_plan(service_url=args.service_url, image=args.image,
                      mode=args.mode)
    print(plan.render())
    if not args.apply:
        print("\n（plan のみ。作成するには --apply --confirm を付けてください）")
        return 1 if plan.collisions else 0

    results = apply_plan(plan, confirm=args.confirm)
    for name, ok, out in results:
        print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f"\n     {out}" if not ok else ""))
    return 0 if all(ok for _, ok, _ in results) else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser("narops", description="地方競馬 予測システム 運用CLI")
    p.add_argument("--db", default="./data/nar_ops.duckdb")
    p.add_argument("--model-root", default="./data/nar-model")
    # 競技ごとに current ポインタを分ける。releases/ は共有で、
    # flat → current.json / banei → current_banei.json を読み書きする。
    p.add_argument("--family", default="flat", choices=("flat", "banei"),
                   help="モデル系統（既定 flat = 平地）")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="DWH を初期化").set_defaults(func=cmd_init)
    sub.add_parser("health", help="鮮度・モデル・配信可否").set_defaults(func=cmd_health)

    s = sub.add_parser("verify-release", help="配布物の完全性を検証")
    s.add_argument("--release", default=None)
    s.set_defaults(func=cmd_verify_release)

    s = sub.add_parser("promote", help="current を切り替える（二段確認必須）")
    s.add_argument("release")
    s.add_argument("--actor", required=True)
    s.add_argument("--confirm", action="store_true")
    s.add_argument("--shadow-days", type=int, default=14)
    s.add_argument("--shadow-nll", type=float, required=True)
    s.add_argument("--shadow-ece", type=float, required=True)
    s.add_argument("--production-nll", type=float, required=True)
    s.add_argument("--production-ece", type=float, required=True)
    s.set_defaults(func=cmd_promote)

    s = sub.add_parser("rollback", help="直前リリースへ戻す")
    s.add_argument("--actor", default="oncall")
    s.set_defaults(func=cmd_rollback)

    s = sub.add_parser("skew-check", help="学習/推論 skew の検証")
    s.add_argument("--date", default=None)
    s.add_argument("--recomputed", default=None)
    s.set_defaults(func=cmd_skew_check)

    s = sub.add_parser("diff-features", help="特定レースの特徴量を並べる")
    s.add_argument("--race-id", required=True)
    s.add_argument("--date", required=True)
    s.set_defaults(func=cmd_diff_features)

    s = sub.add_parser("estimate-cost", help="月額見積")
    s.add_argument("--odds-per-day", type=int, default=110)
    s.set_defaults(func=cmd_estimate_cost)

    s = sub.add_parser("deploy", help="GCP 資源の計画/作成（既定は plan）")
    s.add_argument("--apply", action="store_true", help="実際に作成する")
    s.add_argument("--confirm", action="store_true", help="課金発生の承認")
    s.add_argument("--image", default=None,
                   help="Cloud Run に載せるイメージ（Artifact Registry のURL）")
    s.add_argument("--service-url", default=None,
                   help="Cloud Scheduler が叩く Cloud Run の URL")
    s.add_argument("--mode", default="shadow", choices=["shadow", "paper", "live"],
                   help="運用モード。paper は配信するがベットは仮想。"
                        "live は実投票を前提とする")
    s.set_defaults(func=cmd_deploy)

    s = sub.add_parser("load-history",
                       help="学習側 silver を確定層に投入（実運用の前提）")
    s.add_argument("--silver", required=True, help="learning/data_real/silver のパス")
    s.add_argument("--since", default=None, help="この日付以降だけ入れる（YYYY-MM-DD）")
    s.add_argument("--source-sha256", default="", help="投入元のハッシュ（監査用）")
    s.set_defaults(func=cmd_load_history)

    s = sub.add_parser("upload-release",
                       help="リリースを GCS へ上げる（Cloud Run が読む先）")
    s.add_argument("--release-id", required=True)
    s.add_argument("--set-current", action="store_true",
                   help="GCS 側の current も差し替える（未設定のときのみ）")
    s.set_defaults(func=cmd_upload_release)

    s = sub.add_parser("bootstrap-current",
                       help="最初のリリースを current にする（既に current があれば拒否）")
    s.add_argument("--release-id", required=True)
    s.add_argument("--actor", required=True)
    s.add_argument("--confirm", action="store_true")
    s.add_argument("--replace-unused", action="store_true",
                   help="現在の current がまだ1件も推論していない場合に限り差し替える")
    s.set_defaults(func=cmd_bootstrap_current)

    s = sub.add_parser("publish-release",
                       help="学習側の最終成果物からリリースを組み立てて登録する")
    s.add_argument("--release-id", required=True, help="例: v2026.08.27-A")
    s.add_argument("--final-dir", required=True, help="nar fit-final の出力")
    s.add_argument("--artifacts", required=True, help="learning/artifacts")
    s.add_argument("--dataset-version", default=None,
                   help="省略時は final_meta.json の gold content_hash を使う")
    s.add_argument("--track", default="A")
    s.add_argument("--git-commit", default="")
    s.add_argument("--eval-final-dir", default=None,
                   help="OOS を測った評価用モデルのディレクトリ。出荷用を publish"
                        "するときに、どの学習期間で測った数字かを manifest に残す")
    s.add_argument("--stage", action="store_true",
                   help="組み立てるだけでレジストリに登録しない")
    s.set_defaults(func=cmd_publish_release)

    s = sub.add_parser("check-secrets", help="秘密情報が解決できるか")
    s.add_argument("--env", default="./.env")
    s.set_defaults(func=cmd_check_secrets)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
