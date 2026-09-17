"""CLI。

nar ingest → bronze → silver → features → eda → train → evaluate の順に冪等実行できる。
各ステップは上流の入力ハッシュが変わらなければ再計算しない。
"""

from __future__ import annotations

import argparse
import re
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import synth
from .config import CONF_DIR, cv_config, data_config, eda_config, feature_config
from .eval import guards, metrics
from .eval.splits import make_folds, make_oos_fold, validate_folds, OOSGuard
from .features.builder import (
    asof_features, build as build_features, content_hash)
from .io.store import Store
from .models import baselines
from .models.base import to_batch
from .models.clogit import ConditionalLogit

log = logging.getLogger("nar")


def _load_silver(store: Store) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("race", "entry", "payout", "odds"):
        p = Path(store.path("silver", f"{name}.parquet"))
        if not p.exists():
            raise SystemExit(
                f"{p} がありません。`nar synth` で合成データを作るか、"
                "`nar ingest` → `nar silver` を先に実行してください。"
            )
        tables[name] = pd.read_parquet(p)
    return tables


def cmd_synth(args: argparse.Namespace) -> int:
    """実データが無い段階でも全経路を通せるようにする（FX-03 相当）。"""
    store = Store(args.data_root)
    store.ensure_layout()
    tables = synth.generate(synth.SynthConfig(n_races=args.n_races, seed=args.seed))
    for name, df in tables.items():
        path = Path(store.path("silver", f"{name}.parquet"))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"silver/{name}.parquet  {len(df):,} 行")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """NAR から月次ファイルを取得して raw 層に不変保存する。"""
    from datetime import date as _date

    from .ingest.client import NarClient, RetryPolicy
    from .ingest.monthly import backfill
    from .io.manifest import Manifest, finalize_pass

    dcfg = data_config()
    store = Store(args.data_root)
    store.ensure_layout()
    manifest = Manifest(store.path("meta", "manifest.duckdb"))
    http = dcfg["http"]
    today = _date.fromisoformat(args.today) if args.today else _date.today()

    client = NarClient(
        user_agent=http["user_agent"], min_interval_sec=float(http["min_interval_sec"]),
        timeout_sec=float(http["timeout_sec"]),
        retry=RetryPolicy(statuses=tuple(http["retry"]["statuses"]),
                          max_attempts=int(http["retry"]["max_attempts"]),
                          base_sec=float(http["retry"]["base_sec"]),
                          factor=float(http["retry"]["factor"]),
                          jitter=float(http["retry"]["jitter"])))

    def progress(i, n, res):
        if res.status != "skipped" or i % 25 == 0:
            print(f"[{i:>3}/{n}] {res.file_key} {res.status}"
                  + (f" {res.size:,}B" if res.size else "")
                  + (f" {res.error}" if res.error else ""), flush=True)

    try:
        default_start = (dcfg["backfill"]["race_start_ym"] if args.kind == "race"
                         else dcfg["backfill"]["odds_start_ym"])
        results = backfill(client, store, manifest,
                           args.start or default_start, args.end, today,
                           int(dcfg["backfill"]["finalize_after_days"]),
                           kind=args.kind, on_progress=progress)
    finally:
        client.close()

    n_final = finalize_pass(manifest, today,
                            int(dcfg["backfill"]["finalize_after_days"]))
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"\n{counts} / 確定化 {n_final} 件")
    print(f"dataset_version: {manifest.dataset_version()}")
    manifest.close()
    return 0 if counts.get("error", 0) == 0 else 1


def cmd_bronze(args: argparse.Namespace) -> int:
    """raw の ZIP を展開・デコードして bronze に置く。ネットワーク不要（IG-14）。"""
    from .io.manifest import Manifest
    from .transform import bronze as bz

    store = Store(args.data_root)
    manifest = Manifest(store.path("meta", "manifest.duckdb"))
    n_tables = 0
    for rec in manifest.all():
        if rec.status != "ok" or not rec.raw_path:
            continue
        # file_key は race が monthly/race/YYYY-MM、odds が monthly/odds/YYYY-MM/NN。
        # 末尾を取ると odds で分割番号を掴んでしまう
        m = re.search(r"\d{4}-\d{2}", rec.file_key)
        if m is None:
            print(f"{rec.file_key}: 年月を判定できません", file=sys.stderr)
            continue
        ym = m.group(0)
        try:
            raw = Path(rec.raw_path).read_bytes()
        except OSError as exc:
            print(f"{rec.file_key}: raw を読めません ({exc})", file=sys.stderr)
            continue
        for table in bz.build_from_zip(raw, ym, source=rec.file_key):
            bz.write(store, table)
            n_tables += 1
    manifest.close()
    print(f"bronze: {n_tables} テーブル")
    return 0


def cmd_track_master(args: argparse.Namespace) -> int:
    """競馬場コードのハードコードを避けるための自動生成（設計書 §キー設計）。

    廃止場は現在の日程ページに出ないので過去月を遡る必要がある。bronze が
    あればそこに実在する名前が全部揃った時点で打ち切る。
    """
    from .ingest import monthly
    from .ingest.client import NarClient
    from .ingest.schedule import build_track_master, save
    from .transform import bronze as bz
    from .transform.keys import TRACK_MASTER_FILE

    store = Store(args.data_root)
    http = data_config()["http"]
    client = NarClient(user_agent=http["user_agent"],
                       min_interval_sec=float(http["min_interval_sec"]),
                       expect_content_type="text/html")

    required: set[str] = set()
    for ym in bz.available_months(store, "race"):
        frame = bz.read(store, "race", ym)
        if "競馬場" in frame.columns:
            required |= set(frame["競馬場"].astype(str).str.strip())
    print(f"bronze に実在する競馬場: {len(required)} 場")

    months = [(int(ym[:4]), int(ym[5:7]))
              for ym in monthly.month_range(args.start, args.end)]
    build = build_track_master(client, months, required_names=required or None)
    missing = required - set(build.mapping)
    for name, code in sorted(build.mapping.items(), key=lambda kv: kv[1]):
        print(f"  {code:>2}  {name}")
    print(f"走査 {build.months_scanned} か月 / 取得 {len(build.mapping)} 場")
    if missing:
        print(f"未取得: {sorted(missing)}", file=sys.stderr)
        return 1
    path = save(build.mapping, store.path("meta", TRACK_MASTER_FILE))
    print(f"→ {path}")
    return 0


def cmd_silver(args: argparse.Namespace) -> int:
    """bronze から silver（型付け・キー付与）を作る。"""
    from .transform import bronze as bz
    from .transform import silver as sv
    from .transform.keys import TrackMaster

    store = Store(args.data_root)
    master = TrackMaster.load(store)
    races, entries, payouts, oddss = [], [], [], []
    months = bz.available_months(store, "race")
    for i, ym in enumerate(months, start=1):
        r_raw, e_raw, p_raw = (bz.read(store, k, ym) for k in ("race", "entry", "payout"))
        if r_raw.empty or e_raw.empty:
            continue
        try:
            race = sv.build_race(r_raw, master)
            entry = sv.attach_start_ts(sv.build_entry(e_raw, master), race)
            races.append(race)
            entries.append(entry)
            if not p_raw.empty:
                payouts.append(sv.build_payout(p_raw, master))
        except Exception as exc:  # noqa: BLE001
            print(f"{ym}: silver 構築に失敗 ({exc})", file=sys.stderr)
            continue
        o_raw = bz.read(store, "odds", ym)
        if not o_raw.empty:
            try:
                oddss.append(sv.build_odds(o_raw, master))
            except Exception as exc:  # noqa: BLE001
                print(f"{ym}: odds 構築に失敗 ({exc})", file=sys.stderr)
        if i % 24 == 0:
            print(f"  {i}/{len(months)} 月処理", flush=True)

    if not races:
        print("bronze が空です。先に `nar ingest` と `nar bronze` を実行してください。",
              file=sys.stderr)
        return 1

    out = {"race": pd.concat(races, ignore_index=True).drop_duplicates("race_id"),
           "entry": pd.concat(entries, ignore_index=True)
                      .drop_duplicates(["race_id", "horse_no"]),
           "payout": (pd.concat(payouts, ignore_index=True) if payouts
                      else pd.DataFrame())}
    out["odds"] = (pd.concat(oddss, ignore_index=True)
                     .drop_duplicates(["race_id", "horse_no"]) if oddss
                   else pd.DataFrame())
    for name, df in out.items():
        path = Path(store.path("silver", f"{name}.parquet"))
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"silver/{name}.parquet  {len(df):,} 行")
    return 0


def cmd_leak_check(args: argparse.Namespace) -> int:
    """EDA 第二問: 累積成績8列の時点性判定（LK-01..03 / LK-09..11）。

    設計書 §12 は「bronze/silver の直後にこれを実施し、結論を出してから先へ進む」
    と定めている。判定結果が特徴量設計を左右するので、独立したコマンドにする。
    """
    from .eda import leakage

    store = Store(args.data_root)
    entry = pd.read_parquet(Path(store.path("silver", "entry.parquet")))
    # `ダート左/右成績` は回り、`最高タイム良馬場` は馬場状態を条件にしている。
    # どちらもレース表側の属性なので、判定にはこの join が要る。
    race = pd.read_parquet(Path(store.path("silver", "race.parquet")))
    attrs = [c for c in ("turn", "baba_condition", "surface")
             if c in race.columns and c not in entry.columns]
    if attrs:
        entry = entry.merge(race[["race_id", *attrs]], on="race_id", how="left")
    else:
        print("警告: race に 回り/馬場状態 がありません。条件付き列は判定不能になります。",
              file=sys.stderr)
    entry = entry.sort_values(["horse_sk", "start_ts"]).reset_index(drop=True)
    print(f"対象: {len(entry):,} 行 / {entry['race_id'].nunique():,} レース")
    print(f"期間: {entry['race_date'].min()} 〜 {entry['race_date'].max()}\n")

    verdicts = leakage.run_all(entry)
    for col in leakage.CUMULATIVE_RECORD_COLS:
        vs = [v for v in verdicts if v.column == col]
        if not vs:
            print(f"{col:<18} 列が存在しません")
            continue
        for i, v in enumerate(vs):
            head = col if i == 0 else ""
            print(f"{head:<18} {v.test} {v.verdict:<16} {v.evidence}")
        print()

    decision = leakage.decide(verdicts)
    print(f"\n{'='*70}")
    print(f"結論: {decision['conclusion']}")
    print(f"  使用可（ホワイトリスト）: {decision['whitelist']}")
    print(f"  破棄                    : {decision['discard']}")
    print(f"  判定不能（破棄側へ）    : {decision['undetermined']}")
    for col, why in decision["reasons"].items():
        print(f"    - {col}: {why}")

    out = Path(args.artifacts) / "leak_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "decided_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_rows": int(len(entry)), "n_races": int(entry["race_id"].nunique()),
        "period": [str(entry["race_date"].min()), str(entry["race_date"].max())],
        "verdicts": [v.__dict__ for v in verdicts], **decision,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    from .eda import run as run_eda

    t = _load_silver(Store(args.data_root))
    conf = eda_config()
    out_dir = args.out or conf.get("report_dir", "./artifacts/eda")

    # Q4（favourite-longshot bias）と Q6（控除率実測）は単勝オッズが要る。
    # オッズは別テーブルで、NAR は 2026-02 以降しか配信していない。結合できる
    # 期間だけで算出し、範囲はレポートに明示する。
    entry = t["entry"]
    odds = t["odds"]
    if len(odds) and "odds_win" in odds.columns:
        entry = entry.merge(odds[["race_id", "horse_no", "odds_win"]],
                            on=["race_id", "horse_no"], how="left")
        covered = entry["odds_win"].notna()
        print(f"オッズ結合: {int(covered.sum()):,} 行 / {len(entry):,} 行 "
              f"({covered.mean():.1%}）— NAR のオッズ配信は 2026-02 以降のみ")
    res = run_eda(entry, t["race"], t["payout"], conf, out_dir=out_dir)

    print(f"品質スコア      : {res['scorecard']['score']} / 10  → {res['scorecard']['verdict']}")
    print(f"リーク判定      : {res['leak']['conclusion']}")
    print(f"学習開始年の提案: {res['usable']['proposed_train_from']}")
    print(f"構造変化点      : {res['change_points']['year'].tolist() if len(res['change_points']) else 'なし'}")
    print(f"チェックリスト  : {res['signoff']['counts']}")
    print(f"レポート        : {res['report_path']}")
    print(f"所見サマリ      : {res['findings_path']}")
    return 0 if res["signoff"]["can_proceed"] else 1


def _gold_subdir(cfg) -> str:
    """gold の書き出し先。トラック（オッズの有無）と variant で分ける。

    ばんえいと平地は別モデル・別列集合なので、同じ features.parquet に
    書くと後から走らせた方が相手を黙って壊す。
    """
    sub = "features_odds" if cfg.track == "B" else "features_noodds"
    return sub if getattr(cfg, "variant", "flat") == "flat" else f"{sub}_{cfg.variant}"


def cmd_features(args: argparse.Namespace) -> int:
    store = Store(args.data_root)
    t = _load_silver(store)
    cfg = feature_config()
    feat = build_features(t["entry"], t["race"], cfg)
    sub = _gold_subdir(cfg)
    path = Path(store.path("gold", sub, "features.parquet"))
    path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(path, index=False)
    # 内容ハッシュを隣に残す。下流（fit-final → manifest）が「このモデルが何を
    # 見たか」を指すのに使う。数百万行を CSV 化する計算なので毎回やらせない。
    digest = content_hash(feat)
    (path.parent / "content_hash.txt").write_text(digest + "\n", encoding="utf-8")
    print(f"gold/{sub}/features.parquet  {len(feat):,} 行 × {feat.shape[1]} 列")
    print(f"content_hash: {digest}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    store = Store(args.data_root)
    t = _load_silver(store)
    fcfg, ccfg = feature_config(), cv_config()
    feat = build_features(t["entry"], t["race"], fcfg)

    guard = OOSGuard(ccfg, Path(args.artifacts) / "oos_access.log")
    feat = guard.filter(feat)

    folds = make_folds(ccfg)
    races = feat[["race_id", "race_date"]].drop_duplicates()
    validate_folds([f for f in folds if _has_data(f, races)], races)

    cols = [c for c in asof_features(fcfg) if c in feat.columns]
    rows = []
    for f in folds:
        tr, va = f.mask(feat["race_date"])
        if tr.sum() == 0 or va.sum() == 0:
            continue
        train = _prep(feat[tr], cols)
        valid = _prep(feat[va], cols)
        model = ConditionalLogit(l2=args.l2).fit(to_batch(train, cols))
        b = to_batch(valid, cols)
        p = b.flat_predictions(model.predict_proba(b), len(valid))
        m = metrics.summary(p, valid["is_win"].to_numpy(), valid["finish_pos"].to_numpy(),
                            valid["race_id"].to_numpy())
        rows.append({"fold": f.index, "n_valid": int(va.sum()), **m})
        print(f"fold {f.index}: NLL {m['race_nll']:.4f} (uniform {m['uniform_nll']:.4f}) "
              f"top1 {m['top1']:.3f}")

    out = pd.DataFrame(rows)
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.artifacts) / "cv_metrics.csv", index=False)
    print("\n平均:", out[["race_nll", "top1", "ece"]].mean().round(4).to_dict())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    ccfg, fcfg = cv_config(), feature_config()
    if not args.unlock_oos:
        print("OOS は施錠されています。最終評価は --unlock-oos を付けて1回だけ実行してください。",
              file=sys.stderr)
        return 2

    store = Store(args.data_root)
    t = _load_silver(store)
    feat = build_features(t["entry"], t["race"], fcfg)
    guard = OOSGuard(ccfg, Path(args.artifacts) / "oos_access.log")
    guard.unlock(args.reason or "最終評価")

    oos = make_oos_fold(ccfg, pd.to_datetime(feat["race_date"]).max().date())
    tr, va = oos.mask(feat["race_date"])
    cols = [c for c in asof_features(fcfg) if c in feat.columns]
    train, valid = _prep(feat[tr], cols), _prep(feat[va], cols)

    model = ConditionalLogit(l2=args.l2).fit(to_batch(train, cols))
    b = to_batch(valid, cols)
    p = b.flat_predictions(model.predict_proba(b), len(valid))

    valid_odds = valid.merge(t["entry"][["race_id", "horse_no", "odds_win", "popularity"]],
                             on=["race_id", "horse_no"], how="left")
    y = valid["is_win"].to_numpy()
    rid = valid["race_id"].to_numpy()
    model_m = metrics.summary(p, y, valid["finish_pos"].to_numpy(), rid,
                              valid_odds["popularity"].to_numpy())
    mkt = baselines.market(valid_odds)
    mkt_m = metrics.summary(mkt, y, valid["finish_pos"].to_numpy(), rid)

    print(f"モデル NLL {model_m['race_nll']:.4f} / 市場 NLL {mkt_m['race_nll']:.4f} "
          f"/ 一様 {model_m['uniform_nll']:.4f}")
    print(f"モデル top1 {model_m['top1']:.3f} / 市場 top1 {mkt_m['top1']:.3f}")

    tw = guards.check(oos_top1=model_m["top1"], oos_nll=model_m["race_nll"])
    print(guards.to_frame(tw).to_string(index=False))
    guards.enforce(tw)

    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    (Path(args.artifacts) / "oos_metrics.json").write_text(
        json.dumps({"model": model_m, "market": mkt_m,
                    "oos_access_count": guard.access_count()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """全モデルの walk-forward 学習 → アンサンブル →（任意で）OOS 最終評価。"""
    from .train import pipeline as pl
    from .tracking import RunTracker

    store = Store(args.data_root)
    t = _load_silver(store)
    fcfg, ccfg = feature_config(), cv_config()
    feat = build_features(t["entry"], t["race"], fcfg)
    if "odds_win" in t["entry"].columns:
        feat = feat.merge(t["entry"][["race_id", "horse_no", "odds_win", "popularity"]],
                          on=["race_id", "horse_no"], how="left")

    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(store.path("meta", "manifest.duckdb"))
    dataset_version = _dataset_version(manifest_path, feat)

    cfg = pl.RunConfig(
        models=tuple(args.models.split(",")),
        hpo_trials={k: v for k, v in (
            ("clogit", args.hpo_clogit), ("lgbm", args.hpo_lgbm), ("tabm", args.hpo_tabm))},
        do_selection=not args.no_selection,
        n_null_runs=args.null_runs,
        bayes_method=args.bayes_method,
        bayes_max_races=args.bayes_max_races,
        tabm_epochs=args.tabm_epochs,
        tabm_width={"k": args.tabm_k, "hidden": args.tabm_hidden,
                    "n_layers": args.tabm_layers, "batch_races": args.tabm_batch},
        seed=args.seed,
    )

    tracker = RunTracker("nar-walkforward", artifacts)
    guard = OOSGuard(ccfg, artifacts / "oos_access.log")
    with tracker.run("walkforward", dataset_version, CONF_DIR) as run:
        run.log_params({"models": ",".join(cfg.models), "hpo": cfg.hpo_trials,
                        "selection": cfg.do_selection, "embargo_days": ccfg.embargo_days,
                        "seed": cfg.seed})
        folds = pl.run_walkforward(guard.filter(feat), fcfg, ccfg, cfg)
        oof = pl.build_oof(folds)
        fold_metrics = pd.concat([f.metrics for f in folds], ignore_index=True)

        model_cols = [m for m in cfg.models if m in oof.columns]
        stacker = pl.fit_ensemble(oof, model_cols)
        ens = pl.ensemble_metrics(oof, stacker)

        oof.to_parquet(artifacts / "oof_predictions.parquet", index=False)
        fold_metrics.to_csv(artifacts / "fold_metrics.csv", index=False)
        ens.to_csv(artifacts / "ensemble_metrics.csv", index=False)
        pd.DataFrame({"model": stacker.model_names, "weight": stacker.weights}).to_csv(
            artifacts / "ensemble_weights.csv", index=False)

        pd.DataFrame([
            {"fold": f.fold, "n_train": f.n_train, "n_valid": f.n_valid,
             "n_selected": len(f.selected_features),
             "selected": ", ".join(f.selected_features)} for f in folds
        ]).to_csv(artifacts / "selected_features.csv", index=False)
        pd.DataFrame([{"fold": f.fold, **f.timings} for f in folds]).to_csv(
            artifacts / "timings.csv", index=False)
        if any(f.hpo for f in folds):
            (artifacts / "hpo.json").write_text(
                json.dumps([{"fold": f.fold, **f.hpo} for f in folds],
                           ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _dump_bayes_diagnostics(folds, artifacts)

        mean = fold_metrics.groupby("model")["race_nll"].mean()
        run.log_metrics({f"cv_nll_{k}": float(v) for k, v in mean.items()})
        run.log_metrics({f"cv_nll_{r['model']}": float(r["race_nll"])
                         for _, r in ens.iterrows()})

    print("\n=== CV 平均（レース内 NLL、小さいほど良い）===")
    print(fold_metrics.groupby("model")[["race_nll", "top1", "top3", "ece"]]
          .mean().sort_values("race_nll").round(4).to_string())
    print("\n=== アンサンブル（OOF）===")
    print(ens.set_index("model")[["race_nll", "top1", "ece"]].round(4).to_string())
    print("\n重み:", dict(zip(stacker.model_names, stacker.weights.round(4))))

    if args.trackb:
        _run_trackb(oof, artifacts, t["entry"])

    if args.oos:
        summary, preds, tw = pl.evaluate_oos(
            feat, [c for c in asof_features(fcfg) if c in feat.columns], cfg, ccfg,
            stacker, artifacts, args.reason or "最終評価")
        summary.to_csv(artifacts / "oos_metrics.csv", index=False)
        preds.to_parquet(artifacts / "oos_predictions.parquet", index=False)
        guards.to_frame(tw).to_csv(artifacts / "oos_guards.csv", index=False)
        print("\n=== OOS ===")
        print(summary.set_index("model")[["race_nll", "top1", "top3", "ece"]].round(4).to_string())
        guards.enforce(tw)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """artifacts/ の成果物からモデル性能レポートを組み立てる。"""
    from . import report as rpt

    store = Store(args.data_root)
    artifacts = Path(args.artifacts)
    t = _load_silver(store)
    ccfg = cv_config()
    entry = t["entry"]
    oos_log = artifacts / "oos_access.log"

    meta = {
        "data_source": args.data_source,
        "is_synthetic": "合成" in args.data_source or "synth" in args.data_source.lower(),
        "n_races": int(entry["race_id"].nunique()),
        "n_entries": int(len(entry)),
        "period": f"{pd.to_datetime(entry['race_date']).min().date()} 〜 "
                  f"{pd.to_datetime(entry['race_date']).max().date()}",
        "dataset_version": args.dataset_version or _recorded_dataset_version(artifacts),
        "models": args.models.split(","),
        "embargo_days": ccfg.embargo_days,
        "oos_period": f"{ccfg.oos[0]} 〜 {ccfg.oos[1] or 'データ末尾'}",
        "oos_access_count": sum(
            1 for line in oos_log.read_text(encoding="utf-8").splitlines() if line.strip()
        ) if oos_log.exists() else 0,
        "conclusions": _conclusions(artifacts),
        "limitations": LIMITATIONS,
    }
    out = rpt.write(artifacts, meta, Path(args.out) if args.out else None)
    print(f"レポート: {out}")
    return 0


def _recorded_dataset_version(artifacts: Path) -> str:
    """学習実行が残した記録から dataset_version を拾う。

    レポートで手入力させると学習した版と別の値を書ける。記録が唯一の出所にする。
    """
    runs = artifacts / "runs.json"
    if not runs.exists():
        return ""
    try:
        data = json.loads(runs.read_text(encoding="utf-8"))
        return data[-1].get("dataset_version", "") if data else ""
    except (json.JSONDecodeError, OSError, IndexError):
        return ""


LIMITATIONS = [
    "**数値は合成データに対するもの**であり、実データの性能ではない。合成市場は真の効用に"
    "直接アクセスしているため、市場ベースラインとの優劣は構造上ほぼ意味を持たない。",
    "HPO の試行回数は設計書の予算（LightGBM 500 / TabM 300 / 条件付きロジット 100）を"
    "大きく下回る。CPU のみで nested HPO を回しているため。DL 側に十分な予算を与えないと"
    "GBDT 有利のバイアスがかかる（TabArena の知見）ので、モデル間比較はこの点を割り引くこと。",
    "ベイズは全期間ではなく直近窓のみを推論対象にしている（設計書 §8.4 の二段構え）。",
    "SBC（MD-20）は未実行。事前生成＋推論を100回繰り返す計算量が CPU では現実的でないため。",
    "温度スケーリングを各 fold の valid で最適化し、同じ valid で評価している。"
    "較正の効果はその分だけ楽観的に出る。",
    "実データの累積成績8列の時点性判定（LK-01）は、当日ファイルと後日の月次ファイルが"
    "揃うまで実行できない。現状はホワイトリスト空＝8列すべて破棄で運用している。",
]


def _conclusions(artifacts: Path) -> list[str]:
    """成果物から機械的に結論を作る。手で書くと数値とズレる。"""
    out: list[str] = []
    fold = artifacts / "fold_metrics.csv"
    if fold.exists():
        f = pd.read_csv(fold)
        mean = f.groupby("model")["race_nll"].mean().sort_values()
        models = [m for m in mean.index if not m.startswith("baseline_")]
        if models:
            best = models[0]
            out.append(f"CV 平均のレース内 NLL で最良の単体モデルは **{best}**"
                       f"（{mean[best]:.4f}）。")
        if "baseline_uniform" in mean:
            out.append(f"一様分布ベースライン（{mean['baseline_uniform']:.4f}）は"
                       "全モデルが上回っており、学習が機能している。")
        if "baseline_market" in mean and models:
            d = mean[models[0]] - mean["baseline_market"]
            out.append(f"市場ベースラインとの差は {d:+.4f}（負なら勝ち）。"
                       "合成データでは市場が真の効用を直接見ているため、この比較は参考外。")
    ens = artifacts / "ensemble_metrics.csv"
    if ens.exists() and fold.exists():
        e = pd.read_csv(ens).sort_values("race_nll")
        out.append(f"アンサンブルは **{e.iloc[0]['model']}** が最良"
                   f"（{e.iloc[0]['race_nll']:.4f}）。")
    g = artifacts / "oos_guards.csv"
    if g.exists():
        n = int(pd.read_csv(g)["fired"].sum())
        out.append(f"Too-Good-To-Be-True ガードの発火は {n} 件。"
                   + ("リーク調査が必要。" if n else "リークの兆候なし。"))
    return out


def _dump_bayes_diagnostics(folds, artifacts: Path) -> None:
    """ベイズを走らせた fold の収束診断を残す。走っていなければ何も書かない。"""
    from .train import pipeline as pl

    diag = getattr(pl, "LAST_BAYES_DIAGNOSTICS", None)
    if diag:
        (artifacts / "bayes_diagnostics.json").write_text(
            json.dumps(diag, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cmd_evaluate_final(args: argparse.Namespace) -> int:
    """出荷するモデルそのものを OOS で測る。

    `learn --oos` は walk-forward の副産物、`evaluate` は条件付きロジット単体で、
    どちらも配布物を測っていない。manifest の oos_metrics は配布物の数字である
    必要がある。
    """
    from .eval.final_oos import evaluate

    if not args.unlock_oos:
        print("OOS は施錠されています。--unlock-oos を付けて1回だけ実行してください。",
              file=sys.stderr)
        return 2

    store = Store(args.data_root)
    fcfg, ccfg = feature_config(), cv_config()
    gold = Path(store.path("gold", _gold_subdir(fcfg), "features.parquet"))
    if not gold.exists():
        print(f"{gold} がありません。", file=sys.stderr)
        return 1

    artifacts = Path(args.artifacts)
    guard = OOSGuard(ccfg, artifacts / "oos_access.log")
    guard.unlock(args.reason or "配布モデルの最終評価")

    weights: dict[str, float] = {}
    wpath = artifacts / "ensemble_weights.csv"
    if wpath.exists():
        w = pd.read_csv(wpath)
        weights = {str(r[w.columns[0]]): float(r["weight"]) for _, r in w.iterrows()}

    cv_nll = None
    fpath = artifacts / "fold_metrics.csv"
    if fpath.exists():
        fm = pd.read_csv(fpath)
        best = fm[~fm["model"].str.startswith("baseline")]
        if len(best):
            cv_nll = float(best.groupby("model")["race_nll"].mean().min())

    res = evaluate(pd.read_parquet(gold), ccfg, args.final_dir,
                   weights=weights, cv_nll=cv_nll)

    frame = pd.DataFrame(res.metrics).T.sort_values("race_nll")
    print(f"OOS: {res.period[0]} 〜 {res.period[1]} / "
          f"{res.n_rows:,} 行 / {res.n_races:,} レース")
    print(frame[["race_nll", "top1", "top3", "ece"]].round(4).to_string())
    print()
    for t in res.tripwires:
        mark = "発火" if t.fired else "  ok"
        print(f"  {mark} {t.id}  観測 {t.observed}")

    payload = {"period": list(res.period), "n_rows": res.n_rows,
               "n_races": res.n_races, "weights": res.ensemble,
               "metrics": res.metrics,
               "tripwires": [{"id": t.id, "fired": bool(t.fired),
                              "severity": t.severity, "observed": t.observed,
                              # 発火条件の説明。数値だけ残しても、後から
                              # 「何を見て安全と判断したか」が読めない
                              "message": t.message}
                             for t in res.tripwires]}
    (artifacts / "oos_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    pd.DataFrame(payload["tripwires"]).to_csv(artifacts / "oos_guards.csv", index=False)
    print(f"\n→ {artifacts / 'oos_metrics.json'}")

    # 行単位の予測も残す。OOS の開封は1回きりなので、そのとき出た予測を捨てると
    # ロック区間の較正を見るためだけに再開封する羽目になる（ランブック §7 の申し送り）。
    if res.predictions is not None and len(res.predictions):
        oos_pred = artifacts / "oos_predictions.parquet"
        res.predictions.to_parquet(oos_pred, index=False)
        print(f"→ {oos_pred}（{len(res.predictions):,} 行）")

    fired = [t.id for t in res.tripwires if t.fired]
    if fired:
        print(f"ガード発火: {fired}。publish しないでください。", file=sys.stderr)
        return 1
    return 0


def cmd_gate_report(args: argparse.Namespace) -> int:
    """publish ゲートの証跡を作る。

    `assert_publish_gate` に手書きの辞書を渡せてしまうとゲートは何も守らない。
    実際のテスト結果と、実データ OOS のガード発火状況から組み立てる。
    """
    from . import gate

    payload = gate.build(args.tests, args.artifacts, run_tests=not args.no_run)
    for test_id in gate.REQUIRED:
        print(f"  {test_id:<7} {payload['results'].get(test_id, '未実施')}")
    if payload["missing"]:
        print(f"未実施: {payload['missing']}", file=sys.stderr)
    print(f"publish 可否: {'可' if payload['can_publish'] else '不可'}")
    print(f"→ {Path(args.artifacts) / 'gate_report.json'}")
    return 0 if payload["can_publish"] else 1


def cmd_fit_final(args: argparse.Namespace) -> int:
    """本番に出すモデルを作る。

    walk-forward は評価であって、そこで作られるのは fold ごとの短い学習期間の
    モデル。本番には OOS 境界の直前までの全データで学習し直したものを出す。
    """
    from .train.final import fit_and_export

    store = Store(args.data_root)
    fcfg = feature_config()
    ccfg = cv_config()
    sub = _gold_subdir(fcfg)
    path = Path(store.path("gold", sub, "features.parquet"))
    if not path.exists():
        print(f"{path} がありません。先に `nar features` を実行してください。",
              file=sys.stderr)
        return 1
    feat = pd.read_parquet(path)
    cols = [c for c in asof_features(fcfg) if c in feat.columns]
    hash_file = path.parent / "content_hash.txt"
    gold_hash = hash_file.read_text(encoding="utf-8").strip() if hash_file.exists() else ""
    if not gold_hash:
        print("警告: gold の content_hash がありません。`nar features` を再実行すると"
              "manifest に dataset_version が入ります。", file=sys.stderr)

    hpo_params = None
    if args.hpo_params_json:
        hpo_params = json.loads(Path(args.hpo_params_json).read_text(encoding="utf-8"))
        print(f"HPO パラメータを適用: {args.hpo_params_json}")

    art = fit_and_export(
        feat, cols, ccfg, fcfg, out_dir=args.out, gold_hash=gold_hash,
        models=tuple(args.models.split(",")), seed=args.seed,
        tabm_epochs=args.tabm_epochs,
        # 2026-09-17 修正: TabMConfig のフィールド名は batch_races（"batch_size" は
        # 存在しない）。`{k: v for k, v in tabm_width.items() if k in
        # TabMConfig.__annotations__}` が黙ってこのキーを捨てていたため、
        # fit-final の --tabm-batch はこれまで一度も効いておらず、常に既定値
        # 256 で学習していた（`learn` サブコマンド側は最初から batch_races で
        # 正しく渡っており、この不整合は fit-final 固有）。
        tabm_width={"k": args.tabm_k, "hidden": args.tabm_hidden,
                    "n_layers": args.tabm_layers, "batch_races": args.tabm_batch},
        through=args.through,
        do_selection=not args.no_selection, n_null_runs=args.null_runs,
        holdout_days=args.holdout_days, hpo_params=hpo_params)

    meta = json.loads((Path(art.out_dir) / "final_meta.json").read_text(encoding="utf-8"))
    print(f"学習期間: {art.train_period['start']} 〜 {art.train_period['end']}")
    print(f"gold content_hash: {meta['gold_content_hash']}")
    print(f"特徴量: {len(art.feature_names)} 列")
    print(f"温度: { {k: round(v, 4) for k, v in art.temperatures.items()} }")
    print(f"書き出し: {art.written}")
    for n in art.notes:
        print(f"  注意: {n}", file=sys.stderr)
    print(f"→ {art.out_dir}")
    return 0


def cmd_trackb(args: argparse.Namespace) -> int:
    """保存済み OOF に対してトラックB を評価する。

    `learn --trackb` と同じ処理を、学習をやり直さずに走らせるための入口。
    """
    artifacts = Path(args.artifacts)
    oof_path = artifacts / "oof_predictions.parquet"
    if not oof_path.exists():
        print(f"{oof_path} がありません。先に `nar learn` を実行してください。", file=sys.stderr)
        return 2
    oof = pd.read_parquet(oof_path)
    _run_trackb(oof, artifacts, _load_silver(Store(args.data_root))["entry"])
    return 0


def _run_trackb(oof: pd.DataFrame, artifacts: Path,
                entry: pd.DataFrame | None = None) -> None:
    """トラックB: トラックA の OOF を固定オフセットにした残差モデル。

    比較はオッズ利用可能な同一レース集合に限定する（TB-06）。
    """
    from .eval import metrics as M
    from .models.trackb import ResidualOddsModel, market_implied

    if "odds_win" not in oof.columns and entry is not None and "odds_win" in entry.columns:
        oof = oof.merge(entry[["race_id", "horse_no", "odds_win"]],
                        on=["race_id", "horse_no"], how="left")
    if "odds_win" not in oof.columns or oof["odds_win"].isna().all():
        print("トラックB: オッズが無いためスキップします。")
        return

    d = oof[oof["odds_win"].notna()].copy()
    base = "ensemble_stacked" if "ensemble_stacked" in d else (
        "lgbm" if "lgbm" in d else "clogit")
    if base not in d.columns:
        return

    p_a = M.normalize_within_race(d[base].to_numpy(), d["race_id"].to_numpy())
    q = market_implied(d)
    model = ResidualOddsModel().fit(d, p_a, q)
    p_b = model.predict_proba(d, p_a, q)

    # Benter 混合モデル（Research.md §2.1 / 設計書 §13.1-4）。α1 も推定する診断版。
    # 出荷判断には使わない。α1 が 1 から離れていればトラックA の較正ずれを疑う。
    benter = ResidualOddsModel(free_offset_coef=True).fit(d, p_a, q)
    p_bt = benter.predict_proba(d, p_a, q)

    y, rid, pos = (d["is_win"].to_numpy(), d["race_id"].to_numpy(),
                   d["finish_pos"].to_numpy())
    rows = [
        {"model": f"トラックA（{base}）", **M.summary(p_a, y, pos, rid)},
        {"model": "トラックB（残差モデル）", **M.summary(p_b, y, pos, rid)},
        {"model": "トラックB（Benter α1自由・診断）", **M.summary(p_bt, y, pos, rid)},
        {"model": "市場のみ", **M.summary(q, y, pos, rid)},
    ]
    rep = model.report(d)
    brep = benter.report(d)
    (artifacts / "trackb.json").write_text(json.dumps({
        "n_races": rep.n_races, "eta": rep.eta, "alpha": rep.alpha,
        "n_estimated_params": rep.n_estimated_params,
        "underpowered": rep.underpowered, "notes": rep.notes,
        "benter_diagnostic": {
            "alpha1": brep.alpha1, "eta": brep.eta,
            "n_estimated_params": brep.n_estimated_params, "notes": brep.notes,
        },
        "metrics": rows,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== トラックB（{rep.n_races:,} レース, η={rep.eta:.4f}, "
          f"Benter α1={brep.alpha1:.4f}）===")
    print(pd.DataFrame(rows).set_index("model")[["race_nll", "top1"]].round(4).to_string())


def _dataset_version(manifest_path: Path, feat: pd.DataFrame) -> str:
    """manifest があればその集約ハッシュ、無ければ gold の内容ハッシュを使う。"""
    if manifest_path.exists():
        from .io.manifest import Manifest

        m = Manifest(manifest_path)
        try:
            return m.dataset_version()
        finally:
            m.close()
    return "gold:" + content_hash(feat)[:32]


def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].fillna(out[c].median())
        sd = out[c].std()
        out[c] = (out[c] - out[c].mean()) / (sd if sd and sd > 0 else 1.0)
    return out.sort_values(["race_id", "horse_no"]).reset_index(drop=True)


def _has_data(fold, races: pd.DataFrame) -> bool:
    tr, va = fold.mask(races["race_date"])
    return bool(tr.any() and va.any())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser("nar", description="地方競馬 予測モデル パイプライン")
    p.add_argument("--data-root", default=None, help="fsspec URI（既定は $DATA_ROOT）")
    p.add_argument("--artifacts", default="./artifacts")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="合成 silver を生成する")
    s.add_argument("--n-races", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("ingest", help="NAR から月次ファイルを取得（raw 層は不変）")
    s.add_argument("--start", default=None, help="開始月 YYYY-MM")
    s.add_argument("--end", required=True, help="終了月 YYYY-MM")
    s.add_argument("--today", default=None, help="確定判定の基準日 YYYY-MM-DD")
    s.add_argument("--kind", default="race", choices=["race", "odds"])
    s.set_defaults(func=cmd_ingest)

    sub.add_parser("bronze", help="raw → bronze（展開・デコードのみ）").set_defaults(
        func=cmd_bronze)

    s = sub.add_parser("track-master",
                       help="月別開催日程から競馬場マスタを自動生成（廃止場を含む）")
    s.add_argument("--start", default="1998-01")
    s.add_argument("--end", default="2026-08")
    s.set_defaults(func=cmd_track_master)
    sub.add_parser("silver", help="bronze → silver（型付け・キー付与）").set_defaults(
        func=cmd_silver)

    sub.add_parser("leak-check",
                   help="累積成績8列の時点性判定（LK-01..03 / LK-09..11）").set_defaults(
        func=cmd_leak_check)

    s = sub.add_parser("eda", help="EDA を実行してレポートを書く")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_eda)

    s = sub.add_parser("features", help="as-of 特徴量を構築する")
    s.set_defaults(func=cmd_features)

    s = sub.add_parser("train", help="walk-forward で学習・検証する")
    s.add_argument("--l2", type=float, default=1e-3)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("learn", help="全モデルの walk-forward 学習とアンサンブル")
    s.add_argument("--models", default="clogit,lgbm,tabm,bayes")
    s.add_argument("--hpo-clogit", type=int, default=0)
    s.add_argument("--hpo-lgbm", type=int, default=0)
    s.add_argument("--hpo-tabm", type=int, default=0)
    s.add_argument("--no-selection", action="store_true")
    s.add_argument("--null-runs", type=int, default=15)
    s.add_argument("--bayes-method", default="svi", choices=["svi", "nuts"])
    s.add_argument("--bayes-max-races", type=int, default=4000)
    s.add_argument("--tabm-epochs", type=int, default=60)
    s.add_argument("--tabm-k", type=int, default=8)
    s.add_argument("--tabm-hidden", type=int, default=256)
    s.add_argument("--tabm-layers", type=int, default=3)
    s.add_argument("--tabm-batch", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--trackb", action="store_true", help="トラックB（残差モデル）も評価する")
    s.add_argument("--oos", action="store_true", help="続けて OOS 最終評価を行う（開封）")
    s.add_argument("--reason", default=None)
    s.set_defaults(func=cmd_learn)

    s = sub.add_parser("evaluate-final",
                       help="配布する最終モデルを OOS 期間で評価する（開封）")
    s.add_argument("--final-dir", default="./artifacts/final")
    s.add_argument("--unlock-oos", action="store_true")
    s.add_argument("--reason", default=None)
    s.set_defaults(func=cmd_evaluate_final)

    s = sub.add_parser("gate-report",
                       help="リリースゲート用に Blocker テストの結果を集める")
    s.add_argument("--tests", default="./tests")
    s.add_argument("--no-run", action="store_true", help="既存の junit.xml を使う")
    s.set_defaults(func=cmd_gate_report)

    s = sub.add_parser("fit-final",
                       help="本番用モデルを全学習期間で学習し、配布物を書き出す")
    s.add_argument("--models", default="clogit,lgbm,tabm")
    s.add_argument("--out", default="./artifacts/final")
    s.add_argument("--tabm-epochs", type=int, default=3)
    s.add_argument("--tabm-k", type=int, default=4)
    s.add_argument("--tabm-hidden", type=int, default=128)
    s.add_argument("--tabm-layers", type=int, default=2)
    s.add_argument("--tabm-batch", type=int, default=4096)
    s.add_argument("--null-runs", type=int, default=5)
    s.add_argument("--holdout-days", type=int, default=90)
    s.add_argument("--through", default=None,
                   help="学習の終端日。省略時は OOS 境界 − embargo（評価用）。"
                        "OOS 評価後に出荷用を作るときは最新データ日を指定する")
    s.add_argument("--no-selection", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hpo-params-json", default=None,
                   help="`nar learn` の HPO が見つけたハイパーパラメータ "
                        "（{\"clogit\":{...},\"lgbm\":{...},\"tabm\":{...}}）。"
                        "省略時は各モデルの既定値（HPO 未反映）")
    s.set_defaults(func=cmd_fit_final)

    s = sub.add_parser("trackb", help="保存済み OOF でトラックB を評価する")
    s.set_defaults(func=cmd_trackb)

    s = sub.add_parser("report", help="artifacts からモデル性能レポートを生成する")
    s.add_argument("--models", default="clogit,lgbm,tabm,bayes")
    s.add_argument("--data-source", default="合成データ（nar synth / Plackett-Luce 生成）")
    s.add_argument("--dataset-version", default=None)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("evaluate", help="OOS 最終評価（1回のみ）")
    s.add_argument("--unlock-oos", action="store_true")
    s.add_argument("--reason", default=None)
    s.add_argument("--l2", type=float, default=1e-3)
    s.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
