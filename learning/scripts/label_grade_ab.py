"""LambdaRank のラベル段階数 3 と 5 の A/B（設計書 §13.1-3 / Research.md §2.2）。

`nar learn` の HPO 経由でも選ばせられるが、それだと他のハイパーパラメータと
交絡して「どちらの段階付けが良いか」という問いには答えられない。ここでは
**他の条件を完全に固定**して段階数だけを振り、walk-forward の各 fold で
レース内 NLL（主指標）と Top-1 を比較する。

実行:
    cd learning && PYTHONPATH=src .venv/bin/python scripts/label_grade_ab.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nar.config import cv_config, feature_config
from nar.eval import metrics as M
from nar.eval.splits import make_folds
from nar.features.builder import asof_features
from nar.models.lgbm import LABEL_GRADES, LgbmRanker
from nar.train.pipeline import prepare, trainable

GOLD = Path("data_real/gold/features_noodds/features.parquet")
OUT = Path("artifacts/label_grade_ab.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    # 全 fold × 拡張窓（最大 400 万行）で 10 回学習すると 50 分でも終わらない。
    # 段階数の比較に必要なのは**同一条件での対比**であって全期間の再現ではないので、
    # 直近の fold と固定窓に絞る。この制限は結果報告に明記する。
    ap.add_argument("--folds", default="4,5", help="対象の外側 fold（カンマ区切り）")
    ap.add_argument("--train-years", type=int, default=4,
                    help="学習窓（年）。拡張窓ではなく固定窓にして時間を抑える")
    args = ap.parse_args(argv)
    want = {int(x) for x in args.folds.split(",") if x.strip()}

    fcfg = feature_config()
    ccfg = cv_config()
    feat = trainable(pd.read_parquet(GOLD))
    cols = [c for c in asof_features(fcfg) if c in feat.columns]
    folds = [f for f in make_folds(ccfg, feat["race_date"]) if f.index in want]

    rows = []
    for fold in folds:
        tr_mask, va_mask = fold.mask(feat["race_date"])
        if args.train_years:
            dates = pd.to_datetime(feat["race_date"])
            cutoff = dates[tr_mask].max() - pd.DateOffset(years=args.train_years)
            tr_mask = tr_mask & (dates >= cutoff)
        if not (tr_mask.any() and va_mask.any()):
            continue
        print(f"fold {fold.index}: train {int(tr_mask.sum()):,} 行 / "
              f"valid {int(va_mask.sum()):,} 行", flush=True)
        train = prepare(feat[tr_mask].copy(), cols)
        valid = prepare(feat[va_mask].copy(), cols)
        y = valid["is_win"].to_numpy()
        rid = valid["race_id"].to_numpy()
        pos = valid["finish_pos"].to_numpy()
        for grades in LABEL_GRADES:
            t0 = time.time()
            # 段階数以外は既定値で完全に固定する（seed も共通）
            p = LgbmRanker(label_grades=grades).fit(train, cols).predict_proba(valid)
            s = M.summary(p, y, pos, rid)
            rows.append({"fold": fold.index, "grades": grades,
                         "n_valid_races": int(valid["race_id"].nunique()),
                         "race_nll": s["race_nll"], "top1": s["top1"],
                         "sec": round(time.time() - t0, 1)})
            print(f"fold {fold.index} grades={grades}: NLL={s['race_nll']:.5f} "
                  f"top1={s['top1']:.4f} ({rows[-1]['sec']}s)", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("grades")[["race_nll", "top1"]].mean()
    # fold ごとの対応のある差。fold 数が5しかないので点推定と符号の一貫性を見る。
    wide = df.pivot(index="fold", columns="grades", values="race_nll")
    if 5 not in wide.columns or 3 not in wide.columns:
        print("両方の段階数の結果が揃いませんでした"); return 1
    diff = wide[5] - wide[3]
    print("\n=== fold 平均 ===")
    print(agg.round(5).to_string())
    print(f"\nNLL 差（5段階 − 3段階）fold ごと: {np.round(diff.to_numpy(), 5).tolist()}")
    print(f"平均 {diff.mean():+.5f} / 5段階が良かった fold: {int((diff < 0).sum())}/{len(diff)}")
    winner = int(agg["race_nll"].idxmin())
    print(f"\n主指標（レース内 NLL）で優れるのは {winner} 段階")

    OUT.write_text(json.dumps({
        "setting": {"folds": sorted(want), "train_years": args.train_years,
                    "note": "全 fold・拡張窓では時間内に終わらないため、"
                            "直近 fold と固定窓に絞った比較"},
        "folds": rows,
        "mean_by_grades": agg.round(6).to_dict(orient="index"),
        "nll_diff_5_minus_3": {int(k): float(v) for k, v in diff.items()},
        "winner_by_nll": winner,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
