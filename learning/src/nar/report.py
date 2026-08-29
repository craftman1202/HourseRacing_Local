"""モデル性能レポートの生成。

artifacts/ に落ちた CV・アンサンブル・OOS の結果から Markdown を組み立てる。
数値を手で書き写すと必ずズレるので、レポートは成果物から機械的に作る。

不確実性やデータ品質上の懸念は結果と同じ場所に書く（~/.claude/CLAUDE.md）。
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .eda.report import _md

MODEL_LABELS = {
    "clogit": "条件付きロジット（正則化付き McFadden）",
    "lgbm": "LightGBM LambdaRank",
    "tabm": "TabM（パラメータ効率的アンサンブル）",
    "bayes": "階層ベイズ動的 Plackett-Luce",
    "ensemble_stacked": "アンサンブル（制約付きスタッキング）",
    "ensemble_simple_avg": "アンサンブル（単純平均）",
    "ensemble_log_avg": "アンサンブル（対数平均）",
    "baseline_market": "ベースライン: 市場オッズ正規化",
    "baseline_uniform": "ベースライン: レース内一様分布",
}

METRIC_COLS = ["race_nll", "brier", "top1", "top3", "ndcg@3", "spearman", "ece"]


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _fmt(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).round(4)
    return out


def _oos_frame(artifacts: Path) -> pd.DataFrame:
    """OOS の結果表を読む。

    `evaluate-final`（配布物そのものを測る経路）は JSON で書く。旧来の
    `learn --oos` は CSV。どちらでも読めるようにしないと、配布物を測った
    数字がレポートに出ないまま「未開封」と表示される。
    """
    js = artifacts / "oos_metrics.json"
    if js.exists():
        payload = _read_json(js)
        block = payload.get("metrics") if isinstance(payload, dict) else None
        if isinstance(block, dict) and block:
            frame = pd.DataFrame(block).T.reset_index().rename(
                columns={"index": "model"})
            return frame.sort_values("race_nll").reset_index(drop=True)
    return _read_csv(artifacts / "oos_metrics.csv")


def build(artifacts: Path, meta: dict) -> str:
    """レポート本文を組み立てる。"""
    fold = _read_csv(artifacts / "fold_metrics.csv")
    ens = _read_csv(artifacts / "ensemble_metrics.csv")
    weights = _read_csv(artifacts / "ensemble_weights.csv")
    oos = _oos_frame(artifacts)
    guard = _read_csv(artifacts / "oos_guards.csv")
    trackb = _read_json(artifacts / "trackb.json")
    diag = _read_json(artifacts / "bayes_diagnostics.json")
    timings = _read_csv(artifacts / "timings.csv")
    selection = _read_csv(artifacts / "selected_features.csv")

    lines: list[str] = []
    a = lines.append

    a("# モデル性能レポート — 地方競馬 予測モデル")
    a("")
    a(f"- 実施日: {date.today().isoformat()}")
    a(f"- データ: {meta.get('data_source')}")
    a(f"- 規模: レース {meta.get('n_races'):,} / 出走 {meta.get('n_entries'):,} 件"
      f"（{meta.get('period')}）")
    a(f"- `dataset_version`: `{meta.get('dataset_version', '')[:24]}…`")
    a(f"- 学習モデル: {', '.join(_label(m) for m in meta.get('models', []))}")
    a(f"- embargo: {meta.get('embargo_days')} 日"
      "（`conf/features.yaml` の `max_lookback_days` から自動導出）")
    a("")
    # データ源に応じて前置きを変える。合成データの注意書きを実データのレポートに
    # 残すと、読み手が数値の意味を取り違える。
    if meta.get("is_synthetic", True):
        a("> **読む前に**: 本レポートの数値は**合成データ**に対するものです。実データが")
        a("> 1バイトも無い段階でパイプライン全体を通すために、Plackett-Luce に従う")
        a("> 合成レースを生成して学習しています。**モデル間の相対比較と実装の健全性**は")
        a("> 読み取れますが、**絶対水準は実データの性能を意味しません**。とくに市場")
        a("> ベースラインとの優劣は、合成データの構造上ほぼ意味を持ちません（§6 参照）。")
    else:
        a("> **読む前に**: 本レポートの数値は NAR の月次ファイル由来の**実データ**に")
        a("> 対するものです。検証区間は walk-forward の out-of-fold であり、")
        a("> OOS（ロック区間）とは別です。OOS の数値は §7 にあり、開封回数も")
        a("> そこに記録しています。")
    a("")

    # --------------------------------------------------------------- CV
    a("## 1. walk-forward 交差検証（5-fold）")
    a("")
    a("主指標はレース内 NLL。後段の期待値計算が確率の較正精度に直接依存するため、")
    a("Top-1 精度や NDCG ではなく NLL で優劣を判断します。")
    a("")
    if len(fold):
        mean = (fold.groupby("model")[METRIC_COLS].mean()
                .sort_values("race_nll").reset_index())
        mean.insert(1, "モデル", mean["model"].map(_label))
        a(_md(_fmt(mean.drop(columns=["model"]), METRIC_COLS), 20))
        a("")
        a("### fold 別のレース内 NLL")
        a("")
        pivot = fold.pivot_table(index="model", columns="fold", values="race_nll").round(4)
        pivot.insert(0, "平均", pivot.mean(axis=1).round(4))
        pivot = pivot.sort_values("平均").reset_index()
        pivot["model"] = pivot["model"].map(_label)
        a(_md(pivot.rename(columns={"model": "モデル"}), 20))
        a("")
        spread = fold.groupby("model")["race_nll"].std().round(4)
        a(f"fold 間の標準偏差が最も大きいのは **{_label(spread.idxmax())}**"
          f"（{spread.max():.4f}）。分割ごとの当たり外れが大きいモデルは、"
          "点推定の平均だけで採否を決めない。")
        a("")
    else:
        a("_CV 結果がありません_")
        a("")

    # --------------------------------------------------------- アンサンブル
    a("## 2. アンサンブル（OOF のみから重み推定）")
    a("")
    a("重みは walk-forward の out-of-fold 予測のみから推定しています。訓練内予測を")
    a("混ぜると重みが致命的にバイアスするため、fold 割り当ての無い行が混入したら")
    a("例外を送出する実装にしてあります（EN-02）。")
    a("")
    if len(ens):
        e = ens.copy()
        e.insert(0, "モデル", e["model"].map(_label))
        a(_md(_fmt(e.drop(columns=["model"]), METRIC_COLS), 10))
        a("")
    if len(weights):
        w = weights.copy()
        w["モデル"] = w["model"].map(_label)
        w["weight"] = w["weight"].round(4)
        a(_md(w[["モデル", "weight"]]))
        a("")
        top = w.loc[w["weight"].idxmax()]
        a(f"重みは **{top['モデル']}** に {top['weight']:.1%} 集中しています。")
        a("")

    # ------------------------------------------------------------- 特徴量選択
    if len(selection):
        a("## 3. 特徴量選択（fold 内実行）")
        a("")
        a("Null Importance → 相関/VIF → RFE の3段を、**各 fold の学習期間内でのみ**")
        a("実行しています。全期間で選んでから CV すると選択バイアスが入ります（CV-10）。")
        a("")
        a(_md(selection, 12))
        a("")
        if selection["selected"].nunique() > 1:
            a("fold ごとに選択結果が異なっており、fold 内実行が効いていることが確認できます。")
        else:
            a("**注意**: 全 fold で選択結果が同一です。fold 内実行が効いていない可能性が")
            a("あるため、`features/selection.py` の呼び出し経路を確認してください（CV-10）。")
        a("")

    # ------------------------------------------------------------- ベイズ診断
    if diag:
        a("## 4. ベイズモデルの収束診断")
        a("")
        a("収束診断を通らないモデルは採用しない、という方針で運用します。")
        a("")
        rows = [{"指標": k, "値": v} for k, v in diag.items() if not isinstance(v, (list, dict))]
        a(_md(pd.DataFrame(rows)))
        a("")
        method = diag.get("method", "unknown")
        rhat, ess, ndiv = (diag.get("max_rhat"), diag.get("min_ess_bulk"),
                           diag.get("n_divergent"))
        if method == "nuts" and ndiv is not None:
            a(f"- 発散遷移: **{ndiv} 件**（仕様 MD-18 の要件は 0 件）")
            if rhat is not None:
                ok = "満たす" if rhat < 1.01 else "**満たさない**"
                a(f"- R-hat 最大: **{rhat:.4f}** → 仕様 MD-16（< 1.01）を{ok}")
            if ess is not None:
                ok = "満たす" if ess > 400 else "**満たさない**"
                a(f"- ESS(bulk) 最小: **{ess:.0f}** → 仕様 MD-17（> 400）を{ok}")
        else:
            a(f"- 推論方式: **{method.upper()}**（変分近似）。")
            a("- **R-hat / ESS / 発散遷移（MD-16〜18）は MCMC の診断なので、この実行では")
            a("  評価していません。** 設計書 §8.4 の二段構えのうち SVI 側だけを回しており、")
            a("  最終窓の NUTS 検証は未実施です。SVI で確認できるのは ELBO が改善したこと")
            a(f"  （{diag.get('svi_loss_improved')}）だけで、収束の保証にはなりません。")
            a("- NUTS を回した実測値はテスト（`test_models_deep.py`）に記録してあります:")
            a("  250レース規模で R-hat 1.035 / ESS(bulk) 119 / 発散遷移 0 件。")
            a("  仕様値（R-hat < 1.01、ESS > 400）にはサンプル数が届いていません。")
        a("")

    # ---------------------------------------------------------------- OOS
    a("## 5. OOS 最終評価")
    a("")
    if len(oos):
        a(f"OOS 期間: {meta.get('oos_period')}。開封回数 "
          f"**{meta.get('oos_access_count')} 回**（`artifacts/oos_access.log` に記録）。")
        a("")
        o = oos.copy()
        o.insert(0, "モデル", o["model"].map(_label))
        a(_md(_fmt(o.drop(columns=["model"]), METRIC_COLS), 15))
        a("")
        best = o.iloc[0]
        a(f"OOS 最良は **{best['モデル']}**（NLL {best['race_nll']:.4f}）。")
        cv_best = fold.groupby("model")["race_nll"].mean().min() if len(fold) else np.nan
        if np.isfinite(cv_best):
            gap = float(best["race_nll"]) - float(cv_best)
            a(f"CV 最良平均 {cv_best:.4f} との差は {gap:+.4f}。"
              + ("CV が OOS より 0.10 以上良い場合は RF-05（分割リークまたは過学習）"
                 "が発火します。" if gap >= 0.10 else
                 "CV と OOS が整合しており、分割リークの兆候はありません。"))
        a("")
    else:
        a("_OOS は未開封です。最終評価は `--oos` を付けて1回だけ実行してください。_")
        a("")

    # ------------------------------------------------------------ ガード
    a("## 6. Too-Good-To-Be-True ガード")
    a("")
    a("競馬の市場効率性と控除率の水準を踏まえると、以下の数値は成功ではなくバグの")
    a("徴候です。発火したら成果とみなさず、即座にリーク調査へ回します。")
    a("")
    if len(guard):
        g = guard.copy()
        g["fired"] = g["fired"].map({True: "**発火**", False: "未発火"})
        a(_md(g[["id", "fired", "severity", "observed", "message"]]))
        a("")
        n_fired = int((guard["fired"]).sum())
        a(f"発火 **{n_fired} 件**。" + ("リーク調査が必要です。" if n_fired else
                                        "リークの兆候はありません。"))
        a("")

    # ----------------------------------------------------------- トラックB
    a("## 7. トラックB（オッズ使用）")
    a("")
    if trackb:
        a(f"- オッズ利用可能レース: **{trackb.get('n_races'):,} 件**")
        a(f"- 推定パラメータ数: **{trackb.get('n_estimated_params')} 個**"
          "（η と α のみ。`log p^A` の係数は 1.0 に固定）")
        a(f"- η = {trackb.get('eta'):.4f}")
        if trackb.get("metrics"):
            m = pd.DataFrame(trackb["metrics"])
            a("")
            a(_md(_fmt(m, METRIC_COLS), 10))
        a("")
        for note in trackb.get("notes", []):
            a(f"> {note}")
        a("")
    else:
        a("_トラックB は未実行です。_")
        a("")

    # ------------------------------------------------------------ 実行時間
    if len(timings):
        a("## 8. 実行時間")
        a("")
        a(_md(timings.round(1), 20))
        a("")

    # ---------------------------------------------------- 結論と残る不確実性
    a("## 9. 結論")
    a("")
    for line in meta.get("conclusions", []):
        a(f"- {line}")
    a("")
    a("## 10. この結果の限界")
    a("")
    for line in meta.get("limitations", []):
        a(f"- {line}")
    a("")
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write(artifacts: Path, meta: dict, out_path: Path | None = None) -> Path:
    out = out_path or (artifacts / "model_performance.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(artifacts, meta), encoding="utf-8")
    return out
