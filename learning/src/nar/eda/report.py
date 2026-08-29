"""EDA レポート生成。

programmatic-eda の2成果物（フルレポート + findings_summary）と、
data-quality-audit のスコアカードを同じディレクトリに出す。
不確実性やデータ品質上の懸念は結果と同じ場所に書く。隠さない。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def _cell(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("|", "\\|").replace("\n", " ")


def _md(df: pd.DataFrame, max_rows: int = 30) -> str:
    """DataFrame を GFM テーブルにする。

    pandas.to_markdown は tabulate を要求する。レポート整形のためだけに依存を
    増やしたくないので、ここで直接書く。
    """
    if df is None or len(df) == 0:
        return "_該当なし_"
    head = df.head(max_rows)
    cols = [str(c) for c in head.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in head.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    body = "\n".join(lines)
    if len(df) > max_rows:
        body += f"\n\n_（{len(df)} 行中 {max_rows} 行を表示）_"
    return body


def write_report(result: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ov, sc = result["overview"], result["scorecard"]

    lines = [
        f"# EDA レポート — 地方競馬 予測モデル",
        "",
        f"- 実施日: {date.today().isoformat()}",
        f"- 粒度: **{ov['grain']}**",
        f"- 規模: {ov['n_rows']:,} 行 × {ov['n_columns']} 列（{ov['memory_mb']} MB）",
        f"- データセット判定: **{sc['verdict']}**（総合 {sc['score']} / 10）",
        "",
        "> 本レポートは programmatic-eda の手順（構造 → 欠損 → 外れ値 → 分布 → 相関 →",
        "> チェックリスト → 報告）に data-quality-audit の6次元スコアカードと、",
        "> 設計書 §7 の6問を重ねたもの。",
        "",
        "## 0. 前提条件と意思決定の記録",
        "",
        result["assumptions_md"],
        "",
        "## 1. 構造と粒度",
        "",
        f"- 主キー: `{result['grain']['key']}`",
        f"- 一意性: **{'OK' if result['grain']['is_unique'] else 'NG'}**"
        f"（重複 {result['grain']['n_duplicate_rows']} 行）",
        "",
        "## 2. 欠損プロファイル",
        "",
        _md(result["nulls"][result["nulls"]["null_pct"] > 0]),
        "",
        "### 年別の欠損率（MCAR らしさの確認）",
        "",
        _md(result["missingness_by_year"], 20),
        "",
        "## 3. 外れ値",
        "",
        "IQR と z-score の両方で検出。除去はせず、実データ／誤り／構造的センチネルの",
        "いずれかに分類する（`classification` 列を人手で埋めること）。",
        "",
        _md(result["outliers"]),
        "",
        "## 4. 分布",
        "",
        _md(result["distributions"]),
        "",
        "## 5. 相関",
        "",
        _md(result["correlations"]),
        "",
        "## 6. データ品質スコアカード（6次元）",
        "",
        _md(pd.DataFrame([
            {"次元": k, "重み": f"{int(v * 100)}%", "スコア": sc["scores"].get(k)}
            for k, v in sc["weights"].items()
        ])),
        "",
        f"**総合 {sc['score']} / 10 → {sc['verdict']}**",
        "",
        "### 所見",
        "",
        _md(sc["findings"]),
        "",
        "## Q1. 時系列カバレッジと欠損の地図",
        "",
        f"- 提案する学習開始年: **{result['usable']['proposed_train_from']}**",
        f"- 判定基準: {result['usable']['criteria']}",
        f"- {result['usable']['note']}",
        "",
        _md(result["usable"]["by_year"], 40),
        "",
        "## Q2. リーク検証（最優先）— 累積成績8列の時点性",
        "",
        f"**結論: {result['leak']['conclusion']}**",
        "",
        f"- 使用可（ホワイトリスト）: `{result['leak']['whitelist']}`",
        f"- 破棄: `{result['leak']['discard']}`",
        f"- 判定不能（破棄側に倒す）: `{result['leak']['undetermined']}`",
        "",
        _md(result["leak_detail"]),
        "",
        "判定不能な列を「使う」側に倒してはいけない。破棄しても情報は失われない",
        "（1998年からの全レース結果があるので同じ集計を自前で作れる）が、",
        "誤って残した場合は全評価が無効になる。非対称なので破棄側に倒す。",
        "",
        "### ラベルとの相関スクリーニング",
        "",
        _md(result["label_corr"], 20),
        "",
        "## Q3. 名寄せの検証",
        "",
        _md(pd.DataFrame([result["identity"]]).T.reset_index()
            .rename(columns={"index": "項目", 0: "値"})),
        "",
        "## Q4. ターゲットの基礎統計と favourite-longshot bias",
        "",
        "### 人気別勝率",
        "",
        _md(result["target"].get("popularity_winrate"), 18),
        "",
        "### 市場暗黙確率の較正曲線",
        "",
        "`bias = 実測勝率 − 市場暗黙確率`。低確率帯で負なら穴馬が過剰に買われている。",
        "",
        _md(result["flb"], 15),
        "",
        "## Q5. 分布シフト（年次 PSI）",
        "",
        "### 構造変化点",
        "",
        _md(result["change_points"]),
        "",
        "複数列が同時にシフトした年のみを変化点とみなす。walk-forward 分割の",
        "設計に直接影響するので、fold 境界がこれらの年をまたがないか確認すること。",
        "",
        _md(result["drift"][result["drift"]["status"] != "PASS"], 25),
        "",
        "## Q6. 控除率の実測",
        "",
        "期待値計算に使う控除率は公称値ではなく、ここで出した実測値を使う。",
        "",
        _md(result["takeout"]),
        "",
        "## チェックリスト（40項目）",
        "",
        _md(result["checklist"], 60),
        "",
        f"**サインオフ: {result['signoff']['counts']}**",
        "",
        ("次工程へ進めます。" if result["signoff"]["can_proceed"]
         else f"**進行不可。未解消: {result['signoff']['blocking']}**"),
        "",
    ]
    path = out / "eda_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_findings_summary(result: dict, out_dir: str | Path, top_n: int = 5) -> Path:
    """上位3〜5件の品質問題と次のアクション。読む人が最初に開く1枚。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = result["scorecard"]
    findings = sc["findings"]

    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    top = (
        findings.assign(_o=findings["severity"].map(order)).sort_values(
            ["_o", "rows_affected"], ascending=[True, False]).head(top_n)
        if len(findings) else findings
    )

    lines = [
        "# EDA 所見サマリ",
        "",
        f"- 実施日: {date.today().isoformat()}",
        f"- 総合品質: **{sc['verdict']}**（{sc['score']} / 10）",
        f"- リーク判定: **{result['leak']['conclusion']}**",
        f"- 提案する学習開始年: **{result['usable']['proposed_train_from']}**",
        "",
        "## 上位の品質問題",
        "",
        _md(top.drop(columns=["_o"]) if "_o" in top.columns else top),
        "",
        "## 次のアクション",
        "",
        "1. Q2 のリーク判定結果を `transform/prerace.py` の `ASOF_RACE_WHITELIST` に反映する"
        "（現状は空のまま = 8列すべて破棄）。",
        "2. Q1 の `proposed_train_from` を `conf/features.yaml` の `train_from` に反映する。",
        "3. Q5 の構造変化点が walk-forward の fold 境界をまたいでいないか確認する。",
        "4. Q6 の実測控除率を期待値計算に使う（公称値を使わない）。",
        "5. 外れ値の `classification` 列を人手で埋める（実データ／誤り／構造的）。",
        "",
        "## 残る不確実性",
        "",
        "- 累積成績8列の LK-01（世代間差分）は、当日ファイルと後日の月次ファイルが",
        "  両方揃うまで実行できない。日次取得を今日から回しておくこと（過去は取り返せない）。",
        "- 本レポートの数値が合成データ由来の場合、閾値は参照値であって実測ではない。",
        "",
    ]
    path = out / "findings_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
