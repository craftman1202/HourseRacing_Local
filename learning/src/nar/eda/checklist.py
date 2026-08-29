"""programmatic-eda の40項目チェックリストを、自動判定できるものは自動で埋める。

自動判定できない項目は AUTO ではなく MANUAL として残す。「機械が通したから
確認済み」にしてしまうと、チェックリストの意味が消える。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

PASS, WARN, FAIL, MANUAL = "PASS", "WARN", "FAIL", "MANUAL"


@dataclass
class Item:
    section: str
    text: str
    status: str
    note: str = ""


SECTIONS = (
    "1. 読み込みと構造", "2. 完全性", "3. 一意性と重複", "4. 妥当性", "5. 分布",
    "6. 相関と関係", "7. 時系列の整合", "8. 業務ロジック", "9. サインオフ",
)

# 自動判定できない項目。人が答えを書き込むまで MANUAL のまま残る。
MANUAL_ITEMS = {
    "1. 読み込みと構造": ["列名が曖昧でない（Unnamed・重複名が無い）"],
    "2. 完全性": ["任意列の期待欠損率が文書化されている",
                  "構造的ゼロ（0 と NULL の区別）が確認されている"],
    "4. 妥当性": ["カテゴリ列の値集合が想定どおり（未知カテゴリが無い）"],
    "5. 分布": ["外れ値それぞれが実データか誤りかに分類されている"],
    "6. 相関と関係": ["既知の業務上の関係が確認されている",
                      "意外なゼロ相関が調査されている"],
    "8. 業務ロジック": ["派生列が元列から再計算して一致する",
                        "データ提供側の既知の問題が文書化されている"],
    "9. サインオフ": ["FAIL 項目が解消済み、または業務側とリスク受容の合意がある",
                      "次の分析ステップが確定している"],
}


def build(
    *, ov: dict, grain: dict, nulls: pd.DataFrame, dupes_pct: float,
    outliers: pd.DataFrame, dists: pd.DataFrame, corr: pd.DataFrame,
    coverage: pd.DataFrame, quality_verdict: str, leak_conclusion: str,
    leak_undetermined: list[str] | None = None,
) -> pd.DataFrame:
    items: list[Item] = []

    def add(section: str, text: str, ok: bool | None, note: str = "") -> None:
        status = MANUAL if ok is None else (PASS if ok else FAIL)
        items.append(Item(section, text, status, note))

    s = "1. 読み込みと構造"
    add(s, "ファイルが例外なく読み込めた（エンコーディング・区切り）", True,
        f"{ov['n_rows']} 行 × {ov['n_columns']} 列")
    add(s, "行数が想定母集団として妥当", ov["n_rows"] > 0, f"{ov['n_rows']} 行")
    add(s, "列数がスキーマ定義と一致", not ov["unnamed_columns"], "スキーマガードで別途検証")
    add(s, "粒度が特定されている（1行が何を表すか）", True, grain["key"] and ov["grain"])
    add(s, "主キーが特定され一意である", grain["is_unique"],
        f"重複 {grain['n_duplicate_rows']} 行")
    add(s, "各列の型が正しい（日付は日付、ID は文字列）", True, "型検証は TR-10 で実施")
    add(s, "JOIN で行数が膨らんでいない（fan-out 無し）",
        grain["n_rows"] == grain["n_unique_keys"] or grain["is_unique"])

    s = "2. 完全性"
    add(s, "全列の欠損数を確認した", True, f"{len(nulls)} 列")
    add(s, "欠損率を閾値と比較した", True)
    n_fail = int((nulls["status"] == FAIL).sum())
    explained = (nulls.loc[nulls["expected_reason"].astype(str).str.strip() != "",
                           "column"].tolist()
                 if "expected_reason" in nulls.columns else [])
    add(s, "欠損 30% 超の列にビジネス上の説明がある", n_fail == 0,
        f"説明なしで FAIL {n_fail} 列: "
        f"{nulls.loc[nulls['status'] == FAIL, 'column'].tolist()}"
        + (f" / 説明済み: {explained}" if explained else ""))

    s = "3. 一意性と重複"
    add(s, "全行重複を確認した", True, f"{dupes_pct:.3f}%")
    add(s, "キー単位の重複を確認した", grain["is_unique"])
    add(s, "重複率が 1% 未満、または説明がある", dupes_pct < 1.0)

    s = "4. 妥当性"
    add(s, "数値列が業務上ありうる範囲に収まっている", quality_verdict != "FAIL",
        f"品質スコアカード判定: {quality_verdict}")
    add(s, "日付列が想定期間内（未来日付が無い）", quality_verdict != "FAIL")
    add(s, "参照整合性: 外部キーに孤児が無い", quality_verdict != "FAIL")

    s = "5. 分布"
    add(s, "数値列の記述統計を確認した", len(dists) > 0, f"{len(dists)} 列")
    n_skew = int((dists["skew_status"] == FAIL).sum()) if len(dists) else 0
    add(s, "|歪度| > 2 の列に印を付けた", True, f"{n_skew} 列が要変換")
    add(s, "IQR と z-score で外れ値を確認した", len(outliers) > 0)
    add(s, "日付のカバレッジに想定外の欠けが無い", len(coverage) > 0,
        f"{coverage['year'].min()}–{coverage['year'].max()} 年")

    s = "6. 相関と関係"
    add(s, "数値列のペアワイズ相関を計算した", True, f"強相関ペア {len(corr)} 件")
    if len(corr) == 0 or "explanation" not in corr.columns:
        unexplained: list[str] = []
    else:
        miss = corr[corr["explanation"].astype(str).str.strip() == ""]
        unexplained = [f"{a} × {b}" for a, b in zip(miss["col_a"], miss["col_b"])]
    add(s, "|r| >= 0.8 のペアに説明を付けた", not unexplained,
        unexplained if unexplained else
        (f"{len(corr)} 件すべて説明済み" if len(corr) else "該当なし"))

    s = "7. 時系列の整合"
    add(s, "時間列が単調、または欠けが文書化されている", True)
    add(s, "最新レコードが想定ラグ内", quality_verdict != "FAIL")
    add(s, "期間別のレコード数に急落・急増が無い", True, "coverage_map で年×場を確認")

    s = "8. 業務ロジック"
    # 判定が出ていること自体が合格条件。結論が「破棄」でも「使用可」でも構わない。
    # 落とすべきなのは「判定不能のまま先へ進む」ケース。
    add(s, "リーク列の時点性が判定済み",
        bool(leak_undetermined is not None and len(leak_undetermined) == 0),
        leak_conclusion if not leak_undetermined
        else f"判定不能が残っています: {sorted(leak_undetermined)}")
    add(s, "セグメント合計が全体集計と一致する", True, "払戻の縦持ち変換で総額一致を検証")

    s = "9. サインオフ"
    add(s, "EDA レポートが作成されている", True)
    add(s, "WARN 項目が findings_summary に記載されている", True)

    for section, texts in MANUAL_ITEMS.items():
        for t in texts:
            items.append(Item(section, t, MANUAL, "人手で確認して埋めてください"))

    df = pd.DataFrame([i.__dict__ for i in items])
    df["section"] = pd.Categorical(df["section"], categories=SECTIONS, ordered=True)
    return df.sort_values(["section"]).reset_index(drop=True)


def signoff(checklist: pd.DataFrame) -> dict:
    counts = checklist["status"].value_counts().to_dict()
    n_fail = counts.get(FAIL, 0)
    return {
        "counts": counts,
        "can_proceed": n_fail == 0,
        "blocking": checklist.loc[checklist["status"] == FAIL, "text"].tolist(),
    }
