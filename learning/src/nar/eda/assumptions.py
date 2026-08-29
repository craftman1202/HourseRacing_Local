"""analysis-assumptions-log。

前提条件・除外データ・意思決定の理由を、結果と同じ場所に残す。
納品前に analysis-qa-checklist の手順でここを読み返す。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass
class Assumption:
    id: str
    category: str          # scope / exclusion / method / threshold / limitation
    statement: str
    rationale: str
    impact_if_wrong: str
    evidence: str = ""
    decided_on: str = field(default_factory=lambda: date.today().isoformat())
    status: str = "active"  # active / retired / superseded


class AssumptionLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.items: list[Assumption] = []
        if self.path and self.path.exists():
            self.items = [Assumption(**d) for d in json.loads(self.path.read_text("utf-8"))]

    def add(self, **kw) -> Assumption:
        """同じ前提を二重登録しない。

        EDA を回すたびに seed が積み上がると、記録が「何回実行したか」の履歴になって
        しまい、前提の一覧としては読めなくなる。statement を同一性の基準にする。
        """
        existing = next((a for a in self.items if a.statement == kw.get("statement")), None)
        if existing is not None:
            return existing
        a = Assumption(id=f"A{len(self.items) + 1:03d}", **kw)
        self.items.append(a)
        return a

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(a) for a in self.items], ensure_ascii=False, indent=2), "utf-8"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in self.items])

    def to_markdown(self) -> str:
        if not self.items:
            return "_記録なし_"
        lines = ["| ID | 区分 | 前提 | 理由 | 外れた場合の影響 | 決定日 |",
                 "|---|---|---|---|---|---|"]
        for a in self.items:
            lines.append(
                f"| {a.id} | {a.category} | {a.statement} | {a.rationale} | "
                f"{a.impact_if_wrong} | {a.decided_on} |"
            )
        return "\n".join(lines)


def seed_project_assumptions(log: AssumptionLog) -> AssumptionLog:
    """設計書を読んだ時点で確定している前提。EDA の実測でここが更新される。"""
    log.add(
        category="exclusion",
        statement="ばんえい（baba_code=3）を学習データセットから完全に除外する",
        rationale="距離200m・そりの積載重量という別次元の変数を持つ別競技。"
                  "同一モデルに混ぜると平地・ばんえい双方が劣化する。",
        impact_if_wrong="ばんえいの予測は一切できない。必要なら別モデルを立てる。",
        evidence="設計書 §6",
    )
    log.add(
        category="scope",
        statement="学習の主対象を2010年以降とし、それ以前は種牡馬事前分布の推定にのみ使う",
        rationale="2000年代前半は馬体重・血統の欠損が多いと予想されるため。"
                  "実測はEDA第一問（coverage_map / usable_from_year）で置き換える。",
        impact_if_wrong="学習データが不必要に減る、または欠損だらけの期間を学習してしまう。",
        evidence="設計書 §7 / EDA Q1",
    )
    log.add(
        category="method",
        statement="累積成績8列は原則すべて破棄し、自前の履歴から as-of 集計で再計算する",
        rationale="月次ファイルは毎晩再生成されるため、ファイル生成時点の値である疑いが強い。"
                  "1998年の行に2026年の通算成績が入っていれば未来情報の直接混入になる。",
        impact_if_wrong="破棄しても情報は失われない（同じ集計を自前で作れる）ので損失は無い。"
                        "逆に残して誤っていた場合、全評価が無効になる。",
        evidence="設計書 §5 / テスト仕様 LK-01..03",
    )
    log.add(
        category="threshold",
        statement="embargo 長は特徴量の最大ルックバック窓に等しくする",
        rationale="履歴系特徴量が学習期間末尾の情報を含むため。二重管理を避けるため "
                  "conf/features.yaml の max_lookback_days から自動導出する。",
        impact_if_wrong="valid の特徴量に train 末尾が滲み、CV スコアが楽観的になる。",
        evidence="設計書 §9.1 / テスト仕様 CV-04",
    )
    log.add(
        category="limitation",
        statement="トラックB（オッズ使用）の結果は参考値であり、2年分蓄積まで信頼しない",
        rationale="オッズ情報は2026年3月以降のみ。walk-forward 5-fold を組むと "
                  "1 fold が1か月程度になり統計的検出力が絶望的に不足する。",
        impact_if_wrong="6か月のデータで過学習したモデルを本番投入することになる。",
        evidence="設計書 §10.2 / テスト仕様 TB-01, TB-05",
    )
    log.add(
        category="method",
        statement="前処理は polars ではなく pandas + DuckDB で実装する",
        rationale="実行環境に polars が未導入。as-of 集計の本体は DuckDB の"
                  "ウィンドウ関数なので、性能上の中心は変わらない。",
        impact_if_wrong="数百万行規模で前処理が遅くなる。その時点で polars を導入して"
                        "transform 層だけ差し替える。",
        evidence="設計書 §11 からの逸脱",
    )
    return log
