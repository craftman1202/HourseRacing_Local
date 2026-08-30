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
        statement="ばんえいは平地とは別の variant（conf_banei）として独立に学習する",
        rationale="上の除外を維持したまま、ばんえいを予測するための唯一の方法。"
                  "同じ as-of ビルダーを共有し、特徴量集合だけを差し替える。"
                  "ばんえいは 帯広ば(3) だけでなく 北見ば(1)/岩見ば(2)/旭川ば(4) も"
                  "含める（1998-2006 に実在、競技として同一）。",
        impact_if_wrong="平地モデルの列や順序が動けば、配布済みモデルが"
                        "feature_spec 不一致で止まる。asof_features() の"
                        "平地側が不変であることをテストで固定している。",
        evidence="conf_banei/features.yaml / tests/test_banei.py",
    )
    log.add(
        category="exclusion",
        statement="ばんえいでは distance / d_turn_* / d_dist_* / d_best_speed* を使わない",
        rationale="実データで情報を持たないことを確認した。距離は全レース200mで"
                  "分散ゼロ（標準化が壊れる）、ダート左右成績は100%空欄（直線しかない）、"
                  "最高タイムは93.9%空欄、うち当距離成績は当競馬場成績と99.999%一致。",
        impact_if_wrong="全行NaNの列や分散ゼロの列をモデルに渡すことになる。",
        evidence="src/nar/features/banei.py の docstring / data_real 実測",
    )
    log.add(
        category="threshold",
        statement="ばんえいの含水率が 20% を超える行は測定値とみなさず欠損にする",
        rationale="砂は飽和しても重量比 20% 程度。実データでは 2004 年だけ 60-69 が"
                  "766 行あり（他の年は 0-10 に収まる）、単位か記録対象が違う。"
                  "0 で埋めると「乾いた馬場」に化けて意味が反転するので欠損にする。",
        impact_if_wrong="影響は 466,929 行中 766 行（0.16%）。残すと標準化の尺度が"
                        "5 割ふくらみ、線形モデルの係数が歪む。",
        evidence="banei.MOISTURE_MAX / data_real 実測",
    )
    log.add(
        category="limitation",
        statement="ばんえいの馬体重（b_body_weight）は当日ページから取得できる前提を置く",
        rationale="2026-08-29 の帯広の出馬表で、発走前に馬体重・増減が載っていることを"
                  "実ページで確認した。載らない場合は欠損として通し、0 では埋めない。",
        impact_if_wrong="発走13分前にまだ公表されていない開催があると、"
                        "b_body_weight と b_load_ratio が推論時だけ欠損になる。",
        evidence="TodayRaceInfo/DebaTable（k_babaCode=3）の実取得",
    )
    log.add(
        category="method",
        statement="ばんえいの負担重量・馬体重が欠測している出馬表では推論しない",
        rationale="標準化器は欠測を学習時の中央値で埋める。ばんえいの重量は競技の"
                  "ハンデそのもので、埋めた時点で事実と違う前提の予測になる。"
                  "b_weight_rel はレース内の相対量なので、一部の馬だけ埋まると"
                  "馬同士の優劣が直接歪む。実測（2026-08-30）で、開催日の早朝の"
                  "出馬表には馬体重が1頭も載らず、負担重量も10頭中8頭しか"
                  "埋まっていなかった（発走13分前のページには全頭ぶん載る）。",
        impact_if_wrong="掲載が遅い開催で推論が飛ぶ。カバレッジ（SC-06）の低下として"
                        "観測されるので、黙って劣化した予測を配るより検知しやすい。",
        evidence="narops.features.assert_serving_inputs / banei.REQUIRED_AT_SERVING",
    )
    log.add(
        category="method",
        statement="ばんえいの重量系特徴量は妥当域を外れたら推論を止める",
        rationale="当日ページの解析ミスは欠測ではなく異常値として出る。実測"
                  "（2026-08-30）で、馬体重の正規表現が3桁固定だったため 1022kg が"
                  "22kg と読まれていた。2024年以降のばんえい出走馬の 46.3% が"
                  "1000kg 以上なので、毎レース半数近くが壊れていた。平地は"
                  "1000kg 以上が1頭も存在しないため永久に露見しない種類の不具合。",
        impact_if_wrong="「22kg の馬が 610kg を曳く」予測が自信を持って出る。"
                        "該当馬の p_win がほぼ 0 になり、賭け金が系統的に偏る。",
        evidence="banei.SERVING_RANGES / deba_table._WEIGHT_RE",
    )
    log.add(
        category="limitation",
        statement="ばんえいの重量系特徴量は確定層からは再計算できない",
        rationale="運用の確定層（entry_result_final）は負担重量・馬体重・性別・年齢の"
                  "列を持たない（平地モデルが使わないため）。b_* はすべて行ごとに"
                  "閉じた値なので、推論では出馬表から取れており対象レースの値は変わらない。",
        impact_if_wrong="過去のばんえいレースを確定層だけから再計算すると b_* が NaN に"
                        "なる。推論時の値は feature_snapshot に JSON で全列残る（SK-06）。",
        evidence="db/schema.py の entry_result_final DDL",
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
