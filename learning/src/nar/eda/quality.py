"""data-quality-audit の6次元スコアカード。

Completeness / Accuracy / Consistency / Timeliness / Uniqueness / Validity に
各所見をマップし、CRITICAL / HIGH / MEDIUM / LOW を付ける。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DIMENSIONS = {
    "Completeness": 0.20, "Accuracy": 0.20, "Consistency": 0.20,
    "Timeliness": 0.15, "Uniqueness": 0.15, "Validity": 0.10,
}
CRITICAL, HIGH, MEDIUM, LOW = "CRITICAL", "HIGH", "MEDIUM", "LOW"
# 減点しない通知。データの性質として想定どおりだが記録しておきたい事実に使う
# （未実施レースの存在など）。欠陥として数えると品質スコアが実態とずれる。
INFO = "INFO"

# 業務ルール。値域はドメイン知識なので設定ではなくコードに置き、変更を差分で追う。
BUSINESS_RULES = {
    "distance": (200, 4000, "地方の施行距離。200 はばんえいのみ"),
    "time_sec": (10.0, 400.0, "走破タイム（秒）"),
    "odds_win": (1.0, 9999.9, "単勝オッズ。1.0 未満は元返し以下でありえない"),
    "payout_yen": (100, 10_000_000, "払戻は100円単位、下限は元返し"),
    "finish_pos": (1, 20, "着順"),
    "horse_no": (1, 20, "馬番"),
    "weight_kg": (300.0, 700.0, "馬体重（平地）"),
    "n_runners": (2, 20, "頭数。1頭立ては成立しない"),
}

# ばんえいは別競技で、馬体重も所要時間も平地と桁が違う。実データでは 700kg 超が
# 462,279 行あり、その全部がばんえいだった。平地の値域を当てると「重大な品質問題」に
# 化けるので、競技ごとに値域を分ける。
BANEI_BABA_CODES = frozenset({1, 2, 3, 4})
BANEI_RULES = {
    "weight_kg": (600.0, 1400.0, "馬体重（ばんえい）"),
    "time_sec": (60.0, 600.0, "走破タイム（ばんえい・秒）"),
}


@dataclass
class Finding:
    dimension: str
    severity: str
    check: str
    detail: str
    rows_affected: int


@dataclass
class Scorecard:
    findings: list[Finding] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)

    def add(self, dimension: str, severity: str, check: str, detail: str, rows: int) -> None:
        self.findings.append(Finding(dimension, severity, check, detail, rows))

    def score(self) -> float:
        """次元ごとに最悪の所見から減点する。所見が無ければ 10。"""
        penalty = {CRITICAL: 10.0, HIGH: 4.0, MEDIUM: 2.0, LOW: 0.5, INFO: 0.0}
        for dim in DIMENSIONS:
            hits = [f for f in self.findings if f.dimension == dim]
            self.scores[dim] = max(0.0, 10.0 - sum(penalty[f.severity] for f in hits))
        return round(sum(self.scores[d] * w for d, w in DIMENSIONS.items()), 2)

    def verdict(self) -> str:
        s = self.score()
        return "PASS" if s >= 7.0 else ("CONDITIONAL" if s >= 5.0 else "FAIL")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.findings])


def audit(
    entry: pd.DataFrame, race: pd.DataFrame, payout: pd.DataFrame,
    expected_lag_days: int = 2, today: pd.Timestamp | None = None,
) -> Scorecard:
    sc = Scorecard()
    today = today or pd.Timestamp.today().normalize()

    # 1. Completeness
    for col in ("race_id", "horse_sk", "race_date", "horse_no"):
        if col in entry.columns:
            n = int(entry[col].isna().sum())
            if n:
                sc.add("Completeness", CRITICAL, f"必須列 {col} の欠損",
                       f"{col} に NULL が {n} 件。主キー/外部キーの欠損は下流を全部壊す。", n)

    # 2. Uniqueness
    dup_rows = int(entry.duplicated().sum())
    if dup_rows:
        pct = 100 * dup_rows / len(entry)
        sc.add("Uniqueness", CRITICAL if pct > 1 else HIGH, "全行重複",
               f"完全重複 {dup_rows} 行（{pct:.2f}%）。ZIP の二重展開か JOIN の fan-out。", dup_rows)
    key_dup = int(entry.duplicated(["race_id", "horse_no"]).sum())
    if key_dup:
        sc.add("Uniqueness", CRITICAL, "キー重複",
               f"(race_id, horse_no) の重複が {key_dup} 件。粒度が想定と違う。", key_dup)
    if "race_id" in race.columns:
        rdup = int(race.duplicated(["race_id"]).sum())
        if rdup:
            sc.add("Uniqueness", CRITICAL, "race 主キー重複",
                   f"race テーブルの race_id が {rdup} 件重複。", rdup)

    # 3. Validity（値域）。ばんえいは別の値域で見る
    for col, (lo, hi, note) in BUSINESS_RULES.items():
        for name, df in (("entry", entry), ("race", race), ("payout", payout)):
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            is_banei = (df["baba_code"].isin(BANEI_BABA_CODES)
                        if "baba_code" in df.columns
                        else pd.Series(False, index=df.index))
            for label, mask, (blo, bhi, bnote) in (
                ("", ~is_banei, (lo, hi, note)),
                ("（ばんえい）", is_banei, BANEI_RULES.get(col, (lo, hi, note))),
            ):
                if not mask.any():
                    continue
                v = values[mask]
                bad = int(((v < blo) | (v > bhi)).sum())
                if bad:
                    sc.add("Validity", HIGH, f"{name}.{col} 値域外{label}",
                           f"[{blo}, {bhi}] の外が {bad} 件（{bnote}）。", bad)

    # 4. Accuracy（クロスフィールド整合）
    if {"race_id", "finish_pos"} <= set(entry.columns):
        per_race = entry.groupby("race_id")["finish_pos"]
        # 着順が1行も入っていないレースは「まだ走っていない／中止」であって
        # パースの失敗ではない。1着だけが欠けているレースと区別する。
        all_missing = per_race.apply(lambda s: s.isna().all())
        unrun = int(all_missing.sum())
        no_winner = int(((per_race.min() != 1) & ~all_missing).sum())
        if no_winner:
            sc.add("Accuracy", CRITICAL, "1着不在",
                   f"着順はあるのに1着が無いレースが {no_winner} 件。着順のパースを疑う。",
                   no_winner)
        if unrun:
            sc.add("Completeness", MEDIUM, "結果なしレース",
                   f"着順が1行も無いレースが {unrun} 件（未実施・中止）。"
                   "ラベルが無いので学習からは外れる。", unrun)
    if {"race_id", "n_runners"} <= set(race.columns):
        counted = entry.groupby("race_id").size().rename("counted")
        merged = race.set_index("race_id").join(counted)
        mism = int((merged["n_runners"] != merged["counted"]).sum())
        if mism:
            sc.add("Accuracy", HIGH, "頭数不一致",
                   f"race.n_runners と出馬表の行数が {mism} 件不一致。"
                   "頭数列は出走取消の反映時点が不明なので行数側を採用する。", mism)

    # 5. Consistency（参照整合性）
    # 中止になった開催は出馬表だけ公開されてレース一覧に行が残らない
    # （例: 2018-09-05 園田、台風で中止。出馬表 92 行に対しレース 0 行）。
    # 着順を持つ孤児だけが本当の異常で、それはレース側を取りこぼしたことを意味する。
    is_orphan = ~entry["race_id"].isin(set(race["race_id"]))
    has_result = entry.get("finish_pos", pd.Series(np.nan, index=entry.index)).notna()
    lost = int((is_orphan & has_result).sum())
    cancelled = int((is_orphan & ~has_result).sum())
    if lost:
        sc.add("Consistency", CRITICAL, "entry → race の孤児（結果あり）",
               f"着順があるのに race に無い race_id が {lost} 件。"
               "レース一覧の取りこぼしを疑う。", lost)
    if cancelled:
        sc.add("Consistency", LOW, "entry → race の孤児（結果なし）",
               f"出馬表だけあってレース一覧に無い行が {cancelled} 件（開催中止）。"
               "ラベルもレース属性も無いので学習からは外れる。", cancelled)
    if len(payout):
        p_orphan = int((~payout["race_id"].isin(set(race["race_id"]))).sum())
        if p_orphan:
            sc.add("Consistency", HIGH, "payout → race の孤児",
                   f"race に存在しない race_id の払戻が {p_orphan} 件。", p_orphan)

    # 6. Timeliness
    if "race_date" in entry.columns:
        latest = pd.to_datetime(entry["race_date"]).max()
        lag = (today - latest).days
        if lag > expected_lag_days:
            sc.add("Timeliness", MEDIUM if lag < 30 else HIGH, "鮮度",
                   f"最新レースが {latest.date()}（{lag} 日前）。想定 {expected_lag_days} 日。", 0)
        # 月次ファイルには当月の未実施レースも入る。着順が無ければ「予定」であって
        # 時点の取り違えではない。着順が入った未来日付だけが本当の異常。
        future_mask = pd.to_datetime(entry["race_date"]) > today
        finished = entry.get("finish_pos", pd.Series(np.nan, index=entry.index)).notna()
        scheduled = int((future_mask & ~finished).sum())
        impossible = int((future_mask & finished).sum())
        if impossible:
            sc.add("Timeliness", CRITICAL, "未来日付の確定結果",
                   f"今日より後の race_date に着順が {impossible} 件。時点の取り違えを疑う。",
                   impossible)
        if scheduled:
            sc.add("Timeliness", INFO, "未実施レース",
                   f"今日より後に予定されたレースが {scheduled} 行。推論対象であって欠陥ではない。",
                   scheduled)
    return sc
