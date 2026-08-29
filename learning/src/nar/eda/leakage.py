"""EDA 第二問（最優先）: 累積成績8列の時点性判定。

月次ファイルは毎晩再生成されるため、`全成績` などが「当該レース発走時点」ではなく
「ファイル生成時点」の値である可能性が高い。もしそうなら1998年のレース行に2026年の
通算成績が入っていることになり、未来情報の直接混入になる。

LK-01（世代間差分）→ LK-02（単調性）→ LK-03（終端一致）の順に判定し、
どれか1つでも as-of-download を示したらホワイトリストは空のまま確定させる。
判定を保留したまま先に進んではいけない。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..transform.prerace import CUMULATIVE_RECORD_COLS

Verdict = Literal["as_of_download", "as_of_race", "undetermined"]

RECORD_RE = re.compile(r"^\s*(\d+)-(\d+)-(\d+)-(\d+)\s*$")


@dataclass
class ColumnVerdict:
    column: str
    test: str
    verdict: Verdict
    evidence: str
    usable: bool


def parse_record(s: pd.Series) -> pd.DataFrame:
    """`10-1-3-16` 形式を 1着/2着/3着/着外 に分解し、総出走数を出す。"""
    parsed = s.astype(str).str.extract(RECORD_RE)
    parsed.columns = ["first", "second", "third", "others"]
    out = parsed.apply(pd.to_numeric, errors="coerce")
    out["starts"] = out.sum(axis=1, skipna=False)
    return out


def lk01_generation_diff(
    daily: pd.DataFrame, monthly: pd.DataFrame,
    columns: tuple[str, ...] = CUMULATIVE_RECORD_COLS,
    key: tuple[str, ...] = ("race_id", "horse_no"),
) -> list[ColumnVerdict]:
    """同一 (race_id, horse_no) を当日ファイルと後日の月次ファイルで比較する。

    値が異なれば as-of-download 確定。これが最も直接的で、他の2つより優先する。
    """
    merged = daily.merge(monthly, on=list(key), suffixes=("_daily", "_monthly"))
    out: list[ColumnVerdict] = []
    for col in columns:
        a, b = f"{col}_daily", f"{col}_monthly"
        if a not in merged or b not in merged:
            out.append(ColumnVerdict(col, "LK-01", "undetermined",
                                     "当日ファイルまたは月次ファイルに該当列がありません", False))
            continue
        differ = (merged[a].astype(str) != merged[b].astype(str)).sum()
        if differ:
            out.append(ColumnVerdict(
                col, "LK-01", "as_of_download",
                f"世代間で {int(differ)}/{len(merged)} 行が変化。ファイル生成時点の値です。", False))
        else:
            out.append(ColumnVerdict(
                col, "LK-01", "undetermined",
                f"{len(merged)} 行すべて一致。LK-02 へ進んでください。", False))
    return out


def lk02_monotonicity(
    entry: pd.DataFrame, column: str = "全成績",
    horse_col: str = "horse_sk", order_col: str = "start_ts",
) -> ColumnVerdict:
    """ある馬の連続する出走をトレースし、総出走数が1レースごとに +1 か確認する。

    全行同一値なら as-of-download。単調増加なら as-of-race の候補。
    """
    if column not in entry.columns:
        return ColumnVerdict(column, "LK-02", "undetermined", "該当列がありません", False)

    work = entry[[horse_col, order_col, column]].copy()
    work = work.join(parse_record(work[column]))
    work = work.sort_values([horse_col, order_col])

    diffs = work.groupby(horse_col)["starts"].diff().dropna()
    if diffs.empty:
        return ColumnVerdict(column, "LK-02", "undetermined", "比較できる連続出走がありません", False)

    frac_plus_one = float((diffs == 1).mean())
    frac_zero = float((diffs == 0).mean())
    n_multi = int(work.groupby(horse_col)["starts"].nunique().gt(1).sum())

    if frac_zero > 0.90 and n_multi == 0:
        return ColumnVerdict(
            column, "LK-02", "as_of_download",
            f"複数出走馬の {frac_zero:.1%} で総出走数が変化しません。全行同一値です。", False)
    if frac_plus_one > 0.95:
        return ColumnVerdict(
            column, "LK-02", "as_of_race",
            f"連続出走の {frac_plus_one:.1%} で総出走数が +1。LK-03 で終端を確認してください。", False)
    return ColumnVerdict(
        column, "LK-02", "as_of_download",
        f"+1 が {frac_plus_one:.1%}、変化なしが {frac_zero:.1%} で不整合。使用不可。", False)


def lk03_terminal_match(
    entry: pd.DataFrame, column: str = "全成績",
    horse_col: str = "horse_sk", order_col: str = "start_ts",
    win_col: str = "is_win", pos_col: str = "finish_pos",
) -> ColumnVerdict:
    """各馬の最終出走行の値を、自前集計した通算成績（最終レース含む/含まない）と比較する。

    「含む」側と一致 → その行は自分の結果を知っている → 破棄。
    「含まない」側と一致 → as-of-race → 使用可。
    """
    if column not in entry.columns:
        return ColumnVerdict(column, "LK-03", "undetermined", "該当列がありません", False)

    work = entry.sort_values([horse_col, order_col]).copy()
    work = work.join(parse_record(work[column]))
    counts = work.groupby(horse_col).size().rename("total_starts")
    last = work.groupby(horse_col).tail(1).set_index(horse_col).join(counts)

    incl = (last["starts"] == last["total_starts"]).mean()
    excl = (last["starts"] == last["total_starts"] - 1).mean()

    if excl > 0.90:
        return ColumnVerdict(
            column, "LK-03", "as_of_race",
            f"最終行の {excl:.1%} が「最終レースを含まない」通算と一致。使用可。", True)
    if incl > 0.90:
        return ColumnVerdict(
            column, "LK-03", "as_of_download",
            f"最終行の {incl:.1%} が「最終レースを含む」通算と一致。自分の結果を知っています。", False)
    return ColumnVerdict(
        column, "LK-03", "undetermined",
        f"含む {incl:.1%} / 含まない {excl:.1%} のいずれとも一致しません。名寄せの品質を疑ってください。",
        False)


# ---------------------------------------------------------------------------
# 実データで判明した各列の実体（1998-01〜2026-08、483万行で確認）
#
# LK-02/LK-03 だけでは判定できない。理由は3つ。
#   (a) 「自分の結果を含む」場合も連続出走の差分は +1 になる。LK-02 は
#       as-of-download を否定できても as-of-race を肯定できない。
#   (b) 8列のうち6列は条件付きカウンタで、条件に合致した出走でしか増えない。
#       「毎出走 +1」を期待する LK-02 はこれらを一律に不整合と誤判定する。
#   (c) 中央からの転入馬は初出走行に既にキャリアを持つ。全馬の初回行を見ると
#       「0 のはずが 0 でない」行が半分近く混じり、閾値を割る。
#
# LK-09/10/11 は設計書 LK-01..03（時点性判定）の精緻化として追加したもので、
# 設計書の LK-06..08（ラベルシャッフル・シャッフル ROI・Null Importance）とは別物。
#
# 決め手は「その集計範囲での初回出走行の値」。as-of-race なら 0、自分の結果を
# 含むなら 1 になり、一意に切り分けられる（LK-09）。(c) は、初出走が2歳以下の馬
# だけに絞ってキャリアが完全に捕捉できている個体を使うことで外す。


@dataclass(frozen=True)
class ColumnSpec:
    """列ごとの集計の実体。

    scope   : 「初回」を数える単位
    applies : そのカウンタが増える行の条件（None なら全行）
    kind    : counter（回数）か best_time（記録値）か
    """

    scope: tuple[str, ...]
    applies: str | None = None
    kind: str = "counter"


COLUMN_SPEC: dict[str, ColumnSpec] = {
    "全成績": ColumnSpec(("horse_sk",)),
    "当競馬場成績": ColumnSpec(("horse_sk", "baba_code")),
    "うち当距離成績": ColumnSpec(("horse_sk", "baba_code", "distance")),
    # 左回り／右回りでしか増えない。`回り` はレース表の属性なので join して使う
    "ダート左成績": ColumnSpec(("horse_sk",), applies="turn == '左'"),
    "ダート右成績": ColumnSpec(("horse_sk",), applies="turn == '右'"),
    # 騎手単独の通算ではなく「この馬とこの騎手の組み合わせ」の成績
    "騎手成績": ColumnSpec(("horse_sk", "jockey_sk")),
    "最高タイム": ColumnSpec(("horse_sk", "baba_code", "distance"), kind="best_time"),
    "最高タイム良馬場": ColumnSpec(("horse_sk", "baba_code", "distance"),
                            applies="baba_condition == '良'", kind="best_time"),
}

SCOPE = {c: spec.scope for c, spec in COLUMN_SPEC.items()}
BEST_TIME_COLS = frozenset(c for c, spec in COLUMN_SPEC.items() if spec.kind == "best_time")

# 毎出走かならず +1 する列。LK-02（単調性）が意味を持つのはここだけ。
UNCONDITIONAL_COUNTERS = frozenset(
    c for c, spec in COLUMN_SPEC.items()
    if spec.applies is None and spec.kind == "counter" and spec.scope == ("horse_sk",)
)

# 左側打ち切り（1998年以前から走っている馬）を外す基準日
UNCENSORED_FROM = "1999-01-01"
# デビュー時点でこの年齢以下なら、中央からの転入ではなく本当の初出走とみなす
DEBUT_AGE_MAX = 2


def complete_career_horses(entry: pd.DataFrame, horse_col: str = "horse_sk",
                           order_col: str = "start_ts",
                           age_col: str = "age") -> set:
    """キャリアが丸ごとデータ内に収まっている馬。

    「初回行の値が0か」を問うには、その行が本当に初回でなければならない。
    1998年以前から走っている馬（左打ち切り）と、中央から転入した馬を外す。
    """
    first = entry.sort_values([horse_col, order_col]).groupby(horse_col).head(1)
    ok = first[order_col] >= pd.Timestamp(UNCENSORED_FROM)
    if age_col in first.columns:
        ok &= pd.to_numeric(first[age_col], errors="coerce") <= DEBUT_AGE_MAX
    return set(first.loc[ok, horse_col])


def _applicable(entry: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    if spec.applies is None:
        return entry
    try:
        return entry.query(spec.applies)
    except Exception:  # noqa: BLE001 — 条件列が無い場合は空にして undetermined へ倒す
        return entry.iloc[:0]


def lk09_first_occurrence(
    entry: pd.DataFrame, column: str,
    order_col: str = "start_ts", min_zero_frac: float = 0.95,
    horses: set | None = None,
) -> ColumnVerdict:
    """集計範囲での初回出走行を見る。as-of-race なら 0（記録列なら空）。

    自分の結果を含むなら 1 になる。この2つを分離できるのはこのテストだけ。
    """
    spec = COLUMN_SPEC.get(column)
    if column not in entry.columns or spec is None:
        return ColumnVerdict(column, "LK-09", "undetermined", "該当列がありません", False)
    if not set(spec.scope) <= set(entry.columns):
        return ColumnVerdict(column, "LK-09", "undetermined",
                             f"集計範囲 {spec.scope} の列が揃っていません", False)

    keep = complete_career_horses(entry) if horses is None else horses
    work = _applicable(entry[entry["horse_sk"].isin(keep)], spec)
    if work.empty:
        return ColumnVerdict(column, "LK-09", "undetermined",
                             f"条件 {spec.applies!r} に合致する行がありません", False)
    firsts = work.sort_values([*spec.scope, order_col]).groupby(
        list(spec.scope), sort=False).head(1)

    if spec.kind == "best_time":
        blank = firsts[column].astype(str).str.strip()
        frac, label = float(blank.isin(["", "nan", "None"]).mean()), "空"
        n_known = len(firsts)
    else:
        starts = parse_record(firsts[column])["starts"].dropna()
        if starts.empty:
            return ColumnVerdict(column, "LK-09", "undetermined", "全行が空欄です", False)
        frac, label, n_known = float((starts == 0).mean()), "0", len(starts)

    if frac >= min_zero_frac:
        return ColumnVerdict(
            column, "LK-09", "as_of_race",
            f"{spec.scope} の初回 {n_known:,} 行のうち {frac:.1%} が{label}。"
            "自分のレースを含んでいません。", True)
    if frac <= 0.05:
        return ColumnVerdict(
            column, "LK-09", "as_of_download",
            f"初回行の {frac:.1%} しか{label}でありません。自分または未来を含みます。", False)
    return ColumnVerdict(
        column, "LK-09", "undetermined",
        f"初回行が{label}なのは {frac:.1%}（n={n_known:,}）。閾値 {min_zero_frac:.0%} 未満です。",
        False)


def lk10_own_result_excluded(
    entry: pd.DataFrame, column: str = "全成績",
    order_col: str = "start_ts", win_col: str = "is_win",
    min_frac: float = 0.95,
) -> ColumnVerdict:
    """勝利数の増分がどちらのレースの結果で説明されるかを見る。

    集計範囲内で隣接する2行 i, i+1 の増分 Δ = wins[i+1] - wins[i] は
      as-of-race       なら Δ = is_win[i]      （i の結果は i+1 の行で初めて反映）
      自分の結果を含む なら Δ = is_win[i+1]    （i+1 の結果が i+1 の行に既に入る）
    となる。「次走で +1 か」だけを見ると、自分の結果を含む場合でも次走が勝ちなら
    +1 になるため分離できない。2つの仮説を直接比べる。

    範囲を跨いで隣接を取ると別のカウンタと比べることになるので、必ず scope 内で取る。
    """
    spec = COLUMN_SPEC.get(column)
    if spec is None or column not in entry.columns or win_col not in entry.columns:
        return ColumnVerdict(column, "LK-10", "undetermined", "必要な列がありません", False)
    if not set(spec.scope) <= set(entry.columns):
        return ColumnVerdict(column, "LK-10", "undetermined",
                             f"集計範囲 {spec.scope} の列が揃っていません", False)

    work = _applicable(entry, spec).sort_values([*spec.scope, order_col]).copy()
    if work.empty:
        return ColumnVerdict(column, "LK-10", "undetermined",
                             f"条件 {spec.applies!r} に合致する行がありません", False)
    work["wins"] = parse_record(work[column])["first"]
    g = work.groupby(list(spec.scope), sort=False)
    work["delta"] = g["wins"].shift(-1) - work["wins"]
    work["next_win"] = g[win_col].shift(-1)

    pairs = work[work["delta"].notna() & work["next_win"].notna()]
    # 勝敗が両方混じっていないと2仮説は区別できない
    if len(pairs) < 100 or pairs[win_col].nunique() < 2:
        return ColumnVerdict(column, "LK-10", "undetermined",
                             f"範囲内で隣接する行が {len(pairs)} 件しかありません", False)

    by_current = float((pairs["delta"] == pairs[win_col]).mean())
    by_next = float((pairs["delta"] == pairs["next_win"]).mean())

    if by_current >= min_frac and by_current > by_next:
        return ColumnVerdict(
            column, "LK-10", "as_of_race",
            f"隣接 {len(pairs):,} 組の {by_current:.1%} で増分が「前走の勝敗」に一致"
            f"（当該レースの勝敗では {by_next:.1%}）。自分の結果を含みません。", True)
    if by_next >= min_frac and by_next > by_current:
        return ColumnVerdict(
            column, "LK-10", "as_of_download",
            f"増分の {by_next:.1%} が「当該レースの勝敗」で説明されます。"
            "自分の結果を計上しています。", False)
    return ColumnVerdict(
        column, "LK-10", "undetermined",
        f"前走説明 {by_current:.1%} / 当該レース説明 {by_next:.1%} のいずれも決め手になりません。",
        False)


def lk11_best_time_reconstruction(
    entry: pd.DataFrame, column: str = "最高タイム",
    time_col: str = "time_sec", order_col: str = "start_ts",
    tol_sec: float = 0.15, min_frac: float = 0.95,
) -> ColumnVerdict:
    """記録列はカウンタではないので、自前の累積最小値と突き合わせる。

    「当該レースを含まない過去最速」と一致すれば as-of-race。
    """
    from ..transform.silver import _parse_time

    spec = COLUMN_SPEC.get(column)
    if spec is None or column not in entry.columns or time_col not in entry.columns:
        return ColumnVerdict(column, "LK-11", "undetermined", "必要な列がありません", False)
    if not set(spec.scope) <= set(entry.columns):
        return ColumnVerdict(column, "LK-11", "undetermined",
                             f"集計範囲 {spec.scope} の列が揃っていません", False)

    keep = complete_career_horses(entry)
    work = _applicable(entry[entry["horse_sk"].isin(keep)], spec)
    if work.empty:
        return ColumnVerdict(column, "LK-11", "undetermined",
                             f"条件 {spec.applies!r} に合致する行がありません", False)
    work = work.sort_values([*spec.scope, order_col]).copy()
    work["best"] = _parse_time(work[column])
    keys = list(spec.scope)
    # groupby(...).transform(lambda) は (馬, 場, 距離) の約290万グループを
    # Python ループで回すため実データでは終わらない。shift も cummin も
    # cython 実装があるので二段に分けて掛ける。
    work["_prev"] = work.groupby(keys, sort=False)[time_col].shift(1)
    prior = work.groupby(keys, sort=False)["_prev"].cummin()
    incl = work.groupby(keys, sort=False)[time_col].cummin()

    ok = work["best"].notna() & prior.notna()
    if int(ok.sum()) < 100:
        return ColumnVerdict(column, "LK-11", "undetermined",
                             f"比較できる行が {int(ok.sum())} 件しかありません", False)
    f_prior = float(np.isclose(work["best"][ok], prior[ok], atol=tol_sec).mean())
    f_incl = float(np.isclose(work["best"][ok], incl[ok], atol=tol_sec).mean())

    if f_prior >= min_frac and f_prior > f_incl:
        return ColumnVerdict(
            column, "LK-11", "as_of_race",
            f"{spec.scope} の「当該レースを含まない過去最速」と {f_prior:.1%} 一致"
            f"（含む場合は {f_incl:.1%}、n={int(ok.sum()):,}）。", True)
    if f_incl >= min_frac:
        return ColumnVerdict(
            column, "LK-11", "as_of_download",
            f"「当該レースを含む過去最速」と {f_incl:.1%} 一致。自分の走破時計を含みます。", False)
    return ColumnVerdict(
        column, "LK-11", "undetermined",
        f"含まない {f_prior:.1%} / 含む {f_incl:.1%} のいずれも閾値未満です。", False)


def run_all(entry: pd.DataFrame,
            columns: tuple[str, ...] = CUMULATIVE_RECORD_COLS) -> list[ColumnVerdict]:
    """8列すべてに、その列の実体に合ったテストを当てる。

    LK-02 は無条件カウンタにしか意味がないので、条件付き列には掛けない。
    掛けると「条件に合致しない出走で増えない」ことを不整合と誤判定する。
    """
    horses = complete_career_horses(entry)
    out: list[ColumnVerdict] = []
    for col in columns:
        out.append(lk09_first_occurrence(entry, col, horses=horses))
        if col in BEST_TIME_COLS:
            out.append(lk11_best_time_reconstruction(entry, col))
        else:
            out.append(lk10_own_result_excluded(entry, col))
        if col in UNCONDITIONAL_COUNTERS:
            out.append(lk02_monotonicity(entry, col))
    return out


def decide(verdicts: list[ColumnVerdict]) -> dict[str, object]:
    """列ごとの最終判定。

    ホワイトリストに入れる条件は3つとも満たすこと:
      1. as_of_download の判定が1つも無い
      2. 「初回出走行が0」（LK-09）が as_of_race
      3. 「自分の結果を含まない」（LK-10 か LK-11）が as_of_race
    undetermined しか無い列は破棄側に置く。「判定できなかったから使う」は逆。
    """
    by_col: dict[str, list[ColumnVerdict]] = {}
    for v in verdicts:
        by_col.setdefault(v.column, []).append(v)

    whitelist, discard, pending, reasons = [], [], [], {}
    for col, vs in by_col.items():
        tests = {v.test: v for v in vs}
        blocked = [v for v in vs if v.verdict == "as_of_download"]
        gate_first = tests.get("LK-09")
        gate_own = tests.get("LK-10") or tests.get("LK-11")
        if blocked:
            discard.append(col)
            reasons[col] = f"{blocked[0].test} が as_of_download: {blocked[0].evidence}"
        elif (gate_first is not None and gate_first.verdict == "as_of_race"
              and gate_own is not None and gate_own.verdict == "as_of_race"):
            whitelist.append(col)
            reasons[col] = f"{gate_first.evidence} / {gate_own.evidence}"
        else:
            pending.append(col)
            discard.append(col)
            missing = [n for n, g in (("LK-09", gate_first), ("LK-10/11", gate_own))
                       if g is None or g.verdict != "as_of_race"]
            reasons[col] = f"{'・'.join(missing)} が as_of_race を示しませんでした"

    return {
        "whitelist": sorted(whitelist),
        "discard": sorted(set(discard)),
        "undetermined": sorted(pending),
        "reasons": reasons,
        "conclusion": (
            "全8列を破棄し、自前の履歴から as-of 集計で再計算する"
            if not whitelist else
            f"{len(whitelist)}/{len(by_col)} 列が as-of-race。{sorted(whitelist)} を使用可とする。"
        ),
    }


def future_poisoning_diff(f1: pd.DataFrame, f2: pd.DataFrame,
                          key: tuple[str, ...] = ("race_id", "horse_no"),
                          atol: float = 0.0, rtol: float = 0.0) -> pd.DataFrame:
    """LK-05 の診断。どの列が未来レコードを参照しているかを名指しする。

    一致すべき行だけを突き合わせる（f2 は一部行が削除されているため）。

    既定は完全一致。許容差を入れると、丸め境界で1桁だけずれる列を見逃す。
    LK-05 はハッシュのビット単位一致を判定基準にしている以上、診断側も同じ
    厳しさで見ないと「ハッシュは違うのに差分列は0件」という無情報な出力になる。
    """
    k = list(key)
    common = f1.merge(f2, on=k, suffixes=("_1", "_2"), how="inner")
    rows = []
    for col in f1.columns:
        if col in k or f"{col}_1" not in common:
            continue
        a, b = common[f"{col}_1"], common[f"{col}_2"]
        if pd.api.types.is_numeric_dtype(a):
            differ = ~np.isclose(a.astype(float), b.astype(float),
                                 equal_nan=True, atol=atol, rtol=rtol)
        else:
            differ = a.astype(str) != b.astype(str)
        n = int(differ.sum())
        if n:
            gap = ""
            if pd.api.types.is_numeric_dtype(a):
                d = np.abs(a[differ].astype(float) - b[differ].astype(float))
                gap = f"最大差 {np.nanmax(d):.3e}" if len(d) else ""
            rows.append({"column": col, "n_differ": n,
                         "pct": round(100 * n / len(common), 4),
                         "max_abs_diff": gap,
                         "diagnosis": "未来レコードを参照しています"})
    return pd.DataFrame(rows).sort_values("n_differ", ascending=False) if rows else pd.DataFrame(
        columns=["column", "n_differ", "pct", "max_abs_diff", "diagnosis"])


def label_correlation_screen(feat: pd.DataFrame, label_col: str = "is_win",
                             exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """ラベルとの相関が異常に高い列を洗い出す簡易スクリーニング。

    Null Importance の前段。着順の完全関数のような列はここで即座に立つ。
    """
    y = feat[label_col].astype(float)
    rows = []
    for c in feat.select_dtypes("number").columns:
        if c == label_col or c in exclude:
            continue
        s = feat[c].astype(float)
        if s.nunique() < 2:
            continue
        r = float(np.corrcoef(s.fillna(s.median()), y)[0, 1])
        rows.append({"column": c, "abs_corr": abs(r), "corr": round(r, 4),
                     "suspect": abs(r) > 0.30})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
