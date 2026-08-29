"""NAR が出馬表に載せている累積成績8列から特徴量を作る。

この8列は EDA（LK-06/07/08）で as-of-race であることを確定させた
（artifacts/leak_verdict.json、4,831,906 行）。当該レースの結果を含まない。

自前のウィンドウ集計と重複するように見えるが、価値は重複しない部分にある。
月次ファイルは1998年からしか無く、中央（JRA）のレースも入っていない。
一方この8列は馬の実際の通算成績なので、1998年以前の出走も中央での出走も
含んでいる。`d_extra_starts`（宣言値 − 自前集計）がその差そのもので、
自前の履歴では原理的に作れない情報になる。

運用時も当日の出馬表（DebaTable）に同じ8列が載るため、学習と推論で同じ値を
取れる（train-serving skew にならない）。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

RECORD_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$")

# 列 → 特徴量の接頭辞。d は declared（NAR 申告値）の意味
RECORD_SOURCES = {
    "全成績": "d_all",
    "当競馬場成績": "d_track",
    "うち当距離成績": "d_dist",
    "騎手成績": "d_pair",       # この馬とこの騎手の組み合わせ
}
# 回りは当該レースの回りに合う方を選ぶ。左回りのレースで右回り成績を見ても意味がない
TURN_SOURCES = {"左": "ダート左成績", "右": "ダート右成績"}

DECLARED_FEATURES = (
    "d_all_starts", "d_all_winrate", "d_all_top3rate",
    "d_track_starts", "d_track_winrate",
    "d_dist_starts", "d_dist_winrate",
    "d_pair_starts", "d_pair_winrate",
    "d_turn_starts", "d_turn_winrate",
    "d_extra_starts", "d_has_outside_history",
    "d_best_speed", "d_best_speed_good", "d_best_speed_penalty",
)


def parse_record(s: pd.Series) -> pd.DataFrame:
    """`9-22-14-85` を 1着/2着/3着/着外 に分け、総出走数を出す。

    空欄は 0 ではなく NaN。ばんえいなど当該集計が存在しない行で空欄が出るので、
    0 で埋めると「0戦0勝の馬」と区別がつかなくなる。
    """
    parsed = s.astype(str).str.extract(RECORD_RE)
    parsed.columns = ["first", "second", "third", "others"]
    out = parsed.apply(pd.to_numeric, errors="coerce")
    out["starts"] = out.sum(axis=1, skipna=False)
    return out


def _shrunk_rate(hits: pd.Series, starts: pd.Series, prior: pd.Series,
                 alpha: float) -> pd.Series:
    """経験ベイズ収縮。出走数の少ない馬の勝率が 0/1 に振れるのを抑える。"""
    return (hits + alpha * prior) / (starts + alpha)


def build(entry: pd.DataFrame, race: pd.DataFrame, *,
          alpha: float = 5.0, own_starts: pd.Series | None = None,
          prior_win: pd.Series | None = None) -> pd.DataFrame:
    """entry（8列を含む）と race から申告値ベースの特徴量を作る。

    own_starts に自前集計の出走数を渡すと `d_extra_starts` が出せる。
    prior_win は収縮先。渡さない場合は出走頭数の逆数相当を使う。
    """
    from ..transform.silver import _parse_time

    attrs = [c for c in ("turn", "distance") if c in race.columns]
    df = entry.merge(race[["race_id", *attrs]], on="race_id", how="left",
                     suffixes=("", "_race"))
    dist = pd.to_numeric(df.get("distance_race", df.get("distance")), errors="coerce")

    if prior_win is None:
        prior_win = pd.Series(0.1, index=df.index)
    prior_win = pd.Series(np.asarray(prior_win, dtype=float), index=df.index)

    out = pd.DataFrame({"race_id": df["race_id"], "horse_no": df["horse_no"]})

    for col, prefix in RECORD_SOURCES.items():
        rec = parse_record(df[col]) if col in df.columns else None
        if rec is None:
            out[f"{prefix}_starts"] = np.nan
            out[f"{prefix}_winrate"] = np.nan
            continue
        out[f"{prefix}_starts"] = rec["starts"]
        out[f"{prefix}_winrate"] = _shrunk_rate(rec["first"], rec["starts"],
                                                prior_win, alpha)
        if prefix == "d_all":
            top3 = rec[["first", "second", "third"]].sum(axis=1, skipna=False)
            out["d_all_top3rate"] = _shrunk_rate(top3, rec["starts"], prior_win, alpha * 3)

    # 回り別成績は当該レースの回りに合う列を選ぶ
    turn = df.get("turn", pd.Series("", index=df.index)).astype(str).str.strip()
    turn_starts = pd.Series(np.nan, index=df.index)
    turn_wins = pd.Series(np.nan, index=df.index)
    for label, col in TURN_SOURCES.items():
        if col not in df.columns:
            continue
        rec = parse_record(df[col])
        hit = turn == label
        turn_starts = turn_starts.mask(hit, rec["starts"])
        turn_wins = turn_wins.mask(hit, rec["first"])
    out["d_turn_starts"] = turn_starts
    out["d_turn_winrate"] = _shrunk_rate(turn_wins, turn_starts, prior_win, alpha)

    # 自前の履歴では原理的に作れない部分（1998年以前・中央での出走）
    if own_starts is not None:
        own = pd.Series(np.asarray(own_starts, dtype=float), index=df.index)
        extra = out["d_all_starts"] - own
        # 名寄せの取りこぼしで負に振れることがある。負は情報ではないので 0 に寄せる
        out["d_extra_starts"] = extra.clip(lower=0)
        out["d_has_outside_history"] = (extra > 0).astype(float)
    else:
        out["d_extra_starts"] = np.nan
        out["d_has_outside_history"] = np.nan

    # 自己ベストは秒のままでは距離をまたいで比較できないので速度に直す
    best = _parse_time(df["最高タイム"]) if "最高タイム" in df.columns else pd.Series(np.nan, index=df.index)
    best_good = (_parse_time(df["最高タイム良馬場"]) if "最高タイム良馬場" in df.columns
                 else pd.Series(np.nan, index=df.index))
    out["d_best_speed"] = dist / best.replace(0, np.nan)
    out["d_best_speed_good"] = dist / best_good.replace(0, np.nan)
    # 良馬場に比べてどれだけ落ちるか。道悪適性の代理になる
    out["d_best_speed_penalty"] = out["d_best_speed"] - out["d_best_speed_good"]

    # dtype を float64 に固定する。出走数は整数だが、成績列が空欄の行では
    # NaN になるため、データ次第で int64 と float64 が入れ替わる。学習時に
    # int64、推論時に float64 になると feature_spec が一致せず推論が止まる。
    for c in DECLARED_FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out
