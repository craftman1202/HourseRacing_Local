"""silver 層: 正規化・型付け・キー付与。

実測した列名（2026-07 の月次ファイル）に対応する。

  racelist  (66列): 競馬場 / 競走年月日 / レース番号 / 発走時刻 / 競走種類名称 / ...
  horselist (36列): 競馬場 / 競走年月日 / レース番号 / 枠番 / 馬番 / 馬名 / ...
                    ＋ 累積成績8列 ＋ 着順系5列（どちらも pre-race では落とす）
  payback   (54列): 横持ちの払戻。縦持ちへ変換する

`to_prerace()` を通すのは特徴量を作る直前であり、silver には結果列を残す。
結果は学習ラベルと as-of 集計の材料に必要で、リークは「特徴量に入れないこと」で
防ぐ（設計書 §5）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .keys import TrackMaster, add_horse_sk, make_race_id, person_keys
from .payout import unpivot

log = logging.getLogger(__name__)

# 実ファイルの列名。設計時に想定していた `トラック種別` `馬場状態` は存在せず、
# 実際は `芝ダート区分` `馬場`。名前が違うだけで rename が空振りし、surface と
# baba_condition が silver から黙って消えていた。
# `回り`（左/右）は `ダート左成績` `ダート右成績` が条件にしている属性そのものなので、
# 時点性の判定にも特徴量にも要る。
RACE_COLUMNS = {
    "競馬場": "track_name", "競走年月日": "race_ymd", "レース番号": "race_no",
    "発走時刻": "start_hhmm", "競走種類名称": "class_name", "レース名": "race_name",
    "距離": "distance", "芝ダート区分": "surface", "馬場": "baba_condition",
    "回り": "turn", "条件": "condition_name",
    "天候": "weather", "頭数": "n_runners_reported",
}
ENTRY_COLUMNS = {
    "競馬場": "track_name", "競走年月日": "race_ymd", "レース番号": "race_no",
    "枠番": "waku", "馬番": "horse_no", "馬名": "horse_name", "性": "sex",
    "齢": "age", "毛色": "coat", "生年月日": "birth_ymd", "父馬名": "sire_name",
    "母馬名": "dam_name", "母父馬名": "damsire_name", "騎手名": "jockey_name",
    "騎手所属": "jockey_affil", "負担重量": "weight_carried",
    "調教師": "trainer_name", "調教師所属": "trainer_affil",
    "馬主氏名": "owner_name", "生産牧場名": "breeder",
    "馬体重": "weight_kg", "馬体重増減": "weight_diff",
    "着順": "finish_pos", "タイム": "time_raw", "着差": "margin",
    "上がり3F": "last3f", "人気": "popularity",
}


def _digits(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def _ymd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.strip(), format="%Y%m%d", errors="coerce")


def _start_ts(ymd: pd.Series, hhmm: pd.Series) -> pd.Series:
    """発走時刻。`1430` 形式を日付に足す。

    JST の naive として返す。tz を付けるのは DB 境界の責務（運用側 db/types.py）。
    """
    t = hhmm.astype(str).str.extract(r"(\d{1,2})(\d{2})")
    hours = pd.to_numeric(t[0], errors="coerce").fillna(12).clip(0, 23)
    mins = pd.to_numeric(t[1], errors="coerce").fillna(0).clip(0, 59)
    return ymd + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(mins, unit="m")


def build_race(bronze: pd.DataFrame, master: TrackMaster | None = None) -> pd.DataFrame:
    """racelist → race テーブル。"""
    master = master or TrackMaster()
    df = bronze.rename(columns={k: v for k, v in RACE_COLUMNS.items()
                                if k in bronze.columns}).copy()
    df["baba_code"] = master.codes(df["track_name"])
    ymd = _ymd(df["race_ymd"])
    df["race_date"] = ymd.dt.date
    df["race_no"] = _digits(df["race_no"]).astype("Int64")
    df["start_ts"] = _start_ts(ymd, df.get("start_hhmm", pd.Series("1200", index=df.index)))
    df["race_id"] = [
        make_race_id(b, d, n) for b, d, n in zip(df["baba_code"], ymd, df["race_no"])
    ]
    df["distance"] = _digits(df.get("distance", pd.Series(np.nan, index=df.index)))
    df["class_level"] = _class_level(df.get("class_name", pd.Series("", index=df.index)))
    df["prize_yen"] = _first_prize(bronze)

    keep = ["race_id", "race_date", "start_ts", "baba_code", "track_name", "race_no",
            "distance", "surface", "turn", "class_name", "class_level", "prize_yen",
            "weather", "baba_condition", "condition_name", "race_name"]
    return df[[c for c in keep if c in df.columns]].drop_duplicates("race_id")


def _class_level(names: pd.Series) -> pd.Series:
    """クラス名から序数を作る。細かい格付けは場ごとに違うので粗く揃える。"""
    s = names.astype(str)
    out = pd.Series(3, index=s.index, dtype=int)
    out[s.str.contains("新馬|未勝利", na=False)] = 1
    out[s.str.contains("普通|一般", na=False)] = 2
    out[s.str.contains("特別", na=False)] = 4
    out[s.str.contains("重賞|Ｇ|G[123]|JpnI", na=False, regex=True)] = 5
    return out


def _first_prize(bronze: pd.DataFrame) -> pd.Series:
    for col in bronze.columns:
        if "本賞金" in col or ("賞金" in col and "1" in col):
            v = _digits(bronze[col])
            if v.notna().any():
                return v.fillna(v.median())
    return pd.Series(1_000_000.0, index=bronze.index)


def build_entry(bronze: pd.DataFrame, master: TrackMaster | None = None) -> pd.DataFrame:
    """horselist → entry テーブル（結果列を含む silver）。"""
    master = master or TrackMaster()
    df = bronze.rename(columns={k: v for k, v in ENTRY_COLUMNS.items()
                                if k in bronze.columns}).copy()
    df["baba_code"] = master.codes(df["track_name"])
    ymd = _ymd(df["race_ymd"])
    df["race_date"] = ymd.dt.date
    df["race_no"] = _digits(df["race_no"]).astype("Int64")
    df["race_id"] = [
        make_race_id(b, d, n) for b, d, n in zip(df["baba_code"], ymd, df["race_no"])
    ]
    df["horse_no"] = _digits(df["horse_no"]).astype("Int64")
    df["waku"] = _digits(df.get("waku", pd.Series(np.nan, index=df.index)))
    df["weight_kg"] = _digits(df.get("weight_kg", pd.Series(np.nan, index=df.index)))
    df["age"] = _digits(df.get("age", pd.Series(np.nan, index=df.index)))

    # 着順は数値化する。取消・除外は数値にならないので NaN になり、
    # 「結果が無い行」として扱われる（頭数カウントからも自然に外れる）
    df["finish_pos"] = _digits(df.get("finish_pos", pd.Series(np.nan, index=df.index)))
    df["is_win"] = (df["finish_pos"] == 1).astype(int)
    df["time_sec"] = _parse_time(df.get("time_raw", pd.Series("", index=df.index)))
    df["popularity"] = _digits(df.get("popularity", pd.Series(np.nan, index=df.index)))

    df["birth_ymd"] = df.get("birth_ymd", pd.Series("", index=df.index)).astype(str)
    df = add_horse_sk(df, name="horse_name", birth="birth_ymd", sire="sire_name")
    df = person_keys(df, "jockey_name", "jockey_affil", "jockey")
    df = person_keys(df, "trainer_name", "trainer_affil", "trainer")
    df["sire_sk"] = df.get("sire_name", pd.Series("", index=df.index)).astype(str).str.strip()
    df["dam_sk"] = df.get("dam_name", pd.Series("", index=df.index)).astype(str).str.strip()

    # 発走時刻は race 側から結合する（horselist には無い）
    keep = [
        "race_id", "race_date", "baba_code", "race_no", "horse_no", "waku",
        "horse_sk", "horse_name", "birth_ymd", "sex", "age",
        "sire_sk", "dam_sk", "damsire_name",
        "jockey_sk", "jockey_affil_sk", "jockey_name",
        "trainer_sk", "trainer_affil_sk", "trainer_name",
        "weight_carried", "weight_kg", "weight_diff",
        "finish_pos", "is_win", "time_sec", "popularity", "margin", "last3f",
        # 時点性が未判定の累積成績8列。silver には残し、pre-race で落とす
        "騎手成績", "全成績", "ダート左成績", "ダート右成績",
        "当競馬場成績", "うち当距離成績", "最高タイム", "最高タイム良馬場",
    ]
    out = df[[c for c in keep if c in df.columns]].copy()
    return out.dropna(subset=["horse_no"]).reset_index(drop=True)


GOING_PREFIX_RE = r"^(?:良|稍重|重|不良)"


def _parse_time(s: pd.Series) -> pd.Series:
    """走破タイムを秒に直す。

    実ファイルの形式は区切り無しの `MSSF`（右から 1桁=1/10秒、2桁=秒、残り=分）。
    `1143` → 1分14.3秒 = 74.3 秒、`591` → 59.1 秒、`2045` → 124.5 秒。

    これを素の数値として読むと 1143 秒になり、速度指数が桁ごと壊れる。
    `1:14.3` 形式も来うるので両方受ける。

    `最高タイム良馬場` は馬場状態ラベルを前置した `良1:33.2` 形式で入っている。
    ラベルを剥がさないと全行 NaN になる（実際 310 万行が黙って落ちていた）。
    剥がすのは既知の馬場状態ラベルだけにする。何でも剥がすと壊れた値を
    通してしまう。
    """
    txt = s.astype(str).str.strip().str.replace(GOING_PREFIX_RE, "", regex=True)

    # `M:SS.f` 形式
    mmss = txt.str.extract(r"^(\d+):(\d{1,2})\.?(\d*)$")
    from_colon = (pd.to_numeric(mmss[0], errors="coerce") * 60
                  + pd.to_numeric(mmss[1], errors="coerce")
                  + pd.to_numeric(mmss[2].replace("", "0"), errors="coerce").fillna(0) / 10)

    # 区切り無し `MSSF`
    digits = txt.str.fullmatch(r"\d{3,5}")
    packed = txt.where(digits)
    tenths = pd.to_numeric(packed.str[-1], errors="coerce")
    seconds = pd.to_numeric(packed.str[-3:-1], errors="coerce")
    minutes = pd.to_numeric(packed.str[:-3].replace("", "0"), errors="coerce").fillna(0)
    from_packed = minutes * 60 + seconds + tenths / 10

    return from_colon.fillna(from_packed)


def build_payout(bronze: pd.DataFrame, master: TrackMaster | None = None) -> pd.DataFrame:
    """payback → 縦持ち払戻。"""
    master = master or TrackMaster()
    df = bronze.copy()
    ymd = _ymd(df["競走年月日"])
    baba = master.codes(df["競馬場"])
    race_no = _digits(df["レース番号"]).astype("Int64")
    df["race_id"] = [make_race_id(b, d, n) for b, d, n in zip(baba, ymd, race_no)]
    df["race_date"] = ymd.dt.date

    long = _unpivot_payback(df)
    return long


# 実ファイル（2026-07）の払戻列レイアウト。
#   (券種, 同着枠の通番, 組番列のテンプレート, 払戻列, 人気列)
# 通番と「脚」の区別が要点: `複勝組番1..3` は **3頭ぶんの通番**、
# `枠複組番1,2` は **1つの買い目の2脚**。払戻列に通番が付くかどうかで見分ける。
_PAYBACK_LAYOUT: tuple[tuple[str, tuple[str, ...], tuple[str, ...], str, str], ...] = (
    ("単勝",   ("",),        ("単勝組番",),                         "単勝払戻金（円）",   "単勝人気"),
    ("複勝",   ("1", "2", "3"), ("複勝組番{i}",),                   "複勝払戻金{i}（円）", "複勝人気{i}"),
    ("枠連",   ("",),        ("枠複組番1", "枠複組番2"),             "枠複払戻金（円）",   "枠複人気"),
    ("枠単",   ("",),        ("枠単組番1", "枠単組番2"),             "枠単払戻金（円）",   "枠単人気"),
    ("馬連",   ("",),        ("馬複組番1", "馬複組番2"),             "馬複払戻金（円）",   "馬複人気1"),
    ("馬単",   ("",),        ("馬単組番1", "馬単組番2"),             "馬単払戻金（円）",   "馬単人気1"),
    ("ワイド", ("1", "2", "3"), ("ワイド組番{i}馬番1", "ワイド組番{i}馬番2"),
                                                                    "ワイド払戻金{i}（円）", "ワイド人気{i}"),
    ("3連複",  ("",),        ("３連複組番馬番1", "３連複組番馬番2", "３連複組番馬番3"),
                                                                    "３連複払戻金（円）", "３連複人気"),
    ("3連単",  ("",),        ("３連単組番馬番1", "３連単組番馬番2", "３連単組番馬番3"),
                                                                    "３連単払戻金（円）", "３連単人気"),
)


def _unpivot_payback(df: pd.DataFrame) -> pd.DataFrame:
    """実ファイルの横持ち払戻を縦持ちにする。

    組番が `馬番1/馬番2/馬番3` のように**複数列に分かれている**のが実構造で、
    1列の `9-10` 形式ではない。正規表現で当てにいくと券種を取りこぼすので、
    レイアウトを明示的に持つ。列名が変わればスキーマガードが先に気付く。
    """
    cols = set(map(str, df.columns))
    records: list[dict] = []

    for _, row in df.iterrows():
        for bet, indices, comb_tpl, pay_tpl, pop_tpl in _PAYBACK_LAYOUT:
            seq = 0
            for i in indices:
                pay_col = pay_tpl.format(i=i)
                if pay_col not in cols:
                    continue
                yen = row.get(pay_col)
                if _blank(yen):
                    continue
                legs = []
                for tpl in comb_tpl:
                    c = tpl.format(i=i)
                    if c in cols and not _blank(row.get(c)):
                        legs.append(_int_or_none(row.get(c)))
                if not legs:
                    continue
                legs += [None] * (3 - len(legs))
                seq += 1
                records.append({
                    "race_id": row["race_id"], "race_date": row["race_date"],
                    "bet_type": bet,
                    "comb_1": legs[0], "comb_2": legs[1], "comb_3": legs[2],
                    "payout_yen": int(float(str(yen).replace(",", ""))),
                    "popularity": _int_or_none(row.get(pop_tpl.format(i=i))),
                    "dead_heat_seq": seq,
                })
    return pd.DataFrame(records)


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() in {"", "nan"}


def _split_comb(s: str):
    import re

    parts = [int(p) for p in re.split(r"[-–—>=]", s.strip()) if p.strip().isdigit()]
    parts += [None] * (3 - len(parts))
    return parts[0], parts[1], parts[2]


def _int_or_none(v):
    return None if _blank(v) else int(float(str(v).replace(",", "")))


ODDS_COLUMNS = {
    "競馬場": "track_name", "競走年月日": "race_ymd", "レース番号": "race_no",
    "賭式": "bet_type", "番号1": "comb_1", "番号2": "comb_2", "番号3": "comb_3",
    "オッズ": "odds", "オッズ（最大）": "odds_max", "人気": "popularity",
}


def build_odds(bronze: pd.DataFrame, master: TrackMaster | None = None,
               bet_types: tuple[str, ...] = ("単勝",)) -> pd.DataFrame:
    """odds CSV → 縦持ちオッズ。

    ファイルは全券種を含み1か月80MB近くになる。トラックB が使うのは単勝が主なので、
    既定では単勝だけに絞る。全券種が要るときは bet_types を広げる。

    **これは確定オッズであり締切前オッズではない。** 賭け判断に確定オッズを使うと
    RF-04（時点の取り違え）に該当するので、学習の市場ベースライン用途に限る。
    """
    master = master or TrackMaster()
    df = bronze.rename(columns={k: v for k, v in ODDS_COLUMNS.items()
                                if k in bronze.columns}).copy()
    if bet_types:
        df = df[df["bet_type"].isin(bet_types)]
    if df.empty:
        return pd.DataFrame(columns=["race_id", "horse_no", "odds_win", "popularity"])

    df["baba_code"] = master.codes(df["track_name"])
    ymd = _ymd(df["race_ymd"])
    df["race_date"] = ymd.dt.date
    race_no = _digits(df["race_no"]).astype("Int64")
    df["race_id"] = [make_race_id(b, d, n)
                     for b, d, n in zip(df["baba_code"], ymd, race_no)]
    df["horse_no"] = _digits(df["comb_1"]).astype("Int64")
    df["odds_win"] = _digits(df["odds"])
    df["popularity"] = _digits(df.get("popularity", pd.Series(np.nan, index=df.index)))

    out = df[["race_id", "race_date", "horse_no", "odds_win", "popularity"]]
    # 0 以下のオッズは存在しない。ゼロ埋めの痕跡として落とす（IN-05 と同じ思想）
    out = out[(out["odds_win"] > 0) & out["horse_no"].notna()]
    return out.drop_duplicates(["race_id", "horse_no"]).reset_index(drop=True)


def attach_start_ts(entry: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """entry に発走時刻と距離を結合する。as-of 集計の順序付けに必須。"""
    cols = ["race_id", "start_ts", "distance", "class_level", "prize_yen"]
    return entry.merge(race[[c for c in cols if c in race.columns]], on="race_id",
                       how="left")
