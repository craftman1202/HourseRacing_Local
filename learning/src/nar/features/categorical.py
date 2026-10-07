"""カテゴリ変数（と、文字列のまま使われていなかった数値）の特徴量（2026-10-07）。

これまで平地の特徴量はレース条件を `class_level`（競走種類名称の5値 → 2/4/5 の3段階）
でしか持っておらず、次の情報は gold に1列も入っていなかった:

  レース名      クラス（A/B/C/D とその組）、新馬・未勝利、オープン、JRA 交流、牝馬限定
  競走条件      馬齢限定（2歳/3歳/以上）、アラブ系、重量の決め方（ハンデ/別定/定量/馬齢/規定）
  天候・馬場    馬場状態（良〜不良）、雨・雪、芝
  馬            性（牡/牝/セン — 「セン」と「去」が混在）、年齢、馬体重と増減
  騎手・調教師  負担重量（☆▲△ は見習の減量、◇★ は女性騎手）、所属（JRA・他の開催圏からの遠征）

年齢は 2000 年までが数え年（今の満年齢 + 1）で、生データの `age` は定義が途中で変わる。
ここでは「開催年 − 生年」で満年齢に揃え、2000 年以前のレース名・競走条件の
「n歳 / n才」も 1 引いて今の数え方に直す。

エンコードは序数・2値だけ（全モデル共通で扱える）。ターゲット・エンコーディングは
使わない（as-of を崩しやすいので）。**レース単位で一定の列は、レース内 softmax の
モデル（条件付きロジット・TabM の listwise 損失）では効果が相殺される** — 意味を持つのは
他の特徴量との交互作用（GBDT / TabM）を通してだけ。馬単位の列（性・年齢・負担重量・
所属など）は直接効く。

運用時の入手元: レース名・競走条件・重量の決め方は当日出馬表の見出し、馬・騎手の列は
出馬表の各行（narops.deba_table）。天候・馬場は見出しに出る時刻が遅いことがあり、
推論時は欠測になりうる（標準化器が学習時の中央値で補完する）。
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = (
    # 馬単位
    "c_is_female", "c_is_gelding", "c_age", "c_age_rel",
    "c_weight_carried", "c_weight_carried_rel", "c_jockey_apprentice", "c_jockey_female",
    "c_body_weight", "c_body_weight_diff", "c_body_weight_rel",
    "c_jockey_jra", "c_jockey_visiting", "c_trainer_jra", "c_trainer_visiting",
    # レース単位（交互作用でだけ効く）
    "c_tier", "c_tier_sub", "c_is_newcomer", "c_is_maiden", "c_is_open", "c_is_jra_exchange",
    "c_is_female_only", "c_age_2yo", "c_age_3yo", "c_age_upper", "c_is_arab",
    "c_wr_handicap", "c_wr_betteitei", "c_wr_teiryo", "c_wr_barei", "c_wr_kitei",
    "c_going", "c_rain", "c_snow", "c_turf",
)

# 所属（騎手・調教師）と競馬場を「開催の圏（サーキット）」にまとめる。南関東 4 場・東海
# （名古屋・笠松）・北関東などは同じ圏内で騎手が日常的に乗り回すので、それを「遠征」とは
# 数えない（場単位で数えると 1999 年の平地で 43% が遠征扱いになった）。
_CIRCUIT = {
    # 所属名
    "大井": "南関東", "船橋": "南関東", "川崎": "南関東", "浦和": "南関東",
    "愛知": "東海", "笠松": "東海", "兵庫": "兵庫", "岩手": "岩手", "北海道": "北海道",
    "栃木": "北関東", "宇都宮": "北関東", "高崎": "北関東", "新潟": "新潟", "上山": "上山",
    "金沢": "金沢", "高知": "高知", "佐賀": "九州", "荒尾": "九州", "中津": "九州",
    "福山": "中国", "益田": "中国",
    # 競馬場名（所属名と違うもの）
    "名古屋": "東海", "中京": "東海", "園田": "兵庫", "姫路": "兵庫", "盛岡": "岩手",
    "水沢": "岩手", "門別": "北海道", "札幌": "北海道", "旭川": "北海道", "道営": "北海道",
    "足利": "北関東", "三条": "新潟",
}

_GOING = {"良": 0.0, "稍重": 1.0, "重": 2.0, "不良": 3.0}
_APPRENTICE = set("☆▲△")
_FEMALE_JOCKEY = set("◇★")
_TIER = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
_KANJI_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7",
              "八": "8", "九": "9"}
_TIER_RE = re.compile(r"(?<![A-Z])([ABCD])\s*([0-9]?)")


def _nfkc(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(lambda x: unicodedata.normalize("NFKC", x))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"([-+]?\d+(?:\.\d+)?)")[0], errors="coerce")


def _affil(sk: pd.Series) -> pd.Series:
    """`的場文@大井` → `大井`。運用側の person_keys も同じ形で作る。"""
    return sk.fillna("").astype(str).str.split("@").str[-1].str.strip()


def _visiting(affil: pd.Series, track: pd.Series) -> pd.Series:
    """所属の圏と開催場の圏が違えば 1。JRA・外国・無所属・不明は 0（JRA は別の列で持つ）。"""
    def one(a, t):
        if not a or a in ("JRA", "外国", "無所属"):
            return 0.0
        ca, ct = _CIRCUIT.get(a, a), _CIRCUIT.get(t, t)
        return 0.0 if ca == ct else 1.0
    return pd.Series([one(a, t) for a, t in zip(affil, track)], index=affil.index, dtype="float64")


def _age_class(text: pd.Series, before_2001: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """競走条件・レース名の馬齢限定。2000 年以前は数え年なので 1 引く。"""
    m = text.str.extract(r"(\d)\s*[歳才]\s*(以上)?")
    age = pd.to_numeric(m[0], errors="coerce") - before_2001.astype(float)
    upper = m[1].notna() & age.notna()
    is2 = (age == 2) & ~upper
    is3 = (age == 3) & ~upper
    return is2.astype("float64"), is3.astype("float64"), upper.astype("float64")


def race_level(race: pd.DataFrame) -> pd.DataFrame:
    """レース単位の列（race_id ごとに1行）。"""
    r = race.drop_duplicates("race_id").set_index("race_id")
    name = _nfkc(r.get("race_name", pd.Series("", index=r.index)))
    cond = _nfkc(r.get("condition_name", pd.Series("", index=r.index)))
    year = pd.to_datetime(r["race_date"]).dt.year
    before = year < 2001
    out = pd.DataFrame(index=r.index)

    m = name.str.replace(r"[一二三四五六七八九]", lambda x: _KANJI_NUM[x.group(0)], regex=True) \
        .str.extract(_TIER_RE)
    out["c_tier"] = m[0].map(_TIER)
    out["c_tier_sub"] = pd.to_numeric(m[1].where(m[1] != ""), errors="coerce")
    out["c_is_newcomer"] = name.str.contains("新馬").astype("float64")
    out["c_is_maiden"] = name.str.contains("未勝利|未出走").astype("float64")
    out["c_is_open"] = name.str.contains("オープン|OP").astype("float64")
    out["c_is_jra_exchange"] = name.str.contains("交流|JRA").astype("float64")
    out["c_is_female_only"] = (name.str.contains("牝") | cond.str.contains("牝")).astype("float64")
    a2, a3, up = _age_class(cond, before)
    # 競走条件に馬齢が無いとき（「一般」）はレース名から補う（例: 「3歳以上 C4-3」）
    n2, n3, nup = _age_class(name, before)
    no_cond_age = ~cond.str.contains(r"\d\s*[歳才]")
    out["c_age_2yo"] = np.where(no_cond_age, n2, a2)
    out["c_age_3yo"] = np.where(no_cond_age, n3, a3)
    out["c_age_upper"] = np.where(no_cond_age, nup, up)
    out["c_is_arab"] = cond.str.contains("アラブ").astype("float64")
    for col, kw in (("c_wr_handicap", "ハンデ"), ("c_wr_betteitei", "別定"),
                    ("c_wr_teiryo", "定量"), ("c_wr_barei", "馬齢"), ("c_wr_kitei", "規定")):
        out[col] = cond.str.contains(kw).astype("float64")
    going = r.get("baba_condition", pd.Series("", index=r.index)).fillna("").astype(str).str.strip()
    out["c_going"] = going.map(_GOING)            # ばんえいの含水率（数値）や空欄は NaN
    weather = r.get("weather", pd.Series("", index=r.index)).fillna("").astype(str)
    has_weather = weather.str.strip() != ""
    out["c_rain"] = weather.str.contains("雨").astype("float64").where(has_weather)
    out["c_snow"] = weather.str.contains("雪").astype("float64").where(has_weather)
    surface = r.get("surface", pd.Series("", index=r.index)).fillna("").astype(str)
    out["c_turf"] = surface.str.startswith("芝").astype("float64")
    return out


def build(entry: pd.DataFrame, race: pd.DataFrame, track_names: dict[int, str]) -> pd.DataFrame:
    """entry と同じ行順で CATEGORICAL_FEATURES を返す（race_id, horse_no 付き）。"""
    e = entry.reset_index(drop=True)
    out = pd.DataFrame({"race_id": e["race_id"].to_numpy(), "horse_no": e["horse_no"].to_numpy()})
    rid = e["race_id"]

    sex = e.get("sex", pd.Series("", index=e.index)).fillna("").astype(str).str.strip()
    out["c_is_female"] = (sex == "牝").astype("float64").to_numpy()
    out["c_is_gelding"] = sex.isin(["セン", "せん", "去", "騸"]).astype("float64").to_numpy()

    year = pd.to_datetime(e["race_date"]).dt.year
    birth = pd.to_numeric(e.get("birth_ymd", pd.Series("", index=e.index)).astype(str).str[:4],
                          errors="coerce")
    age = (year - birth).where(birth > 1900)
    out["c_age"] = age.to_numpy(dtype="float64")
    out["c_age_rel"] = (age - age.groupby(rid).transform("mean")).to_numpy(dtype="float64")

    wc_raw = e.get("weight_carried", pd.Series("", index=e.index)).fillna("").astype(str).str.strip()
    wc = _num(wc_raw)
    mark = wc_raw.str[:1]
    out["c_weight_carried"] = wc.to_numpy(dtype="float64")
    out["c_weight_carried_rel"] = (wc - wc.groupby(rid).transform("mean")).to_numpy(dtype="float64")
    out["c_jockey_apprentice"] = mark.isin(_APPRENTICE).astype("float64").to_numpy()
    out["c_jockey_female"] = mark.isin(_FEMALE_JOCKEY).astype("float64").to_numpy()

    body = pd.to_numeric(e.get("weight_kg", pd.Series(np.nan, index=e.index)), errors="coerce")
    body = body.where(body.between(250, 700))      # 平地の馬体重。外れは解析ミスとして欠測
    out["c_body_weight"] = body.to_numpy(dtype="float64")
    out["c_body_weight_diff"] = _num(e.get("weight_diff", pd.Series("", index=e.index))
                                     .fillna("")).to_numpy(dtype="float64")
    out["c_body_weight_rel"] = (body - body.groupby(rid).transform("mean")).to_numpy(dtype="float64")

    # 競馬場名はレース表の track_name（廃止場を含む 31 場）を優先する。TrackMaster は
    # 現存場しか知らず、廃止場のレースが全頭「遠征」扱いになっていた（1999 年で 39%）。
    by_race = (race.drop_duplicates("race_id").set_index("race_id")["track_name"]
               if "track_name" in race.columns else pd.Series(dtype=object))
    track = e["race_id"].map(by_race)
    track = track.where(track.notna() & (track != ""), e["baba_code"].map(track_names)).fillna("")
    for who in ("jockey", "trainer"):
        col = f"{who}_affil_sk"
        aff = _affil(e[col]) if col in e.columns else pd.Series("", index=e.index)
        out[f"c_{who}_jra"] = (aff == "JRA").astype("float64").to_numpy()
        out[f"c_{who}_visiting"] = _visiting(aff, track).to_numpy()

    rl = race_level(race)
    rows = rl.reindex(e["race_id"]).reset_index(drop=True)
    for c in rl.columns:
        out[c] = rows[c].to_numpy(dtype="float64")
    return out[["race_id", "horse_no", *CATEGORICAL_FEATURES]]
