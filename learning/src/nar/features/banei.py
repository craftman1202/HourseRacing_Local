"""ばんえい競馬に固有の発走前特徴量。

平地とは勝敗を決める要因が違う。ばんえいは 200m 直線・2つの障害を、
そりに積んだ**重量**を曳いて越える競技で、距離も回りも存在しない。
平地モデルの 20 特徴量のうち以下はばんえいでは情報を持たない（実データで確認）:

  distance            全レース 200m 固定。分散ゼロで標準化が壊れる
  d_turn_*            「ダート左/右成績」が 100% 空欄（直線しかない）
  d_best_speed*       「最高タイム」が 93.9% 空欄
  d_dist_*            「うち当距離成績」が「当競馬場成績」と 99.999% 一致

代わりに効くのが重量まわりである。ばんえいの重量はクラス・性別・収得賞金で
決まる純然たるハンデで、同一レース内の**重量差**が競走能力の市場評価そのもの
になっている。馬体重（実測 664-1266kg）に対する負担の比も、平地の斤量比とは
桁も意味も違う。

含水率（`baba_condition`）は平地の 良/稍重/重/不良 と違い、ばんえいでは
数値（%）で発表される。走破タイムが倍近く変わる支配的な変数なので落とせない。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

BANEI_FEATURES = (
    "b_weight_carried", "b_weight_rel", "b_is_apprentice",
    "b_body_weight", "b_load_ratio", "b_body_weight_diff",
    "b_moisture", "b_age", "b_is_female", "b_is_gelding",
)

# 負担重量は「☆760」のように減量騎手の印が前置される（実データで 5 万行以上）。
# 数値だけを取り、印そのものも特徴量として残す。
_WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)")
_APPRENTICE_MARK = "☆▲△◇★"
# 去勢馬の表記は「セン」と「去」の両方が実データに存在する
_GELDING = ("セン", "去")

# 含水率の上限。砂は飽和しても重量比 20% 程度で、これを超える値は測定値ではない。
# 実データでは 2004 年だけ 60-69 が 766 行あり（他の年は 0-10 に収まる）、
# 単位か記録対象が違ったとしか考えられない。標準化の尺度がこの 0.16% で
# 5 割ふくらむので、値を捨てて欠損として扱う。0 で埋めない — 「乾いた馬場」に
# 化けさせると意味が反転する。
MOISTURE_MAX = 20.0

# 推論時に「欠測のまま通してはいけない」列。
#
# 標準化器（narops.runtime.Standardizer）は欠測を学習時の中央値で埋める。
# 平地ではそれで良い（欠ける列は履歴の薄い馬の成績で、中央値は妥当な事前）。
# ばんえいでは違う。重量はこの競技のハンデそのもので、埋めた瞬間に
# 「全馬が平均的な重量を曳く」という事実と異なる前提の予測になる。
# しかも `b_weight_rel` はレース内の相対量なので、一部の馬だけ中央値で
# 埋まると馬同士の優劣が直接歪む。
#
# 実測（2026-08-30）: 開催日の**早朝**の出馬表には馬体重が1頭も載らず、
# 負担重量も 10 頭中 8 頭しか埋まっていない。発走13分前（開催中）の
# ページには全頭ぶん載る。つまり「普段は取れるが、早い時刻や公表遅れでは
# 取れない」列であり、黙って埋めると静かに劣化した予測が配信まで流れる。
REQUIRED_AT_SERVING = ("b_weight_carried", "b_body_weight")

# 推論時の妥当域。欠測ではなく「値はあるが明らかにおかしい」を捕まえる。
#
# 当日ページの馬体重は `(\d{3})` で読まれており、4桁（1000kg 超）の馬が
# 下3桁だけ拾われていた。1022kg が 22kg になっても欠測ではないので
# REQUIRED_AT_SERVING では捕まらず、「22kg の馬が 610kg を曳く」という
# 予測が自信を持って出る（実測 2026-08-29 帯広 R1 で 9 頭中 3 頭）。
#
# 値域は実データの実測（馬体重 664-1266kg / 負担重量 450-1000kg）に
# 余裕を持たせたもの。EDA の quality.BANEI_RULES と同じ考え方。
SERVING_RANGES = {
    "b_body_weight": (500.0, 1500.0),
    "b_weight_carried": (300.0, 1200.0),
}


def _num(s: pd.Series) -> pd.Series:
    """先頭の記号を落として数値化する。数値にならない値は NaN。"""
    txt = s.astype(str).str.strip()
    return pd.to_numeric(txt.str.extract(_WEIGHT_RE, expand=False), errors="coerce")


def build(entry: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """entry（発走前列のみ）と race からばんえい特徴量を作る。

    返すのは (race_id, horse_no) + BANEI_FEATURES。全行 NaN になる列も落とさない
    — 列集合がデータ依存になると学習時と推論時で feature_spec がずれる。
    """
    df = entry
    out = pd.DataFrame({"race_id": df["race_id"].to_numpy(),
                        "horse_no": df["horse_no"].to_numpy()})

    carried = _num(df["weight_carried"]) if "weight_carried" in df.columns else pd.Series(
        np.nan, index=df.index)
    carried = pd.Series(carried.to_numpy(dtype="float64"))
    out["b_weight_carried"] = carried

    # 同一レース内の重量差。ばんえいの重量はハンデそのものなので、絶対値より
    # 「この馬が他馬より何 kg 多く曳くか」が効く。自分を含む平均で引く
    # （レース内の相対量なので、自分を除くと頭数依存の偏りが入る）。
    race_mean = carried.groupby(pd.Series(df["race_id"].to_numpy())).transform("mean")
    out["b_weight_rel"] = carried - race_mean

    mark = (df["weight_carried"].astype(str).str.strip().str[0].isin(list(_APPRENTICE_MARK))
            if "weight_carried" in df.columns else pd.Series(False, index=df.index))
    out["b_is_apprentice"] = np.asarray(mark, dtype="float64")

    body = (pd.to_numeric(df["weight_kg"], errors="coerce") if "weight_kg" in df.columns
            else pd.Series(np.nan, index=df.index))
    body = pd.Series(np.asarray(body, dtype="float64"))
    out["b_body_weight"] = body
    # 曳く重量 / 自分の体重。ばんえいは 0.5-0.9 の範囲に収まる
    out["b_load_ratio"] = carried / body.replace(0.0, np.nan)

    diff = (_num(df["weight_diff"].astype(str).str.replace("±", "0", regex=False))
            if "weight_diff" in df.columns else pd.Series(np.nan, index=df.index))
    sign = (df["weight_diff"].astype(str).str.strip().str.startswith("-")
            if "weight_diff" in df.columns else pd.Series(False, index=df.index))
    out["b_body_weight_diff"] = np.where(np.asarray(sign),
                                         -np.asarray(diff, dtype="float64"),
                                         np.asarray(diff, dtype="float64"))

    # 含水率。ばんえいの `baba_condition` は 良/稍重 ではなく数値（%）
    cond = pd.Series(np.nan, index=range(len(df)), dtype="float64")
    if "baba_condition" in race.columns:
        lookup = race.drop_duplicates("race_id").set_index("race_id")["baba_condition"]
        mapped = pd.Series(df["race_id"].to_numpy()).map(lookup)
        cond = pd.to_numeric(mapped.astype(str).str.extract(_WEIGHT_RE, expand=False),
                             errors="coerce").astype("float64")
        cond = cond.where(cond <= MOISTURE_MAX)
    out["b_moisture"] = cond.to_numpy()

    out["b_age"] = (pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype="float64")
                    if "age" in df.columns else np.nan)
    sex = (df["sex"].astype(str).str.strip() if "sex" in df.columns
           else pd.Series("", index=df.index))
    out["b_is_female"] = np.asarray(sex == "牝", dtype="float64")
    out["b_is_gelding"] = np.asarray(sex.isin(_GELDING), dtype="float64")

    for c in BANEI_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out
