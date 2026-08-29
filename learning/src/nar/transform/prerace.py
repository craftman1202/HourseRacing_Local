"""pre-race スキーマの強制。

設計書 §5 の中核。post-race 列は「落とし忘れ」ではなく「落とせなかったら例外」に
する。ホワイトリストは LK-01..03 の判定が as-of-race を示した列だけが入れる。
"""

from __future__ import annotations

import pandas as pd

from ..errors import LeakageError

# 明らかにレース後の情報
POST_RACE_ENTRY_COLS = frozenset({
    "着順", "タイム", "着差", "上がり3F", "人気",
    # 時点が疑わしい累積成績8列。LK-01..03 の判定が出るまでは全部こちら側
    "騎手成績", "全成績", "ダート左成績", "ダート右成績",
    "当競馬場成績", "うち当距離成績", "最高タイム", "最高タイム良馬場",
})

POST_RACE_RACE_COLS = frozenset(
    {"上がり4F", "上がり3F"}
    | {f"ハロンタイム{i}" for i in range(1, 16)}
    | {f"コーナー名称{i}" for i in range(1, 9)}
    | {f"コーナー通過順{i}" for i in range(1, 9)}
    | {"頭数"}  # 出走取消・除外の反映時点が不明。行数から自前でカウントする
)

POST_RACE_ALL = POST_RACE_ENTRY_COLS | POST_RACE_RACE_COLS

# 累積成績8列。EDA の時点性判定（LK-01..03）の対象そのもの
CUMULATIVE_RECORD_COLS = (
    "騎手成績", "全成績", "ダート左成績", "ダート右成績",
    "当競馬場成績", "うち当距離成績", "最高タイム", "最高タイム良馬場",
)

# as-of-race であることが EDA で確定した列だけをここに追加する。
# 空のまま先に進むのが既定であり、判定を保留したまま埋めてはいけない。
#
# 2026-08、1998-01〜2026-08 の 4,831,906 行で LK-06/07/08 を実施し、8列すべてが
# as-of-race であることを確定させた（artifacts/leak_verdict.json）。決め手は
# 「集計範囲での初回出走行が 0」と「勝利数の増分が前走の勝敗で説明される」の2つ。
#   全成績        初回 58,700 行の 96.3% が 0 / 増分の 99.9% が前走で説明
#   当競馬場成績   初回 144,331 行の 100.0% が 0
#   うち当距離成績 初回 316,960 行の 100.0% が 0
#   最高タイム     馬×場×距離の「当該レースを含まない過去最速」と 99.9% 一致
# 運用面でも当日の出馬表（DebaTable）に同じ8列が載っているので、学習と推論で
# 同じ値を取れる（train-serving skew にならない）。
ASOF_RACE_WHITELIST: frozenset[str] = frozenset(CUMULATIVE_RECORD_COLS)

# 発走前に確定するが、運用時に当日ファイルから取る必要がある列
OPERATIONALLY_CONSTRAINED = frozenset({"天候", "馬場", "馬体重"})


def to_prerace(df: pd.DataFrame, whitelist: frozenset[str] = ASOF_RACE_WHITELIST) -> pd.DataFrame:
    """post-race 列を落とす。落ちたことは呼び出し側で assert_prerace() が保証する。"""
    drop = [c for c in df.columns if c in POST_RACE_ALL and c not in whitelist]
    return df.drop(columns=drop)


def assert_prerace(df: pd.DataFrame, whitelist: frozenset[str] = ASOF_RACE_WHITELIST) -> None:
    """列名集合の積が空であることを検証する（LK-04）。"""
    residual = sorted((set(df.columns) & POST_RACE_ALL) - set(whitelist))
    if residual:
        raise LeakageError(
            f"pre-race スキーマに post-race 列が残っています: {residual}。"
            "to_prerace() を通していないか、ホワイトリストが誤って広げられています。"
        )


def market_columns(df: pd.DataFrame, blocklist: list[str]) -> list[str]:
    """トラックA から排除すべき市場情報由来の列（FE-10）。"""
    lowered = {c: c.lower() for c in df.columns}
    return sorted(
        c for c, low in lowered.items()
        if any(term.lower() in low for term in blocklist)
    )


def assert_no_market_info(df: pd.DataFrame, blocklist: list[str]) -> None:
    hits = market_columns(df, blocklist)
    if hits:
        raise LeakageError(
            f"トラックA の特徴量に市場情報由来の列が含まれています: {hits}。"
            "features_noodds/ は市場と独立であることが唯一の存在理由です。"
        )


def operational_availability(df: pd.DataFrame) -> pd.DataFrame:
    """学習では使うが運用時に取得できるかを明示するフラグ表。

    月次ファイルの `天候`/`馬場` はレース後の最終値なので、運用時は当日ファイルから
    取る必要がある。この差を暗黙にすると本番だけ性能が落ちる。
    """
    rows = [
        {"column": c,
         "available_at_training": True,
         "available_at_inference": c not in OPERATIONALLY_CONSTRAINED,
         "source_at_inference": "daily_file" if c in OPERATIONALLY_CONSTRAINED else "monthly_file"}
        for c in df.columns
    ]
    return pd.DataFrame(rows)
