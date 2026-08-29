"""DB 境界での型の正規化。

DuckDB / BigQuery に tz-aware な datetime を渡すと、ドライバがローカル時刻へ
変換したうえで tz を落とす。往復すると「保存は UTC」（DB-10）が黙って破れ、
JST-naive な値が残る。as-of フィルタは UTC 基準で比較するので、この 9 時間の
ズレはそのまま「発走後のレコードを履歴に入れる」事故になる。

そこで **DB 内の naive timestamp は常に UTC を表す** という規約を1か所で強制する。
書き込み前に canonicalize_frame、読み出し後に localize_utc を必ず通す。
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ..clock import UTC, to_utc

TIMESTAMP_COLUMNS = ("start_ts", "captured_at", "merged_at", "computed_at",
                     "as_of_ts", "db_watermark", "sent_at", "changed_at", "run_at")
DATE_COLUMNS = ("race_date", "as_of_date", "business_date")


def canonicalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """書き込み前の正規化。

    - timestamp 列: UTC に変換したうえで tz を外す（naive = UTC の規約）
    - date 列: Python の date に揃える（datetime のまま入れると 00:00:00 が付く）
    """
    out = df.copy()
    for c in TIMESTAMP_COLUMNS:
        if c in out.columns:
            out[c] = _to_utc_naive(out[c])
    for c in DATE_COLUMNS:
        if c in out.columns:
            out[c] = _to_date(out[c])
    return out


def localize_utc(df: pd.DataFrame) -> pd.DataFrame:
    """読み出し後の復元。naive な timestamp に UTC を付け直す。"""
    out = df.copy()
    for c in TIMESTAMP_COLUMNS:
        if c in out.columns and len(out):
            s = pd.to_datetime(out[c], errors="coerce")
            out[c] = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    for c in DATE_COLUMNS:
        if c in out.columns and len(out):
            out[c] = _to_date(out[c])
    return out


def _to_utc_naive(s: pd.Series) -> pd.Series:
    """tz-aware なら UTC 化して tz を外す。naive は**既に UTC** とみなして触らない。

    冪等であることが要件。`clock.to_utc` は naive を JST とみなすので、これを
    ここで使うと2回通すたびに 9 時間ずれる。JST の naive を扱うのは NAR ファイルの
    パース時点の責務で、DB 境界に来た時点では既に tz が付いているか UTC のはず。
    """
    if s.isna().all():
        return pd.to_datetime(s, errors="coerce")
    parsed = pd.to_datetime(s, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    if parsed.dtype == object:
        # aware と naive が混在した列。aware だけ UTC 化して揃える
        conv = parsed.map(
            lambda v: v.astimezone(UTC).replace(tzinfo=None)
            if isinstance(v, datetime) and v.tzinfo is not None else v)
        return pd.to_datetime(conv, errors="coerce")
    return parsed


def _to_date(s: pd.Series) -> pd.Series:
    def one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        return pd.Timestamp(v).date()

    return s.map(one)


def comparable(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """MERGE の差分判定用に、型のゆれを潰した比較用フレームを作る。"""
    out = canonicalize_frame(df[columns])
    for c in columns:
        if c in DATE_COLUMNS:
            out[c] = out[c].map(lambda v: v.isoformat() if v is not None else "")
        elif c in TIMESTAMP_COLUMNS:
            out[c] = pd.to_datetime(out[c], errors="coerce").astype("datetime64[us]")
        elif pd.api.types.is_integer_dtype(out[c]):
            # 取消・除外の馬は着順が NULL。nullable な整数列を素の int64 に
            # 落とすと "cannot convert NA to integer" で止まる。比較用なので
            # 欠損を保てる float64 で揃える（値域は int の範囲に収まる）。
            out[c] = (out[c].astype("float64") if out[c].isna().any()
                      else out[c].astype("int64"))
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype("float64")
        else:
            out[c] = out[c].astype(str)
    return out


# NAR の発走時刻は JST 10:00〜21:00 に収まる（実データ 483 万行で確認）。
# UTC に直すと 01:00〜12:00。DB 内の naive を UTC とみなす規約（DB-10）が
# 守られていれば、保存された時刻はこの窓に入る。
UTC_RACING_HOURS = range(0, 14)
JST_RACING_HOURS = range(10, 22)


def assert_utc_start_ts(df, column: str = "start_ts",
                        min_share: float = 0.98) -> None:
    """発走時刻が UTC で入っていることを検査する。

    JST の naive をそのまま入れると 9 時間ずれ、`start_ts < 発走時刻` の比較で
    **同日の先行レースが丸ごと履歴から落ちる**。実際にそれが起き、園田5R の
    推論で同日の3R・4R が見えなくなっていた。値そのものは妥当に見えるので、
    分布で検出する以外に気付きようがない。
    """
    import pandas as pd

    if column not in df.columns or df.empty:
        return
    ts = pd.to_datetime(df[column], errors="coerce")
    ts = ts.dt.tz_convert("UTC") if getattr(ts.dt, "tz", None) else ts
    hours = ts.dt.hour.dropna()
    if hours.empty:
        return
    in_utc = float(hours.isin(list(UTC_RACING_HOURS)).mean())
    in_jst = float(hours.isin(list(JST_RACING_HOURS)).mean())
    if in_utc < min_share and in_jst > in_utc:
        raise ValueError(
            f"{column} が JST のまま入っているようです（UTC 想定の時間帯に "
            f"{in_utc:.1%} しか収まらず、JST 想定なら {in_jst:.1%}）。"
            "DB 内の naive は UTC という規約（DB-10）に直してください。"
            "ずれたまま入れると同日の先行レースが履歴から落ちます。")
