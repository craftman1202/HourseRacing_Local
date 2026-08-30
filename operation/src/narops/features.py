"""推論時の as-of 特徴量計算とスナップショット保存。

計算そのものは学習側（`nar.features.builder`）を呼ぶ。運用側で書き直さないのが
SK-02 の要件であり、この方針を崩した時点で train-serving skew が不可避になる。

運用側が担うのは「学習時と同じ入力を渡すこと」だけ:
  1. 対象レースの発走時刻より**厳密に前**の履歴に絞る（DB-03）
  2. 確定層とライブ層の UNION を使う（DB-02）
  3. 使った特徴量をそのままスナップショットに残す（SK-06）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from .clock import Clock, business_date, to_utc
from .db.backend import Warehouse
from .db.freshness import watermark
from .db.schema import RECORD_COLUMN_MAP
from .db.types import localize_utc
from .errors import AsOfViolation, InsufficientData
from .model.manifest import FeatureSpec, Manifest, verify_feature_spec
from .shared import ASOF_FEATURES, assert_prerace, to_prerace

# gold に出さない識別列。特徴量ハッシュの対象外
ID_COLUMNS = ("race_id", "horse_no", "horse_sk", "jockey_sk", "trainer_sk", "sire_sk",
              "race_date", "start_ts", "baba_code", "finish_pos", "is_win")


@dataclass
class FeatureResult:
    frame: pd.DataFrame
    spec: FeatureSpec
    as_of_ts: datetime
    db_watermark: datetime | None
    n_runners: int


def history_before(wh: Warehouse, as_of_ts: datetime, lookback_days: int,
                   max_bytes_billed: int | None = None,
                   card: pd.DataFrame | None = None,
                   career_from: date | None = None,
                   course: tuple[int, float] | None = None) -> pd.DataFrame:
    """発走時刻より厳密に前の履歴。

    `<` であって `<=` ではない。同時刻の別レースを含めると、同日同時刻に走る
    他場のレース結果を見たことになる。

    取ってくる範囲は「特徴量が必要とする範囲」で決める。時間窓だけで切ると
    学習時と別物になる。ビルダーの馬・騎手・調教師・種牡馬の集計は全キャリアを
    見るので、直近 lookback_days だけを渡すと実測で h_starts_prior が平均
    22.5 → 5.6（相関 0.51）、j_starts_prior が 8,080 → 365（相関 0.19）まで
    変わる。モデルは学習時と違う入力を見ることになる。

    集めるのは3種類:
      1. 出馬表に出てくる馬・騎手・調教師・種牡馬の**全キャリア**
      2. 直近 lookback_days の全レース（収縮の事前確率）

    速度指数の標準化に使う 場×距離 の統計は取らない。確定層に保存済みの
    速度指数をそのまま使う設計にしたので、母集団を引き直す必要がない。
    以前は同じ場の全履歴（園田で 47.6 万行）を毎回引いており、特徴量構築が
    1レース 48 秒かかっていた。

    これを1本の SQL にまとめると、DuckDB が
    `race_date >= ? AND start_ts < ? AND <列> IN (...)` の3条件で実行計画を
    崩し、10万行のクエリが数十分たっても返らなくなる（2条件までは 0.5 秒）。
    条件を2つに抑えた複数クエリに分け、発走時刻での厳密な絞り込みは
    pandas 側で行う。分けても意味は変わらず、実測で合計 1.6 秒に収まる。
    """
    as_of = to_utc(as_of_ts)
    naive = as_of.replace(tzinfo=None)
    start_date = (as_of - pd.Timedelta(days=lookback_days)).date()
    upto = as_of.date()

    def run(where: str, params: list) -> pd.DataFrame:
        """確定層とライブ層を別々に引いて、pandas 側で二層を合成する。

        `entry_history` ビューを経由しない。ビューは
        `UNION ALL` + `race_id NOT IN (SELECT race_id FROM entry_result_final)`
        で、`IN (...)` フィルタと組み合わせて 10 万行規模を返させると、
        DuckDB の実行計画が崩れて数十分たっても返らなくなる
        （同じ述語を基底テーブルに当てれば 0.5 秒）。

        意味はビューと同じ。両層は同じレースについて同じキー列を持つので、
        同一の述語で引いた断片どうしで race_id を突き合わせれば、
        「確定層にあるレースはライブ層を捨てる」を正しく再現できる。
        """
        final = wh.query(f"SELECT * FROM entry_result_final WHERE {where}", params,
                         max_bytes_billed=max_bytes_billed, allow_full_scan=True)
        live = wh.query(f"SELECT * FROM entry_result_live WHERE {where}", params,
                        max_bytes_billed=max_bytes_billed, allow_full_scan=True)
        if len(live):
            live = live[~live["race_id"].isin(set(final["race_id"]))]
        cols = [c for c in final.columns if c in live.columns] or list(final.columns)
        parts = [f for f in (final[cols], live[cols] if len(live) else None)
                 if f is not None and len(f)]
        return pd.concat(parts, ignore_index=True) if parts else final[cols]

    frames: list[pd.DataFrame] = []
    if card is None or card.empty:
        frames.append(run("race_date >= ? AND race_date <= ?", [start_date, upto]))
    else:
        for col in ("horse_sk", "jockey_sk", "trainer_sk", "sire_sk"):
            values = _key_values(card, col)
            if not values:
                continue
            placeholders = ", ".join("?" for _ in values)
            frames.append(run(f"race_date <= ? AND {col} IN ({placeholders})",
                              [upto, *values]))
        frames.append(run("race_date >= ? AND race_date <= ?", [start_date, upto]))

    raw = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=["race_id", "horse_no"]))
    if len(raw):
        raw = raw.drop_duplicates(["race_id", "horse_no"])

    # DB 内の naive timestamp は UTC。読み出したら必ず tz を付け直す。
    # naive のまま返すと、tz-aware な as_of_ts と混ざった列が object dtype になり、
    # 下流の DuckDB 登録で落ちる（そして落ちなければ 9 時間ずれる）。
    out = localize_utc(raw)
    if len(out):
        # 発走時刻より厳密に前だけを残す（DB-03）。SQL に持たせると上記の
        # 実行計画の問題を踏むので、ここで確実に切る。
        out = out[pd.to_datetime(out["start_ts"]) < as_of]
    # DB は ASCII 列名、学習側のビルダーは月次ファイルの日本語ヘッダを見る。
    # ここで戻さないと申告値特徴量が全行 NaN になり、推論だけ別の入力を見る。
    return out.rename(columns=RECORD_COLUMN_MAP).reset_index(drop=True)


def _key_values(card: pd.DataFrame, col: str) -> list[str]:
    if col not in card.columns:
        return []
    return sorted({str(v) for v in card[col].dropna() if str(v).strip()})


def build_for_race(
    wh: Warehouse,
    entry_card: pd.DataFrame,
    race_row: pd.Series,
    manifest: Manifest,
    feature_config,
    max_bytes_billed: int | None = None,
) -> FeatureResult:
    """1レース分の as-of 特徴量を作る。

    entry_card は当日ファイル由来の出馬表（結果列を含まない）。
    """
    from nar.features.builder import build as build_features

    if entry_card.empty:
        raise InsufficientData("出馬表が空です")
    if entry_card["horse_no"].isna().any():
        raise InsufficientData("枠順・馬番が未確定のため推論しません（IN-09）")

    as_of_ts = to_utc(race_row["start_ts"])

    # post-race 列を物理的に落とす。学習時と同一の関数を通す
    card = to_prerace(entry_card.copy())
    assert_prerace(card)

    history = history_before(wh, as_of_ts, manifest.lookback_days, max_bytes_billed,
                             card=card)
    # `src` は二層のどちらから来たかを示す運用側の列。学習側のビルダーは知らないので
    # ここで落とす。残すと対象レース行との列構成が食い違う。
    history = history.drop(columns=[c for c in ("src",) if c in history.columns])
    if len(history) and to_utc(pd.to_datetime(history["start_ts"]).max()) >= as_of_ts:
        raise AsOfViolation(
            "発走時刻以降のレコードが履歴に含まれています。as-of フィルタが壊れています。")

    # 対象レースは「結果未確定の行」として履歴に足す。学習側のウィンドウ関数は
    # EXCLUDE CURRENT ROW なので、自分の結果は特徴量に入らない。
    # 結果列は float の NaN で置く。pd.NA を混ぜると concat で object 列になり、
    # dtype が history 側と食い違って feature_spec ハッシュが揺れる。
    # 回りと馬場状態はレース表の属性。対象レース行に載せないと、`ダート左/右成績`
    # と `最高タイム良馬場` の選択条件が決まらず d_turn_* が NaN になる。
    target = card.assign(finish_pos=float("nan"), is_win=0, time_sec=float("nan"),
                         speed_index=float("nan"),
                         race_date=race_row["race_date"], start_ts=as_of_ts,
                         baba_code=race_row["baba_code"], distance=race_row["distance"],
                         turn=race_row.get("turn", ""),
                         baba_condition=race_row.get("baba_condition", ""))
    # 列は履歴側に合わせる。ただし `target[history.columns]` と書くと、出馬表にしか
    # 無い列（累積成績8列）がここで落ちる。履歴に無い列は履歴側を NULL で埋める。
    missing_in_history = [c for c in target.columns if c not in history.columns]
    if missing_in_history and len(history):
        history = history.assign(**{c: pd.NA for c in missing_in_history})
    cols = list(history.columns) if len(history) else list(target.columns)
    for c in missing_in_history:
        if c not in cols:
            cols.append(c)
    combined = pd.concat([history.reindex(columns=cols), target.reindex(columns=cols)],
                         ignore_index=True)

    race_meta = pd.DataFrame([{
        "race_id": race_row["race_id"], "race_date": race_row["race_date"],
        "start_ts": as_of_ts, "baba_code": race_row["baba_code"],
        "race_no": race_row.get("race_no", 1), "distance": race_row["distance"],
        "surface": race_row.get("surface", "ダ"),
        # 回りは `ダート左/右成績` の選択条件。落とすと d_turn_* が全部 NaN になる
        "turn": race_row.get("turn", ""),
        "baba_condition": race_row.get("baba_condition", ""),
        "class_level": race_row.get("class_level", 1),
        "prize_yen": race_row.get("prize_yen", 1_000_000),
        "n_runners": len(card),
    }])
    hist_races = (history[["race_id", "race_date", "start_ts", "baba_code", "distance"]]
                  .drop_duplicates("race_id"))
    if len(hist_races):
        counts = history.groupby("race_id").size().rename("n_runners")
        hist_races = hist_races.join(counts, on="race_id")
        turn_by_race = (history.drop_duplicates("race_id").set_index("race_id")["turn"]
                        if "turn" in history.columns else None)
        hist_races = hist_races.assign(
            race_no=1, surface="ダ", class_level=1, prize_yen=1_000_000,
            turn=(hist_races["race_id"].map(turn_by_race) if turn_by_race is not None
                  else ""),
            baba_condition="")
        race_meta = pd.concat([hist_races[race_meta.columns], race_meta], ignore_index=True)

    feat = build_features(combined, race_meta, feature_config)
    out = feat[feat["race_id"] == race_row["race_id"]].copy()
    if out.empty:
        raise InsufficientData(f"{race_row['race_id']} の特徴量が生成されませんでした")

    # 契約は manifest が宣言した列。宣言が無い古いリリースは従来どおり全列で組む。
    declared = getattr(manifest.feature_spec, "names", None) if getattr(
        manifest, "feature_spec", None) else None
    assert_serving_inputs(out, feature_config, declared)
    spec = feature_spec_of(out, declared)
    verify_feature_spec(spec, manifest)
    return FeatureResult(out.sort_values("horse_no").reset_index(drop=True), spec,
                         as_of_ts, watermark(wh), len(out))


def assert_serving_inputs(frame: pd.DataFrame, feature_config,
                          declared: tuple[str, ...] | list[str] | None = None) -> None:
    """埋めてはいけない列が欠測のまま来ていないか（IN-09 と同じ考え方）。

    標準化器は欠測を学習時の中央値で埋める。平地ではそれで良いが、ばんえいの
    重量は競技のハンデそのもので、埋めた時点で事実と違う前提の予測になる。
    `b_weight_rel` はレース内の相対量なので、一部の馬だけ埋まると馬同士の
    優劣が直接歪む。

    実測（2026-08-30）: 開催日の早朝の出馬表には馬体重が1頭も載らない。
    発走13分前には載る。つまり「普段は取れるが取れない時もある」列で、
    黙って埋めると静かに劣化した予測が配信まで流れる。止めるほうを選ぶ。

    要求するのは**このモデルが実際に使う列**だけ（manifest の宣言と突き合わせる）。
    使っていない列の欠測で推論を止める理由は無い。
    """
    if getattr(feature_config, "variant", "flat") != "banei":
        return
    from nar.features.banei import REQUIRED_AT_SERVING, SERVING_RANGES

    used = set(declared) if declared else set(frame.columns)
    for col in REQUIRED_AT_SERVING:
        if col not in used or col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        n_missing = int(values.isna().sum())
        if n_missing:
            raise InsufficientData(
                f"{col} が {n_missing}/{len(frame)} 頭で欠測しています。"
                "ばんえいの重量は競技のハンデそのもので、中央値で埋めると"
                "事実と違う前提の予測になります（出馬表への掲載待ちの可能性）。"
                "この状態では推論しません。")

    # 欠測ではなく「値はあるが明らかにおかしい」を捕まえる。当日ページの
    # 解析ミスは欠測ではなく異常値として出ることがあり（4桁の馬体重が
    # 下3桁だけ読まれるなど）、そちらのほうが静かで危ない。
    for col, (lo, hi) in SERVING_RANGES.items():
        if col not in used or col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        bad = values.notna() & ((values < lo) | (values > hi))
        if bad.any():
            raise InsufficientData(
                f"{col} に妥当域 [{lo:g}, {hi:g}] を外れる値が "
                f"{int(bad.sum())}/{len(frame)} 頭あります"
                f"（例: {sorted(values[bad].tolist())[:3]}）。"
                "当日ページの解析結果を疑ってください。この状態では推論しません。")


def feature_spec_of(frame: pd.DataFrame,
                    names: tuple[str, ...] | list[str] | None = None) -> FeatureSpec:
    """推論が実際に作った特徴量から spec を組む。

    名前だけでなく**順序**を保つ。DataFrame の列順の揺れがハッシュに乗らない
    ようにする。

    names を渡すと、その列だけで spec を作る。spec は「モデルが食う入力の契約」
    なので、特徴量選択で 35 列から 16 列に絞ったなら、契約もその 16 列。
    ビルダーが返すフレームには選ばれなかった列も入っている（スナップショットに
    残すため）が、それを契約に混ぜると manifest と永久に一致しない。
    """
    if names is None:
        names = tuple(c for c in ASOF_FEATURES if c in frame.columns)
    else:
        missing_cols = [n for n in names if n not in frame.columns]
        if missing_cols:
            raise InsufficientData(
                f"モデルが要求する特徴量が生成されていません: {missing_cols}")
        names = tuple(names)
    dtypes = {n: str(frame[n].dtype) for n in names}
    # 学習側は初出走の勝率を 0 ではなく NULL にしている。その表現を凍結する（SK-04）
    missing = {n: "nan" for n in names}
    return FeatureSpec(names, dtypes, missing)


def save_snapshot(wh: Warehouse, result: FeatureResult, race_id: str,
                  model_release: str, clock: Clock) -> int:
    """推論に使った特徴量を必ず残す（SK-06）。

    後日 NAR 側で着順訂正が入っても「推論時点で何を見ていたか」が完全に残る。
    """
    now = clock.now()
    rows = []
    for _, r in result.frame.iterrows():
        payload = {n: (None if pd.isna(r[n]) else float(r[n])) for n in result.spec.names}
        rows.append({
            "race_id": race_id, "horse_no": int(r["horse_no"]),
            "as_of_date": business_date(result.as_of_ts),
            "computed_at": now, "as_of_ts": result.as_of_ts,
            "db_watermark": result.db_watermark, "model_release": model_release,
            "features": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "feature_spec_hash": result.spec.hash(),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    # as_of_date を条件に入れる。feature_snapshot は月パーティションで
    # require_partition_filter が付いており、条件の無い削除は BigQuery に
    # 実行そのものを拒否される（推論がここで落ちていた）。1レースの行は
    # すべて同じ営業日なので先頭の値で足りる。
    as_of = df["as_of_date"].iloc[0]
    wh.execute("DELETE FROM feature_snapshot "
               "WHERE as_of_date = ? AND race_id = ? AND model_release = ?",
               [as_of, race_id, model_release])
    wh.insert_frame("feature_snapshot", df)
    return len(df)


def load_snapshot(wh: Warehouse, business_day, max_bytes_billed: int | None = None
                  ) -> pd.DataFrame:
    df = wh.query(
        "SELECT * FROM feature_snapshot WHERE as_of_date = ?",
        [business_day], max_bytes_billed=max_bytes_billed)
    if df.empty:
        return df
    expanded = pd.json_normalize(df["features"].map(json.loads))
    return pd.concat([df.drop(columns=["features"]).reset_index(drop=True),
                      expanded.reset_index(drop=True)], axis=1)
