"""as-of 特徴量ビルダー。

すべての履歴系特徴量は「当該レースの発走時刻より厳密に前」のレコードのみから計算する。
実装上の担保は EXCLUDE CURRENT ROW と ROWS BETWEEN ... AND 1 PRECEDING の2つで、
これが1か所でも漏れると LK-05（未来汚染テスト）が落ちる。
"""

from __future__ import annotations

import hashlib
import logging

import duckdb
import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..transform.prerace import (
    ASOF_RACE_WHITELIST, assert_no_market_info, assert_prerace,
)
from . import declared
from .declared import DECLARED_FEATURES
from .shrinkage import sql_shrink

log = logging.getLogger(__name__)

# 出力される特徴量列。registry と builder がズレないよう1か所で定義する。
ASOF_FEATURES = (
    "h_starts_prior", "h_wins_prior", "h_winrate_prior", "h_top3rate_prior",
    "h_si_last3", "h_days_since_prev", "h_best_si_prior", "is_first_start",
    "j_starts_prior", "j_winrate_shrunk",
    "t_starts_prior", "t_winrate_shrunk",
    "s_starts_prior", "s_winrate_shrunk",
    "field_size", "draw_rel", "distance", "class_level", "log_prize",
)

# NAR 申告の累積成績由来。EDA で as-of-race を確定させた列だけが元になる
# （ASOF_RACE_WHITELIST が空なら1つも作らない）。
ASOF_FEATURES = ASOF_FEATURES + DECLARED_FEATURES

MARKET_FEATURES = ("odds_win", "market_implied_logit", "popularity")


# 全順序。start_ts だけでは同着ならぬ「同時刻の別場」で並びが決まらず、
# ROWS フレームの境界が非決定になる。決定性（FE-09）とリーク検証（LK-05）の両方に効く。
_ORDER = "start_ts, race_id, horse_no"
_ORDER_COLS = ("start_ts", "race_id", "horse_no")
# 「発走時刻が厳密に前」の行だけを集計する。ROWS ... EXCLUDE CURRENT ROW だと、
# 全順序上で手前に来る**同時刻の別レース**を含んでしまう。同じ分に発走する他場の
# レースは、対象レースの発走時点でまだ終わっていないので未来情報にあたる。
# GROUPS は ORDER BY の値が同じ行を1グループとして扱うので、
# 「1グループ前まで」= 同時刻を丸ごと外す、になる。運用側の `start_ts <` と一致する。
_PRIOR_ORDER = "start_ts"
_PRIOR_FRAME = "GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"

# 標準化の基準統計量を as-of で取る以上、場×距離の最初の数走は「3件から推定した
# 標準偏差」で割ることになり、速度指数が桁で暴れる。基準が固まるまでは NULL を返す。
MIN_PRIOR_FOR_SPEED_INDEX = 100


def _recent_mean_sql(col: str, n: int, window: str = "w2") -> str:
    """直近 n 件の平均を LAG の明示的な和で書く。

    `AVG(x) OVER (ROWS BETWEEN n PRECEDING AND 1 PRECEDING)` は、DuckDB が
    ストリーミングで移動和を持ち回るため、並列実行のたびに加算順が変わる。
    差は 1e-14 程度だが、GOLD_DECIMALS の丸め境界をまたぐ行が稀に出て
    （実データ 409,663 行中 6 行）、LK-05 のビット単位一致が落ちる。

    LAG は各行を独立に引くので蓄積が無い。和の順序も式木で固定される。
    NULL は分母から外す（初出走から数走はまだ n 件揃わない）。
    """
    terms = [f"LAG({col}, {i}) OVER {window}" for i in range(1, n + 1)]
    total = " + ".join(f"COALESCE({t}, 0)" for t in terms)
    count = " + ".join(f"CASE WHEN {t} IS NULL THEN 0 ELSE 1 END" for t in terms)
    return f"({total}) / NULLIF({count}, 0)"


def speed_index(entry: pd.DataFrame) -> pd.DataFrame:
    """速度指数を as-of で計算する。

    地方は場ごとのコース形態差が極端なので、場×距離で標準化しないと場をまたいだ
    比較が成立しない。

    馬場差の補正には「その日の同場同距離の平均タイム」を使うが、当日の**先行レース
    のみ**に限る。当日全レースの平均を使うと、同日の後続レースの結果が先行レースの
    速度指数に混入し、同一馬が同日に複数回走ったときにリークする。先行レースが
    まだ無い場合は、同場同距離の as-of 累積平均にフォールバックする。

    DuckDB のウィンドウ関数ではなく numpy の累積和で書く。
    `AVG(x) OVER (UNBOUNDED PRECEDING ... EXCLUDE CURRENT ROW)` は segment tree で
    評価され、木の形がパーティション全長に依存する。同じ前方要素でも、後ろに続く
    行数が変われば加算順が変わり、結果が 1e-14 だけずれる。未来の行を削っただけで
    過去の特徴量が変わるので、LK-05（ビット単位一致）が落ちる。
    累積和は定義上その位置までの要素だけで決まるので、後ろに何が来ても不変。
    """
    df = entry.sort_values(list(_ORDER_COLS), kind="stable").reset_index(drop=True)
    if "speed_index" in entry.columns:
        # 既に計算済みの列があるなら、一切計算し直さない。
        # 速度指数は 場×距離 の as-of 統計で標準化するため、履歴の部分集合から
        # 計算すると同じ走破時計でも別の値になる。運用時に渡せるのは常に部分集合。
        # 「保存値を優先しつつ欠損だけ埋める」だと、学習時に NULL だった行に
        # 部分集合由来の値が入り、そこだけ学習と食い違う。
        df["speed_index"] = pd.to_numeric(df["speed_index"], errors="coerce")
        return df
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(t)
    filled = np.where(valid, t, 0.0)

    order_col = "start_ts"

    def prior_stats(keys: list[str], want_sd: bool):
        """グループごとに「自分より前の行だけ」の件数・平均・標準偏差。

        全体の累積和からグループ開始時点の値を引く書き方はしない。浮動小数点では
        その引き算に手前の全グループの合計が乗るため、後ろの行を削っただけで
        過去の値が変わる。pandas の groupby.cumsum はグループごとに積み直すので、
        そのグループの前方要素だけで決まる。
        """
        work = pd.DataFrame({"n": valid.astype(float), "x": filled}, index=df.index)
        if want_sd:
            work["x2"] = filled * filled
        incl = work.groupby([df[k] for k in keys], sort=False).cumsum()
        # 同時刻の行はまとめて外す。自分の行だけ除いても、同じ分に発走する
        # 他場のレースが残り、対象レース発走時点でまだ終わっていない結果を見る。
        # DuckDB 側の GROUPS フレームと同じ扱いに揃える。
        peer = work.groupby([df[k] for k in keys] + [df[order_col]], sort=False).transform("sum")
        n_prior = incl["n"].to_numpy() - peer["n"].to_numpy()
        x_prior = incl["x"].to_numpy() - peer["x"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.divide(x_prior, n_prior, out=np.full_like(x_prior, np.nan),
                             where=n_prior > 0)
        if not want_sd:
            return n_prior, mean
        x2_prior = incl["x2"].to_numpy() - peer["x2"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.divide(x2_prior - n_prior * mean * mean, n_prior - 1,
                            out=np.full_like(x_prior, np.nan), where=n_prior > 1)
        return n_prior, mean, np.sqrt(np.clip(var, 0.0, None))

    _, day_mean = prior_stats(["baba_code", "distance", "race_date"], want_sd=False)
    all_n, all_mean, all_sd = prior_stats(["baba_code", "distance"], want_sd=True)

    ref = np.where(np.isfinite(day_mean), day_mean, all_mean)
    with np.errstate(invalid="ignore", divide="ignore"):
        si = -1.0 * (t - ref) / np.where(all_sd > 0, all_sd, np.nan)
    # 標準化の基準が固まるまでは返さない（3件から推定した標準偏差で割ると桁で暴れる）
    si = np.where(all_n >= MIN_PRIOR_FOR_SPEED_INDEX, si, np.nan)

    # 既に値が入っている行はそのまま使う。速度指数は 場×距離 の as-of 統計で
    # 標準化するため、全履歴の一部だけを渡すと同じ走破時計でも別の値になる。
    # 運用時に渡せるのは出走馬・騎手などに絞った部分集合なので、確定層に保存済みの
    # 値（全履歴から計算したもの）を優先しないと学習時と一致しない。
    if "speed_index" in entry.columns:
        stored = pd.to_numeric(df["speed_index"], errors="coerce").to_numpy(dtype=float)
        si = np.where(np.isnan(stored), si, stored)
    df["speed_index"] = si
    return df


def build(
    entry: pd.DataFrame,
    race: pd.DataFrame,
    cfg: FeatureConfig,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """entry(結果込み) と race から as-of 特徴量行列を作る。

    entry には finish_pos / time_sec が入っている必要があるが、当該行の結果は
    ウィンドウから除外されるので出力には漏れない。
    """
    own_con = con is None
    con = con or duckdb.connect()
    try:
        # 実数を畳み込むウィンドウは speed_index() 側で pandas に移してあるので、
        # ここに残る DuckDB のウィンドウは COUNT と SUM/AVG(is_win)（0/1 の整数）
        # だけ。整数和は double で厳密に表せるので加算順に依存しない。
        # h_si_last3 も LAG を明示的に足しているだけで蓄積が無い。
        # したがってスレッド数を絞る必要はない。決定性は FE-09（同じ入力で
        # 同じハッシュ）と LK-05（未来を汚してもビット一致）が担保する。
        # 1スレッドに固定していた頃は推論1レースの特徴量構築が 48 秒かかっていた。
        entry = entry[~entry["baba_code"].isin(cfg.exclude_baba_codes)].copy()
        con.register("entry_raw", entry)
        con.register("race_raw", race)
        con.register("entry_si_raw", speed_index(entry))
        con.execute("CREATE OR REPLACE TEMP VIEW entry AS SELECT * FROM entry_raw")
        con.execute("CREATE OR REPLACE TEMP VIEW entry_si AS SELECT * FROM entry_si_raw")

        a = cfg.shrinkage
        n_recent = cfg.windows["recent_form_n"]
        # 収縮の事前確率は「出走頭数の逆数」。当初は全レースの as-of 累積勝率を
        # 使っていたが、これは**データセット全体に依存する量**で、推論時に渡せる
        # 部分集合からは再現できない（実測で収縮勝率が 4 桁目からずれた）。
        # n 頭立てで1頭が勝つ以上、事前確率 1/n はレース内で完結し、学習時と
        # 推論時で必ず一致する。値としても全期間平均（約 0.107）とほぼ同じで、
        # 頭数の違いを正しく反映する分こちらが妥当。
        prior = "1.0 / NULLIF(e.n_runners, 0)"

        sql = f"""
        WITH e AS (
          SELECT *,
            -- 頭数はレース表の `頭数` 列を使わない。出走取消の反映時点が不明で、
            -- 学習時と推論時で意味が変わる。出馬表の行数なら両方で同じに数えられる。
            COUNT(*) OVER (PARTITION BY race_id) AS n_runners
          FROM entry_si
        ),
        horse AS (
          SELECT race_id, horse_no,
            COUNT(*)      OVER w                       AS h_starts_prior,
            COALESCE(SUM(is_win) OVER w, 0)            AS h_wins_prior,
            AVG(is_win)   OVER w                       AS h_winrate_prior,
            AVG(CASE WHEN finish_pos <= 3 THEN 1 ELSE 0 END) OVER w AS h_top3rate_prior,
            {_recent_mean_sql("speed_index", n_recent)} AS h_si_last3,
            MAX(speed_index) OVER w                    AS h_best_si_prior,
            DATE_DIFF('day', LAG(race_date) OVER w2, race_date) AS h_days_since_prev
          FROM e
          WINDOW w  AS (PARTITION BY horse_sk ORDER BY {_PRIOR_ORDER} {_PRIOR_FRAME}),
                 w2 AS (PARTITION BY horse_sk ORDER BY {_ORDER})
        ),
        jockey AS (
          SELECT race_id, horse_no,
            COUNT(*) OVER wj                AS j_starts_prior,
            COALESCE(SUM(is_win) OVER wj,0) AS j_wins_prior
          FROM e
          WINDOW wj AS (PARTITION BY jockey_sk ORDER BY {_PRIOR_ORDER} {_PRIOR_FRAME})
        ),
        trainer AS (
          SELECT race_id, horse_no,
            COUNT(*) OVER wt                AS t_starts_prior,
            COALESCE(SUM(is_win) OVER wt,0) AS t_wins_prior
          FROM e
          WINDOW wt AS (PARTITION BY trainer_sk ORDER BY {_PRIOR_ORDER} {_PRIOR_FRAME})
        ),
        sire AS (
          SELECT race_id, horse_no,
            COUNT(*) OVER ws                AS s_starts_prior,
            COALESCE(SUM(is_win) OVER ws,0) AS s_wins_prior
          FROM e
          WINDOW ws AS (PARTITION BY sire_sk ORDER BY {_PRIOR_ORDER} {_PRIOR_FRAME})
        )
        SELECT
          e.race_id, e.horse_no, e.horse_sk, e.race_date, e.start_ts, e.baba_code,
          -- エンティティキーは特徴量ではないが、階層ベイズが部分プーリングの
          -- 単位として必要とする。gold に持たせないと下流が silver を引き直す。
          e.jockey_sk, e.trainer_sk, e.sire_sk,
          e.finish_pos, e.is_win,
          h.h_starts_prior, h.h_wins_prior, h.h_winrate_prior, h.h_top3rate_prior,
          h.h_si_last3, h.h_best_si_prior, h.h_days_since_prev,
          CASE WHEN h.h_starts_prior = 0 THEN 1 ELSE 0 END AS is_first_start,
          j.j_starts_prior,
          {sql_shrink("j.j_wins_prior", "j.j_starts_prior", prior, a["alpha_jockey"])}
            AS j_winrate_shrunk,
          t.t_starts_prior,
          {sql_shrink("t.t_wins_prior", "t.t_starts_prior", prior, a["alpha_trainer"])}
            AS t_winrate_shrunk,
          s.s_starts_prior,
          {sql_shrink("s.s_wins_prior", "s.s_starts_prior", prior, a["alpha_sire"])}
            AS s_winrate_shrunk,
          e.n_runners AS field_size,
          (e.horse_no - 1.0) / NULLIF(e.n_runners - 1.0, 0) AS draw_rel,
          e.distance, r.class_level, LN(r.prize_yen) AS log_prize
        FROM e
        JOIN horse   h USING (race_id, horse_no)
        JOIN jockey  j USING (race_id, horse_no)
        JOIN trainer t USING (race_id, horse_no)
        JOIN sire    s USING (race_id, horse_no)
        JOIN race_raw r USING (race_id)
        ORDER BY {", ".join("e." + c for c in _ORDER.split(", "))}
        """
        out = con.execute(sql).df()
    finally:
        if own_con:
            con.close()

    # 初出走は勝率を 0 ではなく NULL にする。0 は「勝てない馬」を意味してしまう（FE-02）
    first = out["is_first_start"] == 1
    for c in ("h_winrate_prior", "h_top3rate_prior", "h_si_last3", "h_best_si_prior"):
        out.loc[first, c] = pd.NA

    # 申告値由来の特徴量。時点性が確定した列だけを使う。
    # 判定を通していない列が混ざれば LK-04 がここで止める。
    usable = sorted(set(entry.columns) & set(ASOF_RACE_WHITELIST))
    if usable:
        assert_prerace(entry[usable].head(0), whitelist=ASOF_RACE_WHITELIST)
        dec = declared.build(entry, race, alpha=a["alpha_jockey"],
                             own_starts=None, prior_win=None)
        dec = dec.set_index(["race_id", "horse_no"])
        idx = pd.MultiIndex.from_arrays([out["race_id"], out["horse_no"]])
        for c in DECLARED_FEATURES:
            out[c] = dec[c].reindex(idx).to_numpy() if c in dec.columns else np.nan
        # 自前集計との差＝1998年以前と中央での出走。ここだけは out 側に揃ってから出す
        extra = out["d_all_starts"] - out["h_starts_prior"]
        out["d_extra_starts"] = extra.clip(lower=0).astype("float64")
        out["d_has_outside_history"] = (extra > 0).astype("float64")
    else:
        log.warning("as-of-race が確定した累積成績列がありません。"
                    "申告値ベースの特徴量は作りません（nar leak-check 未実行）。")

    # 全行 NaN でも列は落とさない。落とすと特徴量スキーマがデータ依存になり、
    # 学習時と推論時で列集合がずれて feature_spec が一致しなくなる（SK-03）。
    # 全 NaN 列の補完は prepare() の責務（中央値が NaN のときは 0 に倒す）。
    empty = [c for c in DECLARED_FEATURES
             if c in out.columns and out[c].isna().all()]
    if empty:
        log.warning("全行 NaN の申告値特徴量があります（列は残します）: %s", empty)

    if cfg.track == "A":
        assert_no_market_info(out, cfg.market_term_blocklist)
    return finalize(out)


# DuckDB のウィンドウ集計は並列実行のため加算順が実行ごとに変わりうる。差は 1e-14
# 程度だが、それだけで gold のハッシュが変わり FE-09（決定性）と LK-05（ビット単位の
# 一致）が落ちる。境界で丸めて成果物を決定的にする。丸め幅は特徴量の意味に対して
# 十分小さく、下流の推定には影響しない。
GOLD_DECIMALS = 10


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 特徴量の dtype を float64 に固定する。COUNT 由来の列は int64、距離は入力の
    # 型次第で int/float が入れ替わり、学習時と推論時で feature_spec が一致しなく
    # なる（実データで distance と d_*_starts の5列が食い違った）。値は変わらない。
    for c in ASOF_FEATURES:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in out.select_dtypes("float").columns:
        out[c] = out[c].round(GOLD_DECIMALS)
    return out


def add_market_features(feat: pd.DataFrame, odds: pd.DataFrame, takeout: float) -> pd.DataFrame:
    """トラックB のみ。市場暗黙確率はレース内で正規化してから logit を取る。"""
    out = feat.merge(odds[["race_id", "horse_no", "odds_win"]], on=["race_id", "horse_no"])
    raw = (1.0 - takeout) / out["odds_win"]
    total = raw.groupby(out["race_id"]).transform("sum")
    q = (raw / total).clip(1e-9, 1 - 1e-9)
    out["market_implied_logit"] = np.log(q / (1 - q))
    return out


def content_hash(df: pd.DataFrame) -> str:
    """gold の決定性検証に使う内容ハッシュ（FE-09 / LK-05）。"""
    buf = df.sort_values(["race_id", "horse_no"]).reset_index(drop=True)
    return hashlib.sha256(
        buf.to_csv(index=False, float_format="%.12g").encode("utf-8")
    ).hexdigest()
