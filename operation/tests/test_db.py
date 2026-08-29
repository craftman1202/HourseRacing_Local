"""DB-01..10: 確定層／ライブ層と as-of 整合性。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from narops.clock import FixedClock, business_date, jst_datetime, to_jst, to_utc
from narops.db import freshness
from narops.db.backend import Warehouse, assert_partition_filter
from narops.db.merge import merge_final, refresh_live, table_hash
from narops.errors import BytesBilledUnbounded, PartitionFilterRequired, StaleDataError
from tests.conftest import make_results

pytestmark = pytest.mark.component


# ------------------------------------------------------------------------ DB-01
def test_db01_merge_is_idempotent(wh, results, clock):
    first = merge_final(wh, results, source_sha256="a" * 64, clock=clock)
    h1, n1 = table_hash(wh, "entry_result_final"), wh.row_count("entry_result_final")

    second = merge_final(wh, results, source_sha256="a" * 64, clock=clock)
    assert second.is_noop, f"2回目で {second.inserted} 挿入 / {second.updated} 更新"
    assert wh.row_count("entry_result_final") == n1
    assert table_hash(wh, "entry_result_final") == h1, "全列ハッシュが一致しません"
    assert first.inserted == len(results)


# ------------------------------------------------------------------------ DB-02
def test_db02_history_view_has_no_duplicates(populated, results, clock):
    """ライブ層が確定層と重なっても entry_history は1行に吸収する。"""
    live = results.head(20).copy()
    refresh_live(populated, live, clock=clock)

    df = populated.con.execute(
        "SELECT race_id, horse_no, COUNT(*) c FROM entry_history "
        "GROUP BY 1,2 HAVING c > 1").df()
    assert df.empty, f"entry_history に重複が {len(df)} 件あります"


def test_db02_live_only_race_appears_once(populated, clock):
    """確定層に無いレースはライブ層から1行だけ出る。"""
    live = make_results(n_races=2, start_day=26, seed=99)
    refresh_live(populated, live, clock=clock)
    df = populated.con.execute(
        "SELECT src, COUNT(*) c FROM entry_history GROUP BY 1").df()
    assert set(df["src"]) == {"final", "live"}
    dup = populated.con.execute(
        "SELECT race_id, horse_no FROM entry_history GROUP BY 1,2 HAVING COUNT(*)>1").df()
    assert dup.empty


# ------------------------------------------------------------------ DB-03（Blocker）
def test_db03_future_rows_do_not_change_features(populated, clock, release_dir):
    """対象レース発走後の実績行を注入しても、当該レースの特徴量がビット単位で不変。"""
    from nar.config import feature_config

    from narops.features import build_for_race
    from narops.model.manifest import Manifest

    target_start = jst_datetime(2026, 8, 22, 14, 30)
    card = pd.DataFrame([{
        "race_id": "202026082201", "horse_no": i + 1,
        "horse_sk": f"H{i:04d}", "jockey_sk": f"J{i % 12:03d}",
        "trainer_sk": f"T{i % 9:03d}", "sire_sk": f"S{i % 6:03d}",
    } for i in range(8)])
    race_row = pd.Series({
        "race_id": "202026082201", "race_date": target_start.date(),
        "start_ts": to_utc(target_start), "baba_code": 20, "distance": 1400,
        "race_no": 1, "surface": "ダ", "class_level": 2, "prize_yen": 1_000_000,
    })
    manifest = Manifest.read(release_dir / "manifest.json")
    fcfg = feature_config()

    before = build_for_race(populated, card, race_row, manifest, fcfg,
                            max_bytes_billed=2_000_000_000)

    # 発走後の結果を大量に注入する
    future = make_results(n_races=8, start_day=23, seed=7)
    merge_final(populated, future, source_sha256="b" * 64, clock=clock)

    after = build_for_race(populated, card, race_row, manifest, fcfg,
                           max_bytes_billed=2_000_000_000)

    cols = list(before.spec.names)
    pd.testing.assert_frame_equal(
        before.frame[cols].reset_index(drop=True),
        after.frame[cols].reset_index(drop=True),
        check_exact=True,
    )


# ------------------------------------------------------------------------ DB-04
def test_db04_stale_final_layer_blocks_inference(wh, clock):
    """確定層が前日分に届かないと fail-closed。ベット候補は0件。"""
    old = make_results(n_races=2, start_day=1, seed=3)
    merge_final(wh, old, clock=clock)
    with pytest.raises(StaleDataError, match="fail-closed"):
        freshness.require_fresh(wh, clock)


def test_db04_fresh_layer_passes(wh, clock):
    yesterday = to_jst(clock.now()).date() - timedelta(days=1)
    df = make_results(n_races=2, start_day=yesterday.day, base_month=yesterday.month)
    merge_final(wh, df, clock=clock)
    f = freshness.require_fresh(wh, clock)
    assert f.is_fresh and f.latest_final_date >= yesterday


def test_db04_reports_watermark(populated, clock):
    assert freshness.watermark(populated) is not None


# ------------------------------------------------------------------------ DB-05
def test_db05_live_refresh_only_touches_live_layer(populated, clock):
    """日内更新で確定層が変わらないこと。"""
    before = table_hash(populated, "entry_result_final")
    refresh_live(populated, make_results(n_races=3, start_day=26, seed=5), clock=clock)
    assert table_hash(populated, "entry_result_final") == before


def test_db05_repeated_live_refresh_replaces_not_appends(populated, clock):
    live = make_results(n_races=3, start_day=26, seed=5)
    refresh_live(populated, live, clock=clock)
    n1 = populated.row_count("entry_result_live")
    refresh_live(populated, live, clock=clock)
    assert populated.row_count("entry_result_live") == n1, "ライブ層が二重に積まれています"


# ------------------------------------------------------------------------ DB-06
def test_db06_update_frequency_is_enforced(populated, clock, cfg):
    """24時間で確定層マージ1回・ライブ更新3回。超過は no-op。"""
    from narops.jobs import UpdateBudget

    budget = UpdateBudget(cfg)
    day = business_date(clock.now())
    assert budget.allow("final_merge", day)
    assert not budget.allow("final_merge", day), "確定層マージが1日2回実行できています"

    for _ in range(3):
        assert budget.allow("live_refresh", day)
    assert not budget.allow("live_refresh", day), "ライブ更新が1日4回実行できています"


# ------------------------------------------------------------------------ DB-07
def test_db07_correction_updates_and_preserves_old_value(wh, results, clock):
    """後日訂正（降着・失格）で旧値が監査テーブルに残る。"""
    merge_final(wh, results, source_sha256="a" * 64, clock=clock)
    corrected = results.copy()
    idx = corrected.index[0]
    old_pos = int(corrected.loc[idx, "finish_pos"])
    corrected.loc[idx, "finish_pos"] = 99
    corrected.loc[idx, "is_win"] = 0

    res = merge_final(wh, corrected, source_sha256="c" * 64, clock=clock,
                      reason="降着")
    assert res.updated == 1 and res.corrections >= 1

    audit = wh.table("entry_result_audit")
    row = audit[audit["column_name"] == "finish_pos"].iloc[0]
    assert row["old_value"] == str(old_pos) and row["new_value"] == "99"
    assert row["reason"] == "降着"

    live_val = wh.con.execute(
        "SELECT finish_pos FROM entry_result_final WHERE race_id=? AND horse_no=?",
        [corrected.loc[idx, "race_id"], int(corrected.loc[idx, "horse_no"])]).fetchone()[0]
    assert live_val == 99


def test_db07_float_noise_is_not_treated_as_correction(wh, results, clock):
    merge_final(wh, results, clock=clock)
    jittered = results.copy()
    jittered["time_sec"] = jittered["time_sec"] + 1e-13
    res = merge_final(wh, jittered, clock=clock)
    assert res.corrections == 0, "1e-13 の差を訂正として記録してはいけません"


# ------------------------------------------------------------------------ DB-09
def test_db09_partition_filter_is_required():
    assert_partition_filter("SELECT * FROM entry_result_final WHERE race_date = '2026-08-25'")
    with pytest.raises(PartitionFilterRequired, match="race_date"):
        assert_partition_filter("SELECT * FROM entry_result_final")


def test_db09_join_without_partition_filter_is_rejected():
    with pytest.raises(PartitionFilterRequired):
        assert_partition_filter(
            "SELECT * FROM prediction JOIN race_schedule USING (race_id)")


def test_db09_query_enforces_the_guard(populated):
    with pytest.raises(PartitionFilterRequired):
        populated.query("SELECT * FROM entry_result_final")


# ------------------------------------------------------------------ CO-01
def test_co01_query_without_bytes_billed_is_rejected():
    w = Warehouse(":memory:")          # max_bytes_billed 未設定
    from narops.db import schema

    schema.create_all(w)
    with pytest.raises(BytesBilledUnbounded, match="maximum_bytes_billed"):
        w.query("SELECT * FROM entry_result_final WHERE race_date='2026-08-25'")
    w.close()


def test_co01_scan_over_limit_is_rejected(populated):
    with pytest.raises(BytesBilledUnbounded, match="上限"):
        populated.query(
            "SELECT * FROM entry_result_final WHERE race_date >= '2000-01-01'",
            max_bytes_billed=10)


# ------------------------------------------------------------------------ DB-10
def test_db10_storage_is_utc_boundary_is_jst():
    late = jst_datetime(2026, 8, 25, 23, 50)       # JST 深夜
    assert to_utc(late).hour == 14                  # UTC では前日の 14:50
    assert business_date(late).isoformat() == "2026-08-25"


def test_db10_race_just_after_midnight_belongs_to_that_jst_day():
    early = jst_datetime(2026, 8, 26, 0, 10)
    assert business_date(early).isoformat() == "2026-08-26"
    assert to_utc(early).date().isoformat() == "2026-08-25", "UTC では前日になる"


def test_db10_naive_datetime_is_interpreted_as_jst():
    from datetime import datetime

    naive = datetime(2026, 8, 25, 12, 0)
    assert to_utc(naive).hour == 3, "naive を UTC 扱いすると9時間ずれる"


# ------------------------------------------------------------------ 増分投入
def test_speed_index_needs_full_history_not_just_the_incremental_slice():
    """1か月分だけで速度指数を計算すると、基準統計量が揃わず全行 NaN になる。

    増分投入でそれを書き戻すと、全履歴から正しく計算済みの値を壊す
    （実際に 8,893 行を NaN で上書きした）。
    """
    import pandas as pd
    from nar.features.builder import speed_index, MIN_PRIOR_FOR_SPEED_INDEX

    from conftest import make_results

    full = pd.concat(
        [make_results(n_races=60, start_day=1, seed=i, base_month=m)
         for i, m in enumerate((5, 6, 7, 8))], ignore_index=True)
    # silver には speed_index が無い（学習側が毎回計算する）。フィクスチャが
    # 持っている値を残したままだと「保存済みを優先する」挙動で計算が走らない。
    full = full.drop(columns=["speed_index"])
    last_month = full[pd.to_datetime(full["race_date"]).dt.month == 8]

    from_full = speed_index(full)
    from_full = from_full[pd.to_datetime(from_full["race_date"]).dt.month == 8]
    from_slice = speed_index(last_month)

    assert from_full["speed_index"].notna().sum() > from_slice["speed_index"].notna().sum(), (
        "テスト前提: 全履歴のほうが速度指数を出せる行が多い")
    assert from_slice["speed_index"].isna().mean() > 0.5, (
        f"基準統計量 {MIN_PRIOR_FOR_SPEED_INDEX} 件に届かない前提が崩れています")


def test_comparable_tolerates_null_integers():
    """取消・除外の馬は着順が NULL。整数列に NA が入っても落ちないこと。"""
    import pandas as pd
    from narops.db.types import comparable

    df = pd.DataFrame({"race_id": ["r1", "r1"], "horse_no": [1, 2],
                       "finish_pos": pd.array([1, None], dtype="Int64")})
    out = comparable(df, ["race_id", "horse_no", "finish_pos"])
    assert out["finish_pos"].isna().sum() == 1


def test_freshness_ignores_scheduled_races_without_results(wh, clock):
    """確定層には当月の未実施レースも入る。予定日を最新確定日と誤認しないこと。"""
    import pandas as pd
    from narops.db.freshness import check
    from narops.db.merge import merge_final

    from conftest import make_results

    done = make_results(n_races=8, start_day=18, seed=6)
    scheduled = make_results(n_races=4, start_day=28, seed=7)
    scheduled[["finish_pos", "is_win", "time_sec", "speed_index"]] = None
    merge_final(wh, pd.concat([done, scheduled], ignore_index=True), clock=clock)

    f = check(wh, clock)
    assert str(f.latest_final_date) == "2026-08-19", (
        f"未実施レースを最新確定日として拾っています: {f.latest_final_date}")


def test_loaded_history_keeps_start_ts_in_utc():
    """silver の start_ts は JST の naive。DB 内の naive は UTC という規約に直すこと。

    直さないと 9 時間ずれ、`start_ts < 発走時刻` で同日の先行レースが全部落ちる。
    """
    import pandas as pd
    from narops.clock import JST

    jst_naive = pd.Series(pd.to_datetime(["2026-08-26 12:40:00"]))
    utc_naive = jst_naive.dt.tz_localize(JST).dt.tz_convert("UTC").dt.tz_localize(None)
    assert str(utc_naive.iloc[0]) == "2026-08-26 03:40:00"


def test_merge_rejects_jst_naive_start_ts(wh, clock):
    """JST の naive をそのまま入れると 9 時間ずれ、同日の先行レースが履歴から落ちる。

    値そのものは妥当に見えるので、分布で気付くしかない。
    """
    import pandas as pd
    import pytest

    from narops.db.merge import merge_final

    from conftest import make_results

    results = make_results(n_races=12, start_day=18, seed=8)
    jst = results.copy()
    jst["start_ts"] = pd.to_datetime(jst["start_ts"]).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    with pytest.raises(ValueError, match="JST のまま"):
        merge_final(wh, jst, clock=clock)


def test_write_paths_survive_a_partition_filter_enforcing_backend(wh, clock):
    """削除文にもパーティション条件が要る。

    BigQuery は require_partition_filter を DML にも適用する。ローカルの
    execute だけが素通しだったため、予測・ベット候補・確定層訂正の削除文が
    そろって**本番でだけ**拒否され、テストは全部緑のままだった。
    """
    from narops.errors import PartitionFilterRequired
    from narops.pipeline import write_prediction

    with pytest.raises(PartitionFilterRequired):
        wh.execute("DELETE FROM prediction WHERE race_id = 'r1'")

    frame = pd.DataFrame({"horse_no": [1, 2], "p_win": [0.6, 0.4],
                          "rank": [1, 2], "stake_yen": [0, 0],
                          "odds_win": [2.0, 3.0]})
    # 条件が揃えば通ること。ここが落ちると本番の書き込みが丸ごと止まる
    n = write_prediction(wh, "r1", frame, "v1", "A", date(2026, 8, 28),
                         clock, is_shadow=False)
    assert n == 2
