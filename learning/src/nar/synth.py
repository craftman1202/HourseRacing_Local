"""合成データ生成（FX-03 / FX-04）。

Plackett-Luce に従わせる。真の効用 u_i = x_i'β + ε_i（ε は Gumbel）で順位を作れば、
条件付きロジットは理論上 β を回収できるはずで、これが実装の正しさの参照点になる。

実データが1バイトも無い段階でも EDA・特徴量・モデル・評価の全経路を通せるように、
silver 相当のテーブルをそのまま吐く。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRACKS = {10: "盛岡", 18: "浦和", 19: "船橋", 20: "大井", 21: "川崎", 24: "名古屋", 31: "高知"}
BANEI_CODE = 3
DISTANCES = (800, 1200, 1400, 1600, 1800, 2000)
FEATURE_COLS = ("x_speed", "x_form", "x_jockey", "x_draw", "x_rest")


@dataclass
class SynthConfig:
    n_races: int = 3000
    start: str = "2010-01-01"
    end: str = "2025-12-31"
    n_horses: int = 4000
    n_jockeys: int = 180
    min_runners: int = 5
    max_runners: int = 16
    beta: tuple[float, ...] = (1.20, 0.70, 0.45, -0.15, -0.25)
    takeout_win: float = 0.20
    include_banei: bool = False
    # 市場は真の効用にノイズを乗せたもの。ノイズが大きいほど市場は非効率になり、
    # 全馬均等買いの ROI が控除率から上振れする。実測の全馬均等 ROI はほぼ (1-τ) に
    # 一致するので、ここは実データの市場効率性に合わせて小さく取る。
    market_noise: float = 0.12
    # >1 で人気薄が過剰に買われる（favourite-longshot bias）。EDA Q4 の検出力検証用。
    longshot_tilt: float = 1.0
    seed: int = 0
    missing_rate_early: float = 0.35   # 2000年代前半の欠損を模す
    field_names: tuple[str, ...] = field(default=FEATURE_COLS)


def generate(cfg: SynthConfig | None = None) -> dict[str, pd.DataFrame]:
    """entry / race / payout / odds の4テーブルを返す。"""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.to_datetime(
        rng.choice(pd.date_range(cfg.start, cfg.end, freq="D"), size=cfg.n_races)
    ).sort_values()
    codes = list(TRACKS) + ([BANEI_CODE] if cfg.include_banei else [])

    horse_ids = np.array([f"H{i:06d}" for i in range(cfg.n_horses)])
    horse_ability = rng.normal(0, 1, size=cfg.n_horses)
    # 脚質。正なら上がりが掛かる（前傾＝逃げ型）、負なら終いが速い（差し型）。
    horse_closing = rng.normal(0, 0.9, size=cfg.n_horses)
    # 各馬に現役期間を持たせる。これが無いと全馬が全期間に出走し、名寄せ監査
    # （現役期間12年以下）が合成データ側の都合で必ず落ちる。
    span_days = (pd.Timestamp(cfg.end) - pd.Timestamp(cfg.start)).days
    career_days = rng.integers(400, 1800, size=cfg.n_horses)
    debut_offset = rng.integers(0, np.maximum(span_days - career_days, 1))
    debut = pd.to_datetime(cfg.start) + pd.to_timedelta(debut_offset, unit="D")
    retire = debut + pd.to_timedelta(career_days, unit="D")
    horse_birth = debut - pd.to_timedelta(rng.integers(700, 1100, size=cfg.n_horses), unit="D")
    debut_i = debut.astype("int64").to_numpy()
    retire_i = retire.astype("int64").to_numpy()
    jockey_ids = np.array([f"J{i:04d}" for i in range(cfg.n_jockeys)])
    jockey_skill = rng.normal(0, 0.8, size=cfg.n_jockeys)

    entries: list[dict] = []
    races: list[dict] = []
    beta = np.asarray(cfg.beta)
    used_today: set[int] = set()
    seen_race_ids: set[str] = set()
    last_date: pd.Timestamp | None = None

    for i, d in enumerate(dates):
        baba = int(rng.choice(codes))
        race_no = int(rng.integers(1, 13))
        race_id = f"{baba:02d}{d.strftime('%Y%m%d')}{race_no:02d}"
        # race_id は (場, 日付, R番) の関数なので、重複を引いたら捨てる。
        # 重複を許すと粒度監査が合成データの都合で落ち、実データの異常と区別できない。
        if race_id in seen_race_ids:
            continue
        seen_race_ids.add(race_id)
        n = int(rng.integers(cfg.min_runners, cfg.max_runners + 1))
        dist = 200 if baba == BANEI_CODE else int(rng.choice(DISTANCES))
        start_ts = d + pd.Timedelta(minutes=int(600 + race_no * 30))

        # 同一馬が同日に2度走ることは実際には無い。ここを許すと名寄せ監査が
        # 合成データの都合で落ち、実データの異常と区別がつかなくなる。
        if d != last_date:
            used_today, last_date = set(), d
        active = np.flatnonzero((debut_i <= d.value) & (retire_i >= d.value))
        active = active[~np.isin(active, list(used_today))] if used_today else active
        if len(active) < n:
            continue
        h_idx = rng.choice(active, size=n, replace=False)
        used_today.update(h_idx.tolist())
        j_idx = rng.choice(cfg.n_jockeys, size=n, replace=False)

        x = np.column_stack([
            horse_ability[h_idx] + rng.normal(0, 0.35, n),   # x_speed
            rng.normal(0, 1, n),                             # x_form
            jockey_skill[j_idx],                             # x_jockey
            (np.arange(n) - n / 2) / n,                      # x_draw
            rng.normal(0, 1, n),                             # x_rest
        ])
        utility = x @ beta
        gumbel = rng.gumbel(0, 1, n)
        order = np.argsort(-(utility + gumbel))
        finish = np.empty(n, dtype=int)
        finish[order] = np.arange(1, n + 1)

        base_time = dist * 0.062
        time_sec = base_time + (finish - 1) * 0.18 + rng.normal(0, 0.6, n)
        # 上がり3F（秒）。ペースバランス特徴量（h_pace_bal_last3）の材料。
        # 脚質を馬ごとに固定しておくと、過去走の平均が意味を持つ列になり、
        # LK-05（未来汚染）が実際にこの経路を検査できる。
        last3f = 39.0 + horse_closing[h_idx] + (finish - 1) * 0.05 + rng.normal(0, 0.4, n)

        m_util = (utility + rng.normal(0, cfg.market_noise, n)) / cfg.longshot_tilt
        p_market = np.exp(m_util) / np.exp(m_util).sum()
        odds_win = np.round((1 - cfg.takeout_win) / p_market, 1).clip(1.0, 999.9)
        popularity = np.empty(n, dtype=int)
        popularity[np.argsort(odds_win)] = np.arange(1, n + 1)

        races.append({
            "race_id": race_id, "race_date": d, "start_ts": start_ts,
            "baba_code": baba, "race_no": race_no, "distance": dist,
            "surface": "ば" if baba == BANEI_CODE else "ダ",
            "class_level": int(rng.integers(1, 6)),
            # 馬場状態。`最高タイム良馬場` の集計条件になるので、実データと同じく
            # 良が多数で稍重以下も出る分布にする
            "baba_condition": str(rng.choice(["良", "稍重", "重", "不良"],
                                             p=[0.62, 0.20, 0.11, 0.07])),
            "turn": "右" if baba % 2 == 0 else "左",
            "prize_yen": int(rng.lognormal(13.0, 0.5)),
            "n_runners": n,
        })
        for k in range(n):
            entries.append({
                "race_id": race_id, "race_date": d, "start_ts": start_ts,
                "baba_code": baba, "distance": dist,
                "horse_no": k + 1, "waku": k // 2 + 1,
                "horse_sk": horse_ids[h_idx[k]],
                "horse_name": f"馬{h_idx[k]:05d}",
                "birth_date": horse_birth[h_idx[k]],
                "sire_sk": f"S{h_idx[k] % 120:03d}",
                "dam_sk": f"D{h_idx[k] % 900:03d}",
                "jockey_sk": jockey_ids[j_idx[k]],
                "trainer_sk": f"T{h_idx[k] % 300:03d}",
                "finish_pos": int(finish[k]),
                "is_win": int(finish[k] == 1),
                "time_sec": float(time_sec[k]),
                "last3f": float(last3f[k]),
                "odds_win": float(odds_win[k]),
                "popularity": int(popularity[k]),
                "weight_kg": float(rng.normal(460, 30)),
                **{name: float(x[k, c]) for c, name in enumerate(cfg.field_names)},
            })

    entry = pd.DataFrame(entries)
    race = pd.DataFrame(races)

    # 古い年ほど馬体重・血統が欠けるという実データの性質を再現しておく。
    # EDA の第一問（どの年から使えるか）が意味のある答えを返すために必要。
    early = entry["race_date"] < pd.Timestamp("2010-01-01") + pd.DateOffset(years=3)
    mask = early & (rng.random(len(entry)) < cfg.missing_rate_early)
    entry.loc[mask, "weight_kg"] = np.nan

    entry = _declared_records(entry, race)

    payout = _payouts(entry, cfg.takeout_win)
    odds = entry[["race_id", "horse_no", "odds_win", "popularity"]].copy()
    odds["captured_at"] = entry["start_ts"] - pd.Timedelta(minutes=5)
    return {"race": race, "entry": entry, "payout": payout, "odds": odds}


# 累積成績8列を as-of-race で作る。
# 実データで8列すべてが as-of-race だと確定した以上、合成データにも同じ性質の列が
# 無いと、その16特徴量が合成側のテスト（ラベルシャッフル・未来汚染・walk-forward）を
# 一切通らないまま実データに出ていくことになる。
DECLARED_COLUMNS = (
    "騎手成績", "全成績", "ダート左成績", "ダート右成績",
    "当競馬場成績", "うち当距離成績", "最高タイム", "最高タイム良馬場",
)


def _fmt_record(first, second, third, others) -> str:
    return f"{first}-{second}-{third}-{others}"


def _fmt_time(sec: float) -> str:
    return "" if not np.isfinite(sec) else f"{int(sec // 60)}:{sec % 60:04.1f}"


def _declared_records(entry: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    """当該レースを含まない累積成績を、集計範囲ごとに作る。

    範囲は実データで判明した実体に合わせる:
      全成績       … 馬
      当競馬場成績 … 馬 × 競馬場
      うち当距離成績 … 馬 × 競馬場 × 距離
      騎手成績     … 馬 × 騎手
      ダート左/右成績 … 馬（レースの回りに合致した出走のみ加算）
      最高タイム   … 馬 × 競馬場 × 距離 の自己ベスト
    """
    df = entry.sort_values(["horse_sk", "start_ts"]).copy()
    # merge は index を振り直すので使わない。振り直すと最後の sort_index() が
    # 元の行順ではなく「馬×時刻順」を復元し、呼び出し側が位置で列を足したときに
    # 行がずれる（実際にレース日付が入れ替わって鮮度ガードが誤発火した）。
    attrs = [c for c in ("turn", "baba_condition") if c in race.columns]
    if attrs:
        lookup = race.drop_duplicates("race_id").set_index("race_id")
        for c in attrs:
            df[c] = df["race_id"].map(lookup[c])
    if "turn" not in df.columns:
        # 場ごとに回りを固定して割り当てる（実データの性質に合わせる）
        df["turn"] = np.where(df["baba_code"] % 2 == 0, "右", "左")
    if "baba_condition" not in df.columns:
        df["baba_condition"] = "良"

    pos = df["finish_pos"].to_numpy()
    scopes = {
        "全成績": ["horse_sk"],
        "当競馬場成績": ["horse_sk", "baba_code"],
        "うち当距離成績": ["horse_sk", "baba_code", "distance"],
        "騎手成績": ["horse_sk", "jockey_sk"],
    }
    for col, keys in scopes.items():
        g = df.groupby(keys, sort=False)
        prior = {}
        for rank in (1, 2, 3):
            prior[rank] = g["finish_pos"].transform(
                lambda s, r=rank: (s == r).cumsum().shift(1).fillna(0))
        n_prior = g.cumcount()
        others = n_prior - prior[1] - prior[2] - prior[3]
        df[col] = [
            _fmt_record(int(a), int(b), int(c), int(d))
            for a, b, c, d in zip(prior[1], prior[2], prior[3], others)
        ]

    for label, col in (("左", "ダート左成績"), ("右", "ダート右成績")):
        hit = (df["turn"] == label).astype(int)
        won = hit * (pos == 1)
        g = df.groupby("horse_sk", sort=False)
        starts = hit.groupby(df["horse_sk"]).cumsum().shift(1).fillna(0)
        # shift はグループ境界を跨ぐので、グループ内先頭を 0 に戻す
        firsts = g.cumcount() == 0
        starts = starts.mask(firsts, 0)
        wins = won.groupby(df["horse_sk"]).cumsum().shift(1).fillna(0).mask(firsts, 0)
        df[col] = [
            _fmt_record(int(w), 0, 0, int(s - w)) for s, w in zip(starts, wins)
        ]

    keys = ["horse_sk", "baba_code", "distance"]
    df["_prev"] = df.groupby(keys, sort=False)["time_sec"].shift(1)
    best = df.groupby(keys, sort=False)["_prev"].cummin()
    df["最高タイム"] = [_fmt_time(x) for x in best]

    # 良馬場限定は良のレースだけを対象に積む。実ファイルは `良1:33.2` のように
    # 馬場状態ラベルを前置した形式で入っている。
    good = df["baba_condition"].astype(str).str.strip() == "良"
    g = df[good]
    prev_good = g.groupby(keys, sort=False)["time_sec"].shift(1)
    best_good = prev_good.groupby([g[k] for k in keys], sort=False).cummin()
    df["最高タイム良馬場"] = ""
    df.loc[good, "最高タイム良馬場"] = [
        "" if not (v := _fmt_time(x)) else "良" + v for x in best_good]
    # 回りと馬場状態はレース表の属性。entry に残すと実データの silver と列構成が
    # ずれ、下流の join が turn_x / turn_y に化ける。
    helper = [c for c in ("turn", "baba_condition") if c in df.columns
              and c not in entry.columns]
    return df.drop(columns=["_prev", *helper]).sort_index()


def _payouts(entry: pd.DataFrame, takeout: float) -> pd.DataFrame:
    win = entry[entry["finish_pos"] == 1]
    rows = [{
        "race_id": r.race_id, "bet_type": "単勝",
        "comb_1": r.horse_no, "comb_2": None, "comb_3": None,
        "payout_yen": int(round(r.odds_win * 100)),
        "popularity": r.popularity, "dead_heat_seq": 1,
    } for r in win.itertuples()]
    return pd.DataFrame(rows)


def inject_leak_column(entry: pd.DataFrame, col: str = "x_leak") -> pd.DataFrame:
    """FX-04: 着順の完全関数である列を1本混ぜる。リーク検出器の検出力を測るため。"""
    out = entry.copy()
    out[col] = 1.0 / out["finish_pos"].astype(float)
    return out


def shuffle_labels_within_race(entry: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """LK-06: レース内で着順を完全シャッフルする。"""
    rng = np.random.default_rng(seed)
    out = entry.copy()
    for _, idx in out.groupby("race_id").groups.items():
        pos = out.loc[idx, "finish_pos"].to_numpy()
        rng.shuffle(pos)
        out.loc[idx, "finish_pos"] = pos
    out["is_win"] = (out["finish_pos"] == 1).astype(int)
    return out


def poison_future(entry: pd.DataFrame, cutoff: str, seed: int = 0) -> pd.DataFrame:
    """LK-05: 基準日 T 以降のレコードを乱数で破壊する。

    T 以前の特徴量行列がこれで1ビットでも変われば、どこかが未来を見ている。
    """
    rng = np.random.default_rng(seed)
    out = entry.copy()
    future = out["start_ts"] >= pd.Timestamp(cutoff)
    n = int(future.sum())
    if n == 0:
        return out
    idx = out.index[future]
    out.loc[idx, "finish_pos"] = rng.permutation(out.loc[idx, "finish_pos"].to_numpy())
    out.loc[idx, "is_win"] = (out.loc[idx, "finish_pos"] == 1).astype(int)
    out.loc[idx, "time_sec"] = out.loc[idx, "time_sec"] + rng.normal(0, 50, n)
    if "last3f" in out.columns:
        out.loc[idx, "last3f"] = out.loc[idx, "last3f"] + rng.normal(0, 5, n)
    for c in FEATURE_COLS:
        out.loc[idx, c] = rng.normal(0, 10, n)
    drop = rng.choice(idx, size=max(1, n // 20), replace=False)
    return out.drop(index=drop).reset_index(drop=True)
