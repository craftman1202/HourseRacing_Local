"""キー設計。

馬の同定が最大の難所。出馬表には馬 ID が無く馬名しかないため、代理キーと
名寄せテーブルを分けて持つ。この品質が過去成績特徴量の精度を直接決める。
"""

from __future__ import annotations

import hashlib
import logging
import re

import pandas as pd

from ..errors import UnknownTrackError

log = logging.getLogger(__name__)

RACE_ID_RE = re.compile(r"^\d{12}$")

# k_babaCode。月別開催日程ページから自動生成した結果と突き合わせる基準値。
# ここは「現在開催中の場」だけで、廃止16場は含まない。廃止場を含む完全なマスタは
# `nar track-master` が meta/track_master.json に書き出す。TR-04 はこの13コードが
# 自動生成結果に包含されることを要求する。
#
# 注意: 当初この辞書には 水沢=10（正しくは11）、姫路=27（正しくは28）と書いていた。
# 盛岡と水沢、園田と姫路が同一コードになり、同日開催で race_id が衝突する誤りだった。
# 自動生成を基準に据える理由がまさにこれ。
KNOWN_BABA_CODES: dict[str, int] = {
    "帯広ば": 3, "門別": 36, "盛岡": 10, "水沢": 11, "浦和": 18, "船橋": 19,
    "大井": 20, "川崎": 21, "金沢": 22, "笠松": 23, "名古屋": 24, "園田": 27,
    "姫路": 28, "高知": 31, "佐賀": 32,
}

TRACK_MASTER_FILE = "track_master.json"


def make_race_id(baba_code: int, race_date: pd.Timestamp | str, race_no: int) -> str:
    """競馬場コード(2桁) + 競走年月日(8桁) + レース番号(2桁)。"""
    d = pd.Timestamp(race_date)
    return f"{int(baba_code):02d}{d.strftime('%Y%m%d')}{int(race_no):02d}"


def assert_race_id_format(s: pd.Series) -> None:
    bad = s[~s.astype(str).str.match(RACE_ID_RE)]
    if len(bad):
        raise ValueError(f"race_id が12桁数字でない行が {len(bad)} 件: {bad.head().tolist()}")


class TrackMaster:
    """競馬場名 → コード。未知名称は黙って NULL にせず例外（TR-03）。"""

    def __init__(self, mapping: dict[str, int] | None = None) -> None:
        self.mapping = dict(mapping or KNOWN_BABA_CODES)

    @classmethod
    def load(cls, store) -> "TrackMaster":
        """meta/track_master.json があればそれを使う。無ければ既知13場のみ。

        自動生成ファイルは既知コードを包含していなければならない（TR-04）。
        包含していなければマスタ生成が壊れているので止める。
        """
        from ..ingest.schedule import load as load_master

        mapping = load_master(store.path("meta", TRACK_MASTER_FILE))
        if mapping is None:
            log.warning("track_master.json がありません。既知 %d 場のみで動きます。"
                        "廃止場を含む期間は silver に落ちません。", len(KNOWN_BABA_CODES))
            return cls()
        bad = {n: (c, mapping.get(n)) for n, c in KNOWN_BABA_CODES.items()
               if mapping.get(n) != c}
        if bad:
            raise UnknownTrackError(
                f"自動生成マスタが既知コードと食い違います（既知, 生成）: {bad}")
        return cls(mapping)

    def code(self, name: str) -> int:
        key = str(name).strip()
        if key not in self.mapping:
            raise UnknownTrackError(
                f"未知の競馬場名 {key!r}。マスタを更新してから再実行してください。"
                "NULL 埋めすると race_id が衝突します。"
            )
        return self.mapping[key]

    def codes(self, names: pd.Series) -> pd.Series:
        return names.map(self.code)

    def covers(self, names) -> bool:
        return set(map(str.strip, map(str, names))) <= set(self.mapping)


def horse_sk(name: str, birth: str, sire: str) -> str:
    """馬名 + 生年月日 + 父馬名 のハッシュ。

    地方では改名と同名馬が実在するため、馬名単独では同定できない。逆に
    (生年月日, 父) まで一致する別馬は事実上ないので、これで分離側は担保される。
    """
    payload = "\x1f".join(_norm(x) for x in (name, birth, sire))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _norm(x: object) -> str:
    return re.sub(r"\s+", "", str(x)).strip()


def add_horse_sk(df: pd.DataFrame, name="馬名", birth="生年月日", sire="父馬名") -> pd.DataFrame:
    out = df.copy()
    out["horse_sk"] = [
        horse_sk(n, b, s) for n, b, s in zip(out[name], out[birth], out[sire])
    ]
    return out


def build_horse_alias(
    df: pd.DataFrame,
    name="馬名", birth="生年月日", sire="父馬名", dam="母馬名",
    date_col="race_date",
) -> pd.DataFrame:
    """改名の名寄せ。

    (生年月日, 父, 母) が同一で馬名だけが異なるレコード群を改名候補とし、
    最も早く出現した馬名の horse_sk を canonical として時系列で連結する。
    母父まで一致すればほぼ確実に同一馬だが、母父が無い年があるので母までで打つ。
    """
    cols = [birth, sire, dam]
    work = df[[name, date_col, "horse_sk", *cols]].copy()
    first_seen = work.groupby([*cols, name], as_index=False)[date_col].min()
    first_seen = first_seen.sort_values([*cols, date_col, name])

    canon = (
        first_seen.groupby(cols, as_index=False)
        .first()
        .rename(columns={name: "canonical_name", date_col: "canonical_first_seen"})
    )
    merged = first_seen.merge(canon, on=cols, how="left")
    sk_of = {
        (r[birth], r[sire], r[name]): r["horse_sk"]
        for _, r in work.drop_duplicates([birth, sire, name]).iterrows()
    }
    merged["horse_sk"] = [
        sk_of.get((r[birth], r[sire], r[name])) for _, r in merged.iterrows()
    ]
    merged["canonical_sk"] = [
        sk_of.get((r[birth], r[sire], r["canonical_name"])) for _, r in merged.iterrows()
    ]
    merged["is_rename"] = merged[name] != merged["canonical_name"]
    return merged[[
        "horse_sk", "canonical_sk", name, "canonical_name", birth, sire, dam,
        date_col, "is_rename",
    ]].rename(columns={name: "alias_name", date_col: "first_seen"})


def person_keys(df: pd.DataFrame, name_col: str, affil_col: str, prefix: str) -> pd.DataFrame:
    """騎手・調教師。移籍があるので氏名単位と氏名+所属単位の両方を持つ。

    特徴量では氏名単位（前者）を主に使う。所属変更で履歴が切れるほうが害が大きい。
    """
    out = df.copy()
    out[f"{prefix}_sk"] = out[name_col].map(_norm)
    out[f"{prefix}_affil_sk"] = out[name_col].map(_norm) + "@" + out[affil_col].map(_norm)
    return out
