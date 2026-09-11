"""NAR からの当日データ取得。

HTTP クライアントは学習側の `NarClient` を使う（レート制限3秒・リトライ・
Content-Type 検証がそこに入っている）。運用側で `httpx` を直接叩かないのは、
アクセス間隔の保証を1か所に閉じ込めるため。

取得失敗時は **fail-closed**。部分的に取れた不完全データで推論するのは禁止で、
当日をスキップして告知するのが正しい（DR-01 / ランブック）。
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import httpx
import pandas as pd
from nar.transform.keys import KNOWN_BABA_CODES

from .clock import JST, to_utc
from . import deba_table, results_table
from .errors import InsufficientData
from .shared import to_prerace

log = logging.getLogger(__name__)

# NAR は 404 でも HTTP 200 を返し、本文にこれが入る
_ERROR_PAGE_RE = re.compile(r"404\s*not\s*Found|お探しのページは見つかりません",
                            re.IGNORECASE)
TODAY_TOP = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/TodayRaceInfoTop"
# 当日メニュー（レース番号・発走時刻・天候の一覧）。出走馬は載っていない
RACE_LIST = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceList"
# 出馬表。当初 RACE_LIST を出馬表として叩いていたが、あれは目次であって
# 出走馬は1頭も入っていない
DEBA_TABLE = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable"
# 単勝・複勝オッズ。NAR に OddsWin というページは無い（存在しない名前を
# 叩いていたので、全レースがエラーページになっていた）。出馬表ページ内の
# リンクが指す実在の名前はこれ。
ODDS_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/OddsTanFuku"
# 当日成績表。ライブ層を埋める唯一の経路（確定層は翌日以降）
RESULT_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/RaceMarkTable"

# 競馬場名 → k_babaCode。学習側の TrackMaster（`nar.transform.keys`）から直接取る。
#
# 以前はここに同じ辞書を手書きで複製しており、学習側で 水沢=10→11、姫路=27→28 の
# 誤りが修正された（`transform/keys.py::KNOWN_BABA_CODES` のコメント参照）ときに
# 運用側だけ古い値のまま取り残された。結果、水沢の当日レースが盛岡と同じ
# baba_code=10 で処理され、速度指数の基準統計量（場×距離で算出）が盛岡のものに
# すり替わったまま推論が回っていた（2026-09、水沢の推論停止として発覚）。
#
# 2つの辞書を維持する限りこの種の乖離は必ず再発する。学習側を唯一の実装にする
# （`shared.py::track_names()` が逆写像で既に同じ方針を取っている）。
BABA_CODE = KNOWN_BABA_CODES


class NarFetchError(Exception):
    """取得・解析の失敗。当日をスキップする理由になる。"""


@dataclass
class NarSource:
    """当日情報の取得。

    HTML 構造の変化は検知して**当日の計画を中止**する。部分結果で走らせない（DR-02）。
    """

    user_agent: str = "nar-ops/1.0 (research; contact: integral.sekibundx@gmail.com)"
    min_interval_sec: float = 3.0
    timeout_sec: float = 60.0
    transport: httpx.BaseTransport | None = None
    _client: object | None = field(default=None, repr=False)

    def __enter__(self) -> "NarSource":
        from nar.ingest.client import NarClient

        self._client = NarClient(
            user_agent=self.user_agent, min_interval_sec=self.min_interval_sec,
            timeout_sec=self.timeout_sec, transport=self.transport,
            expect_content_type="text/html")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._client is not None:
            self._client.close()

    def _get(self, url: str, params: dict) -> str:
        """HTML を取得して文字列にする。

        **当日情報の HTML は UTF-8**、月次 ZIP 内の CSV は CP932。同じサイトでも
        別物なので分けて扱う。cp932 は UTF-8 バイト列も例外なく「復号」してしまい
        （文字化けするだけ）、エラーで気付けないため、順序が重要。
        """
        if self._client is None:
            raise RuntimeError("with 文の中で使ってください")
        raw = self._client.fetch(url, params).content
        text = None
        for codec in ("utf-8", "cp932"):
            try:
                text = raw.decode(codec)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = raw.decode("utf-8", errors="replace")
        # NAR は存在しないページにも 200 でエラー HTML を返す。そのまま下流へ
        # 渡すと「解析できない」という別の顔をした失敗になり、原因が読めない。
        if _ERROR_PAGE_RE.search(text):
            raise NarFetchError(
                f"NAR がエラーページを返しました（{url} params={params}）。"
                "レース番号や日付が存在しないか、まだ公開されていません。")
        return text

    # ------------------------------------------------------------------ 予定
    def fetch_schedule(self, day: date) -> pd.DataFrame:
        """当日の開催場・レース番号・発走時刻。

        抽出できなければ例外。空の DataFrame を返して「開催なし」と誤認させない。
        """
        html = self._get(TODAY_TOP, {"k_raceDate": day.strftime("%Y/%m/%d")})
        rows = parse_schedule(html, day)
        if not rows:
            raise NarFetchError(
                f"{day} のスケジュールを抽出できませんでした。HTML 構造が変わった可能性が"
                "あります。当日の計画を中止します（DR-02）。")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ 出馬表
    def fetch_entry_card(self, race_id: str, baba_code: int, day: date,
                         race_no: int) -> pd.DataFrame:
        """当日の出馬表。

        月次ファイルと同じ列名・同じ値が返ることを確認済み（同一レースで
        馬名・生年月日・父・騎手・調教師・累積成績8列・最高タイムが完全一致）。
        学習時と推論時で同じ特徴量が作れる。
        """
        html = self._get(DEBA_TABLE, {"k_raceDate": day.strftime("%Y/%m/%d"),
                                      "k_babaCode": baba_code, "k_raceNo": race_no})
        card = deba_table.parse(html, race_id, day)

        # レース条件（距離・クラス・賞金）を同じ HTML から読む。別途取りに
        # 行くと1レースあたりの取得が倍になる。読めなければ失敗させる。
        # 固定値に落とすと、モデルが学習中に一度も見ていない値を渡すことになる。
        header = deba_table.parse_race_header(html)
        if "class_level" not in header or "distance" not in header:
            raise InsufficientData(
                f"{race_id}: レース条件（距離・クラス）を読めませんでした。"
                "固定値では推論しません。")
        for key in ("distance", "class_level", "class_name", "prize_yen", "surface",
                    "turn", "baba_condition"):
            if key in header:
                card[key] = header[key]

        # 結果列が混じっていたら物理的に落とす。学習時と同一の関数を通す
        return to_prerace(card)

    # ------------------------------------------------------------------ オッズ
    def fetch_odds(self, race_id: str, baba_code: int, day: date,
                   race_no: int) -> pd.DataFrame:
        html = self._get(ODDS_URL, {"k_raceDate": day.strftime("%Y/%m/%d"),
                                    "k_babaCode": baba_code, "k_raceNo": race_no})
        return parse_win_odds(html, race_id)

    # ------------------------------------------------------------ 当日の結果
    def fetch_results(self, race_id: str, baba_code: int, day: date,
                      race_no: int) -> pd.DataFrame:
        """確定した当日レースの着順。ライブ層（`entry_result_live`）用。

        代理キーは出馬表から結合する。成績表には父も生年月日も載っておらず、
        `horse_sk` をここで組み立てると同じ馬に別キーが振られ、過去成績が
        1件も引けなくなる（SK-02）。
        """
        html = self._get(RESULT_URL, {"k_raceDate": day.strftime("%Y/%m/%d"),
                                      "k_babaCode": baba_code, "k_raceNo": race_no})
        outcome = results_table.parse(html)
        if outcome.empty:
            return outcome            # 未確定。正常系

        card = self.fetch_entry_card(race_id, baba_code, day, race_no)
        keys = ["race_id", "horse_no", "horse_sk", "jockey_sk", "trainer_sk",
                "sire_sk", "distance"]
        merged = card[[c for c in keys if c in card.columns]].merge(
            outcome, on="horse_no", how="inner")
        merged["race_date"] = day
        merged["baba_code"] = int(baba_code)
        return merged


# ---------------------------------------------------------------------- 解析
_TIME_RE = re.compile(r"(\d{1,2})[:：](\d{2})")
_RACE_NO_RE = re.compile(r"(\d{1,2})\s*R")


# レース枠を1つ消費する状態表示。発走済み・取消のレースは時刻の代わりに
# これが出るので、**レース番号は進める**。読み飛ばすと以降の番号が全部ずれ、
# 別レースの race_id で推論することになる。
_SLOT_STATES = frozenset({"成績", "確定前", "発売中", "締切", "取消", "中止", "除外"})
# 格付けラベル。枠を消費しない装飾
_GRADE_LABELS = frozenset({"特別", "重賞", "認定", "交流"})
_END_LABELS = frozenset({"払戻金", "月別開催日程"})


def parse_schedule(html: str, day: date) -> list[dict]:
    """開催一覧の抽出。

    HTML の実構造は「競馬場名 → 発走時刻がレース番号順に並ぶ」形で、間に
    `特別` `重賞` などのラベルが挟まる。列見出しの `1R..12R` はページ先頭に
    1度だけ出るので、レース番号は**時刻の出現順**から振る。

    NAR は HTML を予告なく変えるので、抽出できなければ空を返し、呼び出し側が
    当日を中止する（DR-02）。壊れた構造から部分的に拾って走らせない。
    """
    text = re.sub(r"<[^>]+>", "\n", html)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    out: list[dict] = []
    current: str | None = None
    race_no = 0
    for line in lines:
        if line in BABA_CODE:
            current, race_no = line, 0
            continue
        if current is None:
            continue
        if line in _END_LABELS:
            current = None
            continue
        if line in _GRADE_LABELS:
            continue
        if line in _SLOT_STATES:
            # 発走済み・取消。枠は消費するが推論対象にはしない
            race_no += 1
            continue

        m = _TIME_RE.fullmatch(line)
        if not m:
            continue
        hh, mm = int(m.group(1)), int(m.group(2))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            continue
        race_no += 1
        baba_code = BABA_CODE[current]
        start = datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST)
        out.append({
            "race_id": f"{baba_code:02d}{day.strftime('%Y%m%d')}{race_no:02d}",
            "race_date": day, "baba_code": baba_code, "race_no": race_no,
            "track_name": current, "start_ts": to_utc(start), "status": "scheduled",
        })

    seen: dict[str, dict] = {}
    for r in out:
        seen.setdefault(r["race_id"], r)
    return sorted(seen.values(), key=lambda r: (r["baba_code"], r["race_no"]))


def parse_entry_card(html: str, race_id: str) -> pd.DataFrame:
    """出馬表。馬番・馬名・騎手名・枠番を取る。

    実運用では NAR の当日 ZIP を使うほうが確実だが、HTML からも取れるようにして
    ZIP 未公開の時間帯に備える。
    """
    try:
        # 生の文字列を渡すと、新しい pandas はファイルパスとして開こうとして
        # FileNotFoundError になる（本文がそのままエラーに載る）。
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame()

    for t in tables:
        cols = {str(c) for c in t.columns}
        if {"馬番"} <= cols or any("馬番" in str(c) for c in t.columns):
            df = t.copy()
            df.columns = [str(c) for c in df.columns]
            rename = {}
            for c in df.columns:
                if "馬番" in c:
                    rename[c] = "horse_no"
                elif "馬名" in c:
                    rename[c] = "horse_name"
                elif "騎手" in c:
                    rename[c] = "jockey_name"
                elif "枠" in c:
                    rename[c] = "waku"
            df = df.rename(columns=rename)
            if "horse_no" not in df.columns:
                continue
            df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce")
            df = df.dropna(subset=["horse_no"])
            df["horse_no"] = df["horse_no"].astype(int)
            df.insert(0, "race_id", race_id)
            return df.reset_index(drop=True)
    return pd.DataFrame()


def parse_win_odds(html: str, race_id: str) -> pd.DataFrame:
    """単勝オッズ。取れなければ空を返す（トラックA 単独へ縮退する）。"""
    try:
        # 生の文字列を渡すと、新しい pandas はファイルパスとして開こうとして
        # FileNotFoundError になる（本文がそのままエラーに載る）。
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return pd.DataFrame(columns=["race_id", "horse_no", "odds_win"])

    for t in tables:
        cols = [str(c) for c in t.columns]
        if not any("馬番" in c for c in cols):
            continue
        odds_col = next((c for c in cols if "単勝" in c or "オッズ" in c), None)
        no_col = next(c for c in cols if "馬番" in c)
        if odds_col is None:
            continue
        df = t.copy()
        df.columns = cols
        out = pd.DataFrame({
            "race_id": race_id,
            "horse_no": pd.to_numeric(df[no_col], errors="coerce"),
            "odds_win": pd.to_numeric(
                df[odds_col].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce"),
        }).dropna()
        out["horse_no"] = out["horse_no"].astype(int)
        # 0 以下のオッズは存在しない。ゼロ埋めの痕跡として弾く（IN-05）
        return out[out["odds_win"] > 0].reset_index(drop=True)
    return pd.DataFrame(columns=["race_id", "horse_no", "odds_win"])
