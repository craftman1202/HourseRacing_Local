"""当日出馬表（DebaTable）の解析。

当初 `TodayRaceInfo/RaceList` を出馬表として叩いていたが、あれは当日メニュー
（レース番号・発走時刻・天候の一覧）で、出走馬は1頭も載っていない。実際の出馬表は
`TodayRaceInfo/DebaTable?k_raceDate=&k_raceNo=&k_babaCode=`。

DebaTable は1頭を5行の `<tr>` で表す（rowspan で列を跨ぐ）ため、pd.read_html では
列が崩れる。行の役割が固定なので、その構造をそのまま読む。

  1行目: 枠番 / 馬番 / 馬名 / 騎手 / オッズ・人気 / 着別成績5種 + 最高タイム
  2行目: 性齢 / 毛色 / 生年月日(MM.DD生) / 負担重量
  3行目: 父馬名 / 調教師 / 馬体重(増減)
  4行目: 母馬名 / 馬主
  5行目: （母父馬名）/ 生産牧場

学習側の `horse_sk` は 馬名 + 生年月日(YYYYMMDD) + 父馬名 のハッシュ。生年は
ページに無いが、日本の競走馬の年齢は「開催年 − 生年」なので、性齢の年齢から
一意に復元できる。ここが合わないと過去成績と結合できず、全特徴量が初出走扱いに
なるので、復元できない行は捨てずに例外にする。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup

from .errors import InsufficientData

DEBA_URL = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable"

# 着別成績のラベル → 月次ファイルの列名。学習時と同じ名前に揃える
_RECORD_LABELS = {
    "全": "全成績", "左": "ダート左成績", "右": "ダート右成績",
    "場": "当競馬場成績", "距": "うち当距離成績",
}
# 性別表記は 牡/牝/セン（騸馬）。「セン」を1文字クラスで書くと「ン」が漏れる
_SEX_AGE_RE = re.compile(r"([^\d\s]+)\s*(\d+)")
_BIRTH_RE = re.compile(r"(\d{1,2})[.．](\d{1,2})\s*生")
_WEIGHT_RE = re.compile(r"(\d{3})\s*\(([-+±]?\d+)\)")
_ODDS_RE = re.compile(r"([\d,]+\.\d+)")
_POP_RE = re.compile(r"\((\d+)人気\)")
_TIME_RE = re.compile(r"^(?:良|稍重|重|不良)?\d+:\d{2}\.\d$")


@dataclass
class HorseRow:
    values: dict


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def _cells(tr) -> list[str]:
    return [_text(td) for td in tr.select("td")]


def _records(result_td) -> dict:
    """着別成績5種と最高タイム2種。

    `全 0- 2- 1- 5` のように区切りが独立セルに入るので、行ごとに結合して
    月次ファイルと同じ `0-2-1-5` 形式に直す。
    """
    out: dict[str, str] = {}
    for tr in result_td.select("table.arrival tr"):
        cells = [_text(td) for td in tr.select("td")]
        if not cells:
            continue
        head = cells[0].strip()
        if head in _RECORD_LABELS:
            digits = "".join(cells[1:]).replace(" ", "")
            out[_RECORD_LABELS[head]] = digits
        else:
            times = [c for c in cells if _TIME_RE.match(c.strip())]
            for t in times:
                key = "最高タイム良馬場" if t[0] not in "0123456789" else "最高タイム"
                out.setdefault(key, t.strip())
    return out


def _birth_ymd(sex_age: str, birth_md: str, race_year: int) -> str:
    """`牡3` と `04.01生` と開催年から YYYYMMDD を作る。

    日本の競走馬の年齢は満年齢ではなく「開催年 − 生年」。したがって生年は
    一意に決まる。
    """
    m_age = _SEX_AGE_RE.search(sex_age)
    m_md = _BIRTH_RE.search(birth_md)
    if not m_age or not m_md:
        raise InsufficientData(
            f"性齢 {sex_age!r} と生年月日 {birth_md!r} から生年を復元できません。"
            "馬の同定に失敗すると過去成績が全部欠けます。")
    year = race_year - int(m_age.group(2))
    return f"{year:04d}{int(m_md.group(1)):02d}{int(m_md.group(2)):02d}"


def parse(html: str, race_id: str, race_date: pd.Timestamp | str) -> pd.DataFrame:
    """出馬表を1頭1行に直す。"""
    soup = BeautifulSoup(html, "html.parser")
    heads = soup.select("tr.tBorder")
    if not heads:
        raise InsufficientData(
            f"{race_id}: 出馬表に出走馬の行がありません（未確定か HTML 構造の変更）。")

    race_year = pd.Timestamp(race_date).year
    rows: list[dict] = []
    # 枠番は1枠に複数頭が入ると rowspan で束ねられ、2頭目以降のセルが無い。
    # 取りこぼすと相対枠順（draw_rel）が欠ける。
    last_waku: int | None = None
    for tr in heads:
        rest = tr.find_next_siblings("tr", limit=4)
        c2, c3, c4, c5 = ([_cells(x) for x in rest] + [[]] * 4)[:4]

        waku = tr.select_one("td.courseNum")
        no = tr.select_one("td.horseNum")
        name = tr.select_one("a.horseName")
        jockey = tr.select_one("a.jockeyName")
        odds_td = tr.select_one("td.odds_weight")
        result_td = tr.select_one("td.result")
        if no is None or name is None:
            continue

        odds_text = _text(odds_td)
        m_odds, m_pop = _ODDS_RE.search(odds_text), _POP_RE.search(odds_text)
        weight = _WEIGHT_RE.search(" ".join(c3))

        if waku is not None and _text(waku).strip():
            last_waku = int(re.sub(r"\D", "", _text(waku)) or 0)

        lineage = name.get("href", "")
        m_lin = re.search(r"k_lineageLoginCode=(\d+)", lineage)

        row = {
            "race_id": race_id,
            "horse_no": int(re.sub(r"\D", "", _text(no)) or 0),
            "waku": last_waku,
            "horse_name": _text(name),
            # NAR が振っている血統登録番号。名前より強い同定子なので保持する
            "lineage_code": m_lin.group(1) if m_lin else "",
            "jockey_name": re.sub(r"（.*?）", "", _text(jockey)).strip() if jockey else "",
            "jockey_affil": (re.search(r"（(.*?)）", _text(jockey)).group(1)
                             if jockey and "（" in _text(jockey) else ""),
            "sex": (_SEX_AGE_RE.search(c2[0]).group(1) if c2 and _SEX_AGE_RE.search(c2[0])
                    else ""),
            "age": (int(_SEX_AGE_RE.search(c2[0]).group(2))
                    if c2 and _SEX_AGE_RE.search(c2[0]) else None),
            "birth_ymd": _birth_ymd(c2[0] if c2 else "", c2[2] if len(c2) > 2 else "",
                                    race_year),
            # 2行目の4セル目は「負担重量 + 騎手成績」。この馬とこの騎手の組み合わせ
            # 成績で、月次ファイルの `騎手成績` 列と同じもの
            "weight_carried": (re.match(r"([\d.]+)", c2[3]).group(1)
                               if len(c2) > 3 and re.match(r"([\d.]+)", c2[3]) else ""),
            "騎手成績": (m.group(1) if len(c2) > 3
                     and (m := re.search(r"(\d+-\d+-\d+-\d+)", c2[3])) else ""),
            "sire_name": c3[0] if c3 else "",
            "dam_name": c4[0] if c4 else "",
            "damsire_name": (c5[0].strip("（）()") if c5 else ""),
            "trainer_name": re.sub(r"（.*?）", "", c3[1]).strip() if len(c3) > 1 else "",
            "trainer_affil": (re.search(r"（(.*?)）", c3[1]).group(1)
                              if len(c3) > 1 and "（" in c3[1] else ""),
            "owner_name": c4[1] if len(c4) > 1 else "",
            "breeder": c5[1] if len(c5) > 1 else "",
            "weight_kg": float(weight.group(1)) if weight else None,
            "weight_diff": weight.group(2) if weight else "",
            "odds_win": float(m_odds.group(1).replace(",", "")) if m_odds else None,
            "popularity": int(m_pop.group(1)) if m_pop else None,
        }
        row.update(_records(result_td) if result_td else {})
        rows.append(row)

    card = pd.DataFrame(rows)
    # 回りが片方しかない競馬場では、その回りの成績行しか描画されない。
    # 列自体は必ず作る（欠けたまま下流へ渡すと KeyError になる）。空欄は
    # 「該当なし」であって 0 戦 0 勝ではないので、0 埋めはしない。
    for col in (*_RECORD_LABELS.values(), "最高タイム", "最高タイム良馬場", "騎手成績"):
        if col not in card.columns:
            card[col] = ""
        card[col] = card[col].fillna("")
    if card.empty:
        raise InsufficientData(f"{race_id}: 出馬表を1行も抽出できませんでした。")
    dup = card["horse_no"].duplicated().sum()
    if dup:
        raise InsufficientData(f"{race_id}: 馬番が {dup} 件重複しています。解析を疑ってください。")
    return attach_keys(card).sort_values("horse_no").reset_index(drop=True)


# ------------------------------------------------------------ レース条件
# 例: 「馬い！淡路洲本農園玉ねぎ記念Ｃ３一 ダート 1500ｍ（左） 天候：曇
#      馬場：良 サラブレッド系 一般 ＊電話投票コード：32# 賞金 1着900,000円 …」
_DISTANCE_RE = re.compile(r"(\d{3,4})\s*[ｍm]")
_SURFACE_RE = re.compile(r"(ダート|芝)")
_TURN_RE = re.compile(r"[（(]\s*(左|右|直線?)\s*[）)]")
_BABA_RE = re.compile(r"馬場\s*[：:]\s*(良|稍重|重|不良)")
_PRIZE_RE = re.compile(r"1着\s*([\d,]+)\s*円")


def parse_race_header(html: str) -> dict:
    """レース条件（距離・馬場・クラス・賞金）を当日ページから読む。

    ここが無かったため、推論は `class_level=1` / `distance=1200` /
    `prize_yen=100万` という固定値で走っていた。**学習データに
    class_level=1 は1件も無い**（2 が 77.6%、4 が 19.8%、5 が 2.6%）ので、
    モデルが一度も見ていない値を毎回渡していたことになる。

    読めなければ空 dict を返す。呼び出し側で固定値に落とさず失敗させる。
    """
    soup = BeautifulSoup(html, "lxml")
    node = soup.select_one(".raceTitle")
    if node is None:
        return {}
    text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))

    out: dict = {"race_title": text[:120]}
    if m := _DISTANCE_RE.search(text):
        out["distance"] = int(m.group(1))
    if m := _SURFACE_RE.search(text):
        out["surface"] = "ダ" if m.group(1) == "ダート" else "芝"
    if m := _TURN_RE.search(text):
        out["turn"] = m.group(1)
    if m := _BABA_RE.search(text):
        out["baba_condition"] = m.group(1)
    if m := _PRIZE_RE.search(text):
        out["prize_yen"] = float(m.group(1).replace(",", ""))

    out["class_name"] = _class_name(text)
    # 序数化は学習側の関数に通す。ここで別に書くと、学習と推論で
    # 違う水準が振られる（SK-02 と同じ理由）。
    from nar.transform.silver import _class_level

    out["class_level"] = int(_class_level(pd.Series([out["class_name"]])).iloc[0])
    return out


# 月次ファイルの「競走種類名称」は5値しか取らない（普通 / 一般 / 特別 /
# 重賞 / 準重賞）。学習はこの文字列から水準を作っているので、当日ページの
# 自由文から同じ5値へ寄せる。ここを合わせないと、学習に存在しない水準
# （1 や 3）がモデルに渡る。
# 全角・半角が混在する（実ページは「ＪpnIII」のように Ｊ だけ全角）。
_GRADE_RE = re.compile(r"重賞|[ＧG][ⅠⅡⅢI１２３123]|[ＪJ]pn\s?[ⅠⅡⅢI１２３123]{1,3}")


def _class_name(text: str) -> str:
    """レース見出しから競走種類名称を復元する。

    「特別」は園田のように距離表記より前（レース名側）に出ることがあるので
    見出し全体を見る。どれにも当たらなければ「普通」。普通と一般は
    同じ水準（2）で、学習データの 77.6% を占める最頻値でもある。
    """
    if _GRADE_RE.search(text):
        return "重賞"
    if "特別" in text:
        return "特別"
    if "一般" in text:
        return "一般"
    return "普通"


def attach_keys(card: pd.DataFrame) -> pd.DataFrame:
    """学習側と同一の関数で代理キーを付ける。

    ここを運用側で書き直すと、同じ馬・同じ騎手が学習時と違うキーになり、
    過去成績が1件も引けなくなる。キー生成は学習側の実装を必ず共有する（SK-02）。
    """
    from nar.transform.keys import add_horse_sk, person_keys

    out = add_horse_sk(card, name="horse_name", birth="birth_ymd", sire="sire_name")
    out = person_keys(out, "jockey_name", "jockey_affil", "jockey")
    out = person_keys(out, "trainer_name", "trainer_affil", "trainer")
    out["sire_sk"] = out["sire_name"].astype(str).str.strip()
    out["dam_sk"] = out["dam_name"].astype(str).str.strip()
    return out
