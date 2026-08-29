"""共有フィクスチャ。

ネットワーク依存のテストは全体の5%以下に抑える方針なので、既定では
httpx の MockTransport とローカル合成データだけで完結させる。
"""

from __future__ import annotations

import io
import zipfile

import pytest

from nar import synth


@pytest.fixture(scope="session")
def synth_tables():
    """FX-03 相当。真の β が既知の Plackett-Luce 合成レース。"""
    return synth.generate(synth.SynthConfig(n_races=1200, seed=7))


@pytest.fixture(scope="session")
def synth_large():
    """MD-04（真の β の回収）用。係数の SE を締めるため件数を増やす。"""
    return synth.generate(synth.SynthConfig(n_races=6000, seed=11))


@pytest.fixture
def entry(synth_tables):
    return synth_tables["entry"]


@pytest.fixture
def race(synth_tables):
    return synth_tables["race"]


@pytest.fixture
def payout(synth_tables):
    return synth_tables["payout"]


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _race_csv(extra_columns: int = 0) -> bytes:
    """レース一覧は公式仕様で66列。列数はスキーマガードの第一関門なので実数に合わせる。"""
    named = ["競走年月日", "競馬場名", "レース番号", "馬名", "騎手名"]
    cols = named + [f"予備{i}" for i in range(1, 66 - len(named) + 1 + extra_columns)]
    vals = ["20260724", "大井", "11", "テスト馬", "山田太郎"] + ["0"] * (len(cols) - len(named))
    return (",".join(cols) + "\n" + ",".join(vals) + "\n").encode("cp932")


@pytest.fixture
def golden_zip() -> bytes:
    """FX-01 相当。実 ZIP が手元に無いので、CP932 の日本語列名を持つ66列 CSV で代替する。"""
    return make_zip({"RACE.csv": _race_csv()})


@pytest.fixture
def drift_zip() -> bytes:
    """FX-06: 列を1本追加した改変 ZIP（66 → 67列）。"""
    return make_zip({"RACE.csv": _race_csv(extra_columns=1)})
