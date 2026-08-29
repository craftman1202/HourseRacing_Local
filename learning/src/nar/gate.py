"""リリースゲートの証跡を作る。

`assert_publish_gate` は学習側 Blocker が全 GREEN であることを要求する。
その判定を手書きの辞書で渡せてしまうと、ゲートは何も守らない。
実際にテストを走らせた結果と、実データの OOS ガードの発火状況から作る。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger(__name__)

# 運用側 release.REQUIRED_GREEN と同じ集合。片方だけ増えると気付けないので、
# 突き合わせるテストを置いてある。
REQUIRED = ("IG-14", "LK-05", "LK-06", "EV-03", "RF-01", "RF-02", "RF-03",
            "RF-07", "RF-08", "CV-07")


def _normalize(test_id: str) -> str:
    return test_id.replace("-", "").lower()


def run_pytest(tests_dir: str | Path, report_path: str | Path,
               extra_args: tuple[str, ...] = ()) -> Path:
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-q", "--tb=no",
           f"--junitxml={out}", *extra_args]
    subprocess.run(cmd, check=False)
    if not out.exists():
        raise RuntimeError(f"pytest のレポートが作られませんでした: {' '.join(cmd)}")
    return out


def outcomes_from_junit(report_path: str | Path) -> dict[str, str]:
    """テスト ID ごとの GREEN/RED。

    ノード名に ID（ハイフン無し・小文字）が含まれるテストを集め、1件でも
    失敗・エラーがあれば RED。1件も無ければ辞書に入れない（= 未実施）。
    未実施を GREEN と読み替えないのが要点。
    """
    root = ET.parse(report_path).getroot()
    cases = root.iter("testcase")
    hits: dict[str, list[bool]] = {t: [] for t in REQUIRED}
    for case in cases:
        name = _normalize(f"{case.get('classname', '')}.{case.get('name', '')}")
        failed = any(case.find(tag) is not None for tag in ("failure", "error"))
        for test_id in REQUIRED:
            if _normalize(test_id) in name:
                hits[test_id].append(not failed)
    return {t: ("GREEN" if all(r) else "RED") for t, r in hits.items() if r}


def guards_from_oos(guards_csv: str | Path) -> dict[str, str]:
    """実データの OOS ガード結果。

    RF 系は「単体テストが通ること」と「本番相当の数値で発火しないこと」の
    両方が要る。単体テストだけ見ていると、実データで RF-01 が鳴っていても
    publish できてしまう。
    """
    import pandas as pd

    path = Path(guards_csv)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "id" not in df.columns or "fired" not in df.columns:
        return {}
    fired = df.groupby("id")["fired"].any()
    return {str(k): ("RED" if bool(v) else "GREEN") for k, v in fired.items()}


def build(tests_dir: str | Path, artifacts: str | Path,
          run_tests: bool = True) -> dict:
    art = Path(artifacts)
    report = art / "junit.xml"
    if run_tests:
        run_pytest(tests_dir, report)
    results = outcomes_from_junit(report) if report.exists() else {}

    guards = guards_from_oos(art / "oos_guards.csv")
    # 実データで発火したガードは、単体テストが緑でも RED に落とす
    for key, verdict in guards.items():
        if key in REQUIRED and verdict == "RED":
            results[key] = "RED"

    missing = [t for t in REQUIRED if t not in results]
    payload = {
        "results": results,
        "missing": missing,
        "oos_guards": guards,
        "can_publish": not missing and all(v == "GREEN" for v in results.values()),
    }
    (art / "gate_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
