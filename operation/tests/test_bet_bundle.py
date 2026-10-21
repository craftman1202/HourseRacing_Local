"""全券種の買い目（全モデルアンサンブルのリリース、2026-10-17）の運用側。

判定の単一実装は学習側 nar.eval.recommend / nar.eval.payout。ここでは
リリースからの読み込み・推論での組み立て（Kelly の賭け金は使わない）・記録・通知を固定する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from narops.clock import FixedClock, jst_datetime, to_utc
from narops.discord.format import race_embed
from narops.inference import run_inference
from narops.model.manifest import Manifest
from narops.pipeline import write_bet_candidates
from narops.runtime import load_bet_bundle
from narops.shared import BetPlan, popularity_from_odds

pytestmark = pytest.mark.unit
START = jst_datetime(2026, 10, 18, 15, 30)
TABLE = {f"{r}_{n}": v for n in range(5, 13)
         for r, v in zip(range(1, n + 1), np.linspace(0.62, 0.02, n) / np.linspace(0.62, 0.02, n).sum())}


def _payout_booster(path: Path, k: int, level: float) -> None:
    """払戻予測のダミー（log 払戻 ≈ level のほぼ定数）。特徴量の並びは本物と同じ。"""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 400
    cols = {f"r{i + 1}": rng.integers(1, 10, n) for i in range(k)}
    X = pd.DataFrame({**cols, "n": rng.integers(6, 12, n), "logp_pop": rng.normal(-3, 1, n),
                      "track": rng.integers(10, 30, n), "rsum": sum(cols.values())})
    if path.name in ("payout_place.txt", "payout_wide.txt"):
        X["slots"] = 3
    y = np.full(n, level) + rng.normal(0, 0.01, n)
    lgb.train({"objective": "regression", "verbosity": -1, "min_data_in_leaf": 50},
              lgb.Dataset(X, label=y), 5).save_model(str(path))


@dataclass
class _Rel:
    path: Path
    release_id: str = "test"


@pytest.fixture
def bets(tmp_path):
    k = {"win": 1, "place": 1, "quinella": 2, "exacta": 2, "wide": 2, "trio": 3, "trifecta": 3}
    # 3連複は高い予想配当（EV が閾値を超える）、それ以外は低い
    level = {"trio": np.log(400.0), "quinella": np.log(2.0)}
    models = {}
    for bt, kk in k.items():
        _payout_booster(tmp_path / f"payout_{bt}.txt", kk, level.get(bt, np.log(1.5)))
        models[bt] = {"file": f"payout_{bt}.txt", "smear": 1.0}
    meta = {"models": models, "pop_table": TABLE,
            "ev_rules": [{"bet": "trio", "t": 2.0, "mode": "top1", "note": "検証 1.16"},
                         {"bet": "quinella", "t": 2.0, "mode": "top1"}],
            "odds_free_rules": [{"structure": "単勝 1位", "lam": 0.5, "min_p": 0.3},
                                {"structure": "枠連 1点", "lam": 0.0, "min_p": 0.3}]}
    (tmp_path / "payout_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return load_bet_bundle(_Rel(tmp_path), rules_file=tmp_path / "none.yaml")


class _Scorer:
    name = "m"

    def score(self, features):
        return np.log(np.array([0.05, 0.45, 0.2, 0.1, 0.12, 0.08])[: len(features)])


def _infer(release_dir, cfg, bets, odds, budget=50_000, waku=(1, 2, 3, 4, 5, 6)):
    m = Manifest.read(Path(release_dir) / "manifest.json")
    n = len(odds)
    feats = pd.DataFrame({"horse_no": np.arange(1, n + 1), "f1": np.zeros(n)})
    odds = pd.Series(odds, dtype=float)
    return run_inference(race_id="212026101801", features=feats, models={"m": _Scorer()},
                         manifests=[m], weights={"m": 1.0}, temperature=1.0,
                         clock=FixedClock(START - timedelta(minutes=10)), cfg=cfg, odds=odds,
                         popularity=popularity_from_odds(odds).to_numpy(),
                         waku=np.asarray(waku, dtype=float), bets=bets,
                         day_budget_remaining=budget)


def test_bundle_loads_rules_and_models(bets):
    assert set(bets.payout) == {"win", "place", "quinella", "exacta", "wide", "trio", "trifecta"}
    assert [r.bet for r in bets.ev_rules] == ["trio", "quinella"]
    assert [r.structure for r in bets.odds_free] == ["単勝 1位", "枠連 1点"]


def test_inference_builds_plans_and_drops_kelly_stakes(release_dir, cfg, bets):
    res = _infer(release_dir, cfg, bets, [9.0, 1.6, 4.0, 7.0, 6.0, 12.0])
    assert (res.frame["stake_yen"] == 0).all()
    by = {pl.bet_type: pl for pl in res.exotic}
    assert set(by) == {"単勝", "枠連", "3連複"}            # 馬連は予想配当が低く EV 未達
    trio = by["3連複"]
    assert len(trio.tickets) == 1 and trio.axis in trio.tickets[0]
    assert trio.detail and trio.detail[0]["ev"] >= 2.0
    assert by["単勝"].tickets == [(2,)]


def test_missing_popularity_beyond_two_means_no_plans(release_dir, cfg, bets):
    res = _infer(release_dir, cfg, bets, [9.0, np.nan, np.nan, np.nan, 6.0, 12.0])
    assert res.exotic == []
    assert any("人気" in n for n in res.notes)


def test_budget_skips_plans_that_do_not_fit(release_dir, cfg, bets):
    res = _infer(release_dir, cfg, bets, [9.0, 1.6, 4.0, 7.0, 6.0, 12.0], budget=150)
    assert sum(pl.stake_yen for pl in res.exotic if pl.stake) <= 150
    assert any("見送り" in n for n in res.notes)


def test_notification_shows_axis_partners_and_tickets(release_dir, cfg, bets):
    res = _infer(release_dir, cfg, bets, [9.0, 1.6, 4.0, 7.0, 6.0, 12.0])
    names = {1: "アカ", 2: "ホシノヒカリ", 3: "ミドリ", 4: "キイロ", 5: "シロ", 6: "クロ"}
    e = race_embed(race_id="R", track_name="名古屋", race_no=1, class_name="C1", distance=1400,
                   start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
                   model_release="v", track_used="A+B", candidates=res.frame, min_ev=1.05,
                   horse_names=names, exotic=res.exotic, day_budget_remaining=50_000)
    d = e.description
    assert "**予想（勝率）**" in d and "◎2 ホシノヒカリ" in d
    assert "🎯 **3連複**" in d and "軸 2番 ホシノヒカリ" in d and "相手" in d
    assert "EV" in d and "予想" in d and "倍" in d
    assert "🎯 **単勝**" in d and "🎯 **枠連**" in d
    assert d.index("🎯 **単勝**") < d.index("🎯 **3連複**") < d.index("🎯 **枠連**")
    assert "買い目合計" in d


def test_records_one_row_per_plan(wh, clock):
    pl = BetPlan("3連複", "3連複 EV≥2（最大1点）", 2, [3, 5], 0.45, [(2, 3, 5)], 100, stake=True)
    frame = pd.DataFrame({"horse_no": [2], "stake_yen": [0]})
    assert write_bet_candidates(wh, "212026101801", frame, "v", date(2026, 10, 18), clock,
                                exotic=[pl]) == 1
    got = wh.query("SELECT horse_no, bet_type, stake_yen FROM bet_candidate WHERE race_date = ?",
                   [date(2026, 10, 18)])
    assert got.to_dict("records") == [{"horse_no": 2, "bet_type": "3連複", "stake_yen": 100}]


def test_rules_file_overrides_the_release_rules(tmp_path):
    """方針（買い目の規則）は conf/bet_rules.yaml が配布物より優先する。"""
    (tmp_path / "payout_meta.json").write_text(json.dumps(
        {"models": {}, "pop_table": {}, "ev_rules": [{"bet": "trio", "t": 2.0}], "odds_free_rules": []}))
    rules = tmp_path / "r.yaml"
    rules.write_text("ev_rules:\n- {bet: win, t: 1.5, p_min: 0.05, mode: all}\nodds_free_rules:\n- {structure: 枠単 1点, lam: 0.0, min_p: 0.6}\n")
    b = load_bet_bundle(_Rel(tmp_path), rules_file=rules)
    assert [(r.bet, r.t, r.p_min, r.mode) for r in b.ev_rules] == [("win", 1.5, 0.05, "all")]
    assert [r.structure for r in b.odds_free] == ["枠単 1点"]


def test_probability_floor_removes_long_shot_tickets(release_dir, cfg, bets):
    """EV が高くても的中確率が足切り未満の組は買わない。"""
    from dataclasses import replace
    loose = replace(bets, ev_rules=[r for r in bets.ev_rules if r.bet == "trio"], odds_free=[])
    strict = replace(loose, ev_rules=[replace(r, p_min=0.9) for r in loose.ev_rules])
    assert _infer(release_dir, cfg, loose, [9.0, 1.6, 4.0, 7.0, 6.0, 12.0]).exotic
    assert _infer(release_dir, cfg, strict, [9.0, 1.6, 4.0, 7.0, 6.0, 12.0]).exotic == []


def test_repo_bet_rules_file_is_valid_and_has_a_cutoff_story():
    import yaml
    from narops.config import CONF_DIR
    from narops.shared import recommend_structures

    d = yaml.safe_load((CONF_DIR / "bet_rules.yaml").read_text(encoding="utf-8"))
    assert {r["bet"] for r in d["ev_rules"]} <= {"win", "place", "quinella", "exacta", "wide", "trio", "trifecta"}
    assert all(r["mode"] in ("all", "top1") and r["t"] >= 1.0 and 0 <= r["p_min"] < 1 for r in d["ev_rules"])
    assert all(r["structure"] in recommend_structures() for r in d["odds_free_rules"])


def test_no_plans_means_prediction_only_notification_not_legacy_win_ev():
    """全券種の買い目の版では、規則に合う買い目が無いレースは「買い目なし」の予想だけ通知する。
    （以前は旧来の「単勝 EV ≥ 基準」の通知に切り替わって出ていた。発注されないのに買い目らしく見える）"""
    cand = pd.DataFrame({"horse_no": [1, 2], "p_win": [0.5, 0.3], "p_top3": [0.8, 0.6],
                         "ev_adjusted": [1.6, 1.2], "stake_yen": [0, 0]})
    kw = dict(race_id="R", track_name="笠松", race_no=3, class_name="C1", distance=1400,
              start_ts=to_utc(START), now=to_utc(START) - timedelta(minutes=10),
              model_release="v", track_used="A+B", candidates=cand, min_ev=1.05, exotic=[])
    assert race_embed(**kw) is not None                    # 旧来の経路（ばんえいなど）は従来どおり
    e = race_embed(**kw, plans_only=True)                  # 買い目が無くても予想は通知する（買い目なしと明記）
    assert e is not None and "買い目なし" in e.description and "単勝" not in e.description
