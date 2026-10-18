"""指定レースの買い目を、本番の推論と同じ部品で計算して発注計画（JSON）にする。

  python scripts/plan_race.py <race_id> [出力先]    例: 312026101805

クラウドの自動配信（NAROPS_ORDER_BUCKET）が繋がるまでの手動運用・動作確認用。発注はしない。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "learning/src"))
import pandas as pd, numpy as np
from datetime import timedelta
from narops.config import OpsConfig
from narops.gcp import BigQueryWarehouse
from narops.nar_source import NarSource
from narops.model.registry import ModelRegistry
from narops.runtime import load_models, load_bet_bundle
from narops.shared import feature_config_for, popularity_from_odds
from narops.features import build_for_race
from narops.inference import run_inference
from narops.discord.format import race_embed
from narops.clock import FixedClock, to_utc
from narops.service import _schedule_row
cfg = OpsConfig.load()
wh = BigQueryWarehouse(project=cfg.project, dataset=cfg.raw["gcp"]["resources"]["bq_dataset"], location=cfg.region, max_bytes_billed=cfg.max_bytes_billed)
rel = ModelRegistry('data/nar-model').load('v2026.10.18-A', verify=True)
models, std = load_models(rel); bets = load_bet_bundle(rel)
fcfg = feature_config_for("flat_pop_cat")
race_id = sys.argv[1]
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / ".cache/nar-orders" / f"plan_{race_id}.json"
row = _schedule_row(wh, race_id)
day = pd.Timestamp(row["race_date"]).date()
with NarSource() as src:
    card = src.fetch_entry_card(race_id, int(row["baba_code"]), day, int(row["race_no"]))
    odds_df = src.fetch_odds(race_id, int(row["baba_code"]), day, int(row["race_no"]))
for key in ("distance", "class_level", "class_name", "prize_yen", "surface", "turn", "baba_condition", "race_name", "condition_name", "weather"):
    if key in card.columns and card[key].notna().any():
        row[key] = card[key].iloc[0]
pop_by = dict(zip(odds_df["horse_no"], popularity_from_odds(odds_df["odds_win"])))
card = card.assign(popularity=card["horse_no"].map(pop_by).astype(float))
feats = build_for_race(wh, card, row, rel.manifest, fcfg, max_bytes_billed=cfg.max_bytes_billed)
scored = std.apply(feats.frame, list(feats.spec.names))
merged = feats.frame[["horse_no"]].merge(odds_df, on="horse_no", how="left")
start = to_utc(row["start_ts"])
res = run_inference(race_id=race_id, features=scored, models=models, manifests=[rel.manifest] * len(models),
    weights=rel.manifest.ensemble_weights, temperature=1.0, model_temperatures=rel.manifest.model_temperatures,
    clock=FixedClock(start - timedelta(minutes=13)), cfg=cfg, odds=merged["odds_win"] if merged["odds_win"].notna().all() else None,
    baba_code=int(row["baba_code"]), class_level=int(row.get("class_level", 1)), day_budget_remaining=50000,
    popularity=feats.frame["horse_no"].map(pop_by).to_numpy(dtype=float),
    waku=pd.to_numeric(feats.frame["horse_no"].map(dict(zip(card["horse_no"], card["waku"]))), errors="coerce").to_numpy(dtype=float),
    bets=bets)
from narops.orders.plan import from_bet_plans
plan = from_bet_plans([p for p in res.exotic if hasattr(p, "structure")], race_id=race_id, race_date=day, start_ts=start,
                      track_name=str(row.get("track_name", "")), race_no=int(row["race_no"]), release=rel.release_id)
out.write_text(plan.to_json(), encoding="utf-8")
print("発走", start.astimezone().strftime("%H:%M"), "| notes:", res.notes)
for ln in plan.lines:
    print(f"  {ln.bet_type}: {ln.tickets} ¥{ln.stake_yen}")
print("→", out)
