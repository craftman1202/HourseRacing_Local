import sys, time
import pandas as pd

from nar.models.tabm import TabM, TabMConfig
from nar.models.base import to_batch
from nar.train.pipeline import prepare, trainable

print("loading gold...", flush=True)
t0 = time.time()
feat = trainable(pd.read_parquet('data_real/gold/features_noodds/features.parquet'))
print(f"loaded {len(feat):,} rows in {time.time()-t0:.1f}s", flush=True)

cols = ['h_top3rate_prior', 'h_si_last3', 'h_winrate_prior', 'j_winrate_shrunk',
        'jt_winrate_wilson', 'd_pair_winrate', 'd_all_starts', 'd_track_winrate',
        'd_track_starts', 'd_dist_winrate', 'h_starts_prior', 'class_level',
        'h_days_since_prev', 'd_best_speed', 'd_pair_starts', 't_winrate_shrunk',
        'd_dist_starts', 's_winrate_shrunk', 'd_turn_starts', 'd_extra_starts',
        'd_has_outside_history']

dates = pd.to_datetime(feat['race_date'])
tr_mask = dates < '2023-02-01'
va_mask = (dates >= '2023-02-01') & (dates <= '2023-12-31')
t0 = time.time()
train = prepare(feat[tr_mask].copy(), cols)
valid = prepare(feat[va_mask].copy(), cols)
print(f"prepare: {time.time()-t0:.1f}s", flush=True)
print(f"train {len(train):,} rows / {train['race_id'].nunique():,} races", flush=True)
print(f"valid {len(valid):,} rows / {valid['race_id'].nunique():,} races", flush=True)

t0 = time.time()
tb = to_batch(train, cols)
vb = to_batch(valid, cols)
print(f"to_batch: {time.time()-t0:.1f}s  train x shape={tb.x.shape}", flush=True)

cfg = TabMConfig(epochs=2)
print(f"cfg: {cfg}", flush=True)
t0 = time.time()
model = TabM(cfg).fit(tb, vb)
dt = time.time() - t0
print(f"\n2 epochs took {dt:.1f}s -> 60 epochs ~ {dt/2*60/60:.2f}h", flush=True)
