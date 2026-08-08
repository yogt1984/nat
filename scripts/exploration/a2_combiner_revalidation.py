"""A-2 combiner revalidation. Criteria PRE-REGISTERED here, before any result is seen:
(a) walk-forward pooled IC > 0.05   (b) positive-fold share >= 0.55
(c) no fold > 30% of |IC| mass      (d) composite must beat its best single feature
(e) fold ICs must NOT be monotone   (the §5 tell)
"""
import sys, warnings, json; warnings.filterwarnings("ignore")
sys.path.insert(0,'/home/onat/nat/scripts')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from cluster_pipeline.loader import load_parquet
from alpha.walkforward_ic import walk_forward_ic, fit_ic_weights, _composite
from algorithms.hierarchical_combiner import L1_FEATURES, L2_FEATURES, L3_FEATURES

def _base(c):            # only the TRAILING bar-aggregation suffix
    for s in ('_mean','_last','_sum'):
        if c.endswith(s): return c[: -len(s)]
    return c
RAW = {_base(c): c for c in L1_FEATURES + L2_FEATURES + L3_FEATURES}
DATA = Path('/home/onat/nat/data/features')
days = sorted(d.name for d in DATA.iterdir() if d.is_dir())[-26:-1]
HZ = 6   # bars (30 min) — the longest horizon this sample can support

for sym in ("BTC","ETH","SOL"):
    cols = ['timestamp_ns','symbol','raw_midprice'] + list(RAW)
    df = load_parquet(str(DATA), symbols=[sym], start_date=days[0], end_date=days[-1],
                      columns=cols, max_memory_mb=8000)
    df['bar'] = df.timestamp_ns // (300*10**9)
    agg = {c:'mean' for c in RAW}; agg['raw_midprice']='last'
    bars = df.groupby('bar').agg(agg).reset_index(drop=True)
    bars['fwd_ret'] = bars.raw_midprice.shift(-HZ)/bars.raw_midprice - 1
    bars = bars.dropna().reset_index(drop=True)
    # NON-OVERLAPPING: consecutive bars share HZ-1 bars of their forward window, so a
    # naive IC is computed on ~n/HZ independent observations while claiming n. XS-3 used
    # non-overlapping daily rebalances for the same reason.
    bars = bars.iloc[::HZ].reset_index(drop=True)
    feats = [c for c in RAW if c in bars.columns]
    r = walk_forward_ic(bars, feats, 'fwd_ret', n_folds=5, embargo=1, min_train=60)
    best = max((abs(spearmanr(bars[c], bars.fwd_ret).statistic), c) for c in feats)
    print(f"\n{sym}: {len(bars):,} NON-OVERLAPPING obs over {len(days)} days, {r['n_folds_used']} folds")
    print(f"  pooled IC       {r['pooled_ic']:+.4f}   (§5 claimed .178/.248/.359)")
    print(f"  positive folds  {r['positive_fold_share']:.2f}   max-fold share {r['max_fold_share']:.2f}")
    print(f"  fold ICs        {[round(f['ic'],3) for f in r['folds']]}")
    print(f"  monotone?       {r['fold_ic_monotone']}   trend {r['fold_ic_trend']:+.4f}")
    print(f"  best single     {best[1]} |IC|={best[0]:.4f} (full-sample, so generous)")
    v = {'a':r['pooled_ic']>0.05,'b':r['positive_fold_share']>=0.55,
         'c':r['max_fold_share']<=0.30,'d':abs(r['pooled_ic'])>best[0],
         'e':not r['fold_ic_monotone']}
    print(f"  PRE-REGISTERED: {v} -> {'SURVIVES' if all(v.values()) else 'FAIL(' + ','.join(k for k,x in v.items() if not x) + ')'}")
