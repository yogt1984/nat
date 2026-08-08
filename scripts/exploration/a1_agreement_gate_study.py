import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'/home/onat/nat/scripts')
import numpy as np, pandas as pd
from pathlib import Path
from cluster_pipeline.loader import load_parquet
from processes.base import ProcessContext
from processes.agreement_gate_eval import AgreementGateEval

DATA = Path('/home/onat/nat/data/features')
days = sorted(d.name for d in DATA.iterdir() if d.is_dir())[-26:-1]
FAST, SLOW = 'imbalance_qty_l1', 'regime_divergence_1h'   # §2 contract: fast dir, slow bias
for sym in ("BTC","ETH","SOL"):
    df = load_parquet(str(DATA), symbols=[sym], start_date=days[0], end_date=days[-1],
                      columns=['timestamp_ns','symbol','raw_midprice',FAST,SLOW],
                      max_memory_mb=8000)
    df['bar'] = df.timestamp_ns // (300*10**9)
    bars = df.groupby('bar').agg(timestamp_ns=('timestamp_ns','last'),
                                 raw_midprice=('raw_midprice','last'),
                                 **{FAST:(FAST,'mean'), SLOW:(SLOW,'mean')}).reset_index(drop=True)
    bars['symbol']=sym
    ctx = ProcessContext(symbol=sym, timeframe='5min', price_col='raw_midprice',
                         horizons={'30m':6,'2h':24}, costs={})
    r = AgreementGateEval(fast=FAST, slow=SLOW, n_shuffles=300).evaluate(bars, ctx)
    print(f"\n{sym}: {len(bars):,} bars, {len(days)} days")
    for f in r.findings:
        e=f.extras
        print(f"  hz={f.horizon:<4} ic_uncond={e['ic_unconditional']:+.4f} "
              f"ic_agree={e['ic_agree']:+.4f} ic_disagree={e['ic_disagree']:+.4f}")
        print(f"           lift={e['raw_lift']:+.4f} z={e['z']} n_agree={e['n_agree']} "
              f"fracD={e['frac_days_informative']:.2f} -> {e['verdict']} "
              f"{'INFORMATIVE' if f.informative else ''}")
