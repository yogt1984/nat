"""B-5a study driver: run the wide-pair breakeven screen over accumulated XS-8 sweeps.

Prints the survivors-by-beta table (the sound output), the median breakeven exponent,
and the depth-floor curve. Deliberately reports no single survivor count — see
`xs/breakeven.py` for why. Re-run at >= 3 days for B-5c.

Usage:  python -m exploration.b5a_breakeven_study
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'/home/onat/nat/scripts')
import numpy as np, pandas as pd
from xs.capacity import load_l2_snapshots, aggregate_l2
from xs.breakeven import screen, BTC_HALF_SPREAD_BPS, BTC_ADVERSE_BPS

snaps = load_l2_snapshots()
agg = aggregate_l2(snaps, min_snapshots=12)
med = agg.half_spread_bps.median()
print(f"snapshots {len(snaps):,} rows | pairs with >=12 sweeps: {len(agg)}")
print(f"half-spread bps: p5={agg.half_spread_bps.quantile(.05):.3f} p50={med:.3f} p95={agg.half_spread_bps.quantile(.95):.3f}")
if 'BTC' in agg.index:
    print(f"BTC measured here: {agg.loc['BTC'].half_spread_bps:.4f} bps (anchor {BTC_HALF_SPREAD_BPS})")

r = screen(agg, n_snapshots=12)
print("\n=== survivors by adverse-selection scaling exponent ===")
for b, syms in r.survivors_by_beta.items():
    adv = BTC_ADVERSE_BPS*(med/BTC_HALF_SPREAD_BPS)**b
    print(f"  beta={b:<5} adverse@median_pair={adv:8.3f} bps   survivors={len(syms):>3}/{len(r.pairs)}")
print(f"\nmedian breakeven beta* = {r.summary()['median_breakeven_beta']:.3f}")

rj = screen(agg, min_touch_notional=5_000, n_snapshots=12)
print(f"\n=== joint wide-AND-deep (XS-5 floor: touch >= $5k) ===")
print(f"  admitted {len(rj.pairs)} of {len(agg)}; rejected {len(rj.rejected)}")
for b, syms in rj.survivors_by_beta.items():
    print(f"  beta={b:<5} survivors={len(syms):>3}")
if len(rj.pairs):
    print("\ntop admitted pairs by beta*:")
    print(rj.pairs.head(8)[['half_spread_bps','touch_notional','capture_bps','breakeven_beta']].to_string(float_format=lambda x: f"{x:,.3f}"))
