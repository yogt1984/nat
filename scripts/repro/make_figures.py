"""DOCS-4 — regenerate NAT's headline descriptive numbers from the frozen slice.

A stranger with this repo and `pandas + pyarrow + matplotlib` runs `./reproduce.sh`
and gets three figures plus a JSON of headline numbers, computed from data frozen in
git (`reproduce/slice/`) — not from prose. The three claims reproduced:

  1. FINDINGS §7.2 — the universe half-spread distribution: the median pair quotes
     an order of magnitude wider than BTC.
  2. FINDINGS §7.10 — touch depth: the median pair carries only hundreds of dollars
     at the touch; almost none hold $5k.
  3. FINDINGS §4.11 — the maker ladder: at BTC's touch, net edge per fill is
     negative until deep into the rebate tiers ("zero fees are not free money").

Frozen inputs, declared: the slice is 2 L2 snapshot days (2026-08-07, 2026-08-10;
XS-8 sampler output) and the adverse-selection-given-fill constants measured in
FINDINGS §4.7 (E[adverse|fill] = 0.228 bps bid / 0.242 bps ask, BTC 2026-07-29→30).
Everything else — spreads, depths, the fee ladder — is recomputed here from the
slice and from `config/costs.toml` (the SSOT; no fee literal appears in this file).

Reproduced numbers are the *slice's own*: they should match the recorded claims in
shape and order of magnitude, not to the third decimal — the recorded values were
measured over more days. The pinning test (`scripts/tests/test_reproduce_slice.py`)
asserts the claim-level facts, and `reproduce/out/headlines.json` carries the exact
regenerated values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from utils.costs import load_costs  # noqa: E402
from xs.capacity import aggregate_l2, load_l2_snapshots  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SLICE = ROOT / "reproduce" / "slice"
DEFAULT_OUT = ROOT / "reproduce" / "out"

#: FINDINGS §4.7 (BTC, 2026-07-29→30): expected adverse move conditional on a
#: passive fill at the touch, in bps. A frozen *measured* input, not a parameter —
#: reproducing it needs tick data the slice cannot carry.
ADVERSE_BPS = {"bid": 0.228, "ask": 0.242}

# Reference palette (dataviz skill) — light mode, single-hue + diverging poles.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
BLUE = "#2a78d6"
RED = "#e34948"


# ── computation (pure; the pinning test calls these) ─────────────────────────

def slice_aggregate(slice_dir: Path | str = DEFAULT_SLICE) -> pd.DataFrame:
    return aggregate_l2(load_l2_snapshots(Path(slice_dir) / "l2"), min_snapshots=10)


def spread_stats(agg: pd.DataFrame) -> dict:
    hs = agg["half_spread_bps"]
    btc = float(hs.loc["BTC"])
    return {
        "n_pairs": int(len(hs)),
        "btc_half_spread_bps": round(btc, 3),
        "median_half_spread_bps": round(float(hs.median()), 3),
        "median_to_btc_ratio": round(float(hs.median() / btc), 1),
    }


def touch_stats(agg: pd.DataFrame) -> dict:
    touch = agg["touch_notional"]
    return {
        "median_touch_usd": round(float(touch.median()), 0),
        "n_pairs_touch_ge_5k": int((touch >= 5000).sum()),
        "n_pairs": int(len(touch)),
    }


def maker_ladder(agg: pd.DataFrame, costs: dict) -> dict:
    """Net edge per FILL at BTC's touch, by maker rung: hs + rate − E[adverse|fill].

    Sign convention from the SSOT ladder: positive = rebate earned. The recorded
    §4.11 numbers are EV per POSTING (they multiply by fill probability); the sign
    structure — which rung first goes positive — is identical and is the claim.
    """
    btc_hs = float(agg.loc["BTC", "half_spread_bps"])
    rates = costs["hyperliquid_maker_tiers"]["rates_bps"]  # KeyError if SSOT moves
    rungs = {name: {
        "rate_bps": float(rate),
        "edge_bid_bps": round(btc_hs + float(rate) - ADVERSE_BPS["bid"], 3),
        "edge_ask_bps": round(btc_hs + float(rate) - ADVERSE_BPS["ask"], 3),
    } for name, rate in rates.items()}
    breakeven = round(ADVERSE_BPS["bid"] - btc_hs, 3)
    viable = [n for n, r in sorted(rungs.items(), key=lambda kv: kv[1]["rate_bps"])
              if r["edge_bid_bps"] > 0 and r["edge_ask_bps"] > 0]
    return {"btc_half_spread_bps": round(btc_hs, 3),
            "breakeven_maker_rate_bps": breakeven,
            "first_viable_rung": viable[0] if viable else None,
            "rungs": rungs}


# ── figures ──────────────────────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_spread(agg: pd.DataFrame, stats: dict, out: Path):
    import matplotlib.pyplot as plt
    hs = agg["half_spread_bps"]
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    bins = np.logspace(np.log10(max(hs.min(), 0.01)), np.log10(hs.max()), 30)
    ax.hist(hs, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    for x, label, dy in ((stats["btc_half_spread_bps"], "BTC", 0),
                         (stats["median_half_spread_bps"],
                          f"median = {stats['median_half_spread_bps']} bps "
                          f"({stats['median_to_btc_ratio']}x BTC)", 0)):
        ax.axvline(x, color=INK_2, linewidth=1, linestyle="--")
        ax.annotate(label, (x, ax.get_ylim()[1] * (0.95 - dy)), color=INK,
                    fontsize=9, ha="left", xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("median half-spread (bps, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("pairs", color=INK_2, fontsize=9)
    ax.set_title(f"Half-spread across {stats['n_pairs']} perp pairs — "
                 "the venue is not its majors (§7.2)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig1_spread_distribution.png", dpi=150)
    plt.close(fig)


def fig_touch(agg: pd.DataFrame, stats: dict, out: Path):
    import matplotlib.pyplot as plt
    touch = agg["touch_notional"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    bins = np.logspace(0, np.log10(touch.max()), 30)
    ax.hist(touch, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
    ax.set_xscale("log")
    ax.axvline(5000, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"$5k — held by {stats['n_pairs_touch_ge_5k']}/{stats['n_pairs']} pairs",
                (5000, ax.get_ylim()[1] * 0.95), color=INK, fontsize=9,
                ha="left", xytext=(4, 0), textcoords="offset points")
    ax.axvline(stats["median_touch_usd"], color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(f"median ${stats['median_touch_usd']:.0f}",
                (stats["median_touch_usd"], ax.get_ylim()[1] * 0.85), color=INK,
                fontsize=9, ha="left", xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("median touch notional (USD, log scale)", color=INK_2, fontsize=9)
    ax.set_ylabel("pairs", color=INK_2, fontsize=9)
    ax.set_title("Depth at the touch — wide pairs are nearly empty (§7.10)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2_touch_depth.png", dpi=150)
    plt.close(fig)


def fig_ladder(ladder: dict, out: Path):
    import matplotlib.pyplot as plt
    order = sorted(ladder["rungs"].items(), key=lambda kv: kv[1]["rate_bps"])
    names = [n for n, _ in order]
    edges = [r["edge_bid_bps"] for _, r in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    colors = [BLUE if e > 0 else RED for e in edges]  # diverging by polarity
    bars = ax.bar(names, edges, color=colors, width=0.6)
    ax.axhline(0, color=INK_2, linewidth=1)
    for bar, e in zip(bars, edges):
        ax.annotate(f"{e:+.2f}", (bar.get_x() + bar.get_width() / 2, e), color=INK,
                    fontsize=9, ha="center",
                    xytext=(0, 4 if e > 0 else -12), textcoords="offset points")
    ax.set_ylabel("net edge per fill at BTC touch (bps)", color=INK_2, fontsize=9)
    ax.set_title(f"Maker ladder — breakeven rebate "
                 f"+{ladder['breakeven_maker_rate_bps']} bps (§4.11)",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "fig3_maker_ladder.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slice", default=str(DEFAULT_SLICE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-plots", action="store_true",
                    help="headline numbers only (no matplotlib needed)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    agg = slice_aggregate(args.slice)
    costs = load_costs()
    headlines = {
        "spread": spread_stats(agg),
        "touch": touch_stats(agg),
        "maker_ladder": maker_ladder(agg, costs),
        "slice": {"l2_days": sorted(p.name for p in
                                    (Path(args.slice) / "l2").iterdir()),
                  "n_pairs_aggregated": int(len(agg))},
    }
    (out / "headlines.json").write_text(json.dumps(headlines, indent=2))

    s, t, m = headlines["spread"], headlines["touch"], headlines["maker_ladder"]
    print(f"pairs measured: {s['n_pairs']}")
    print(f"§7.2  half-spread: BTC {s['btc_half_spread_bps']} bps, universe median "
          f"{s['median_half_spread_bps']} bps = {s['median_to_btc_ratio']}x BTC")
    print(f"§7.10 touch depth: median ${t['median_touch_usd']:.0f}, "
          f"{t['n_pairs_touch_ge_5k']}/{t['n_pairs']} pairs hold $5k")
    print(f"§4.11 maker ladder: breakeven rebate +{m['breakeven_maker_rate_bps']} bps "
          f"-> first viable rung: {m['first_viable_rung']}")

    if not args.no_plots:
        fig_spread(agg, s, out)
        fig_touch(agg, t, out)
        fig_ladder(m, out)
        print(f"figures -> {out}/fig1..3*.png")
    print(f"headlines -> {out}/headlines.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
