# NAT8 — `nat viz spectral/regime/correlation` — Lightweight Wrappers

**Priority**: 8 (~4h total)
**Status**: NOT STARTED
**Depends on**: NAT3 (viz library foundation)

---

## Objective

Wrap existing analysis scripts into three `nat viz` subcommands: `spectral`, `regime`, and `correlation`. These are lightweight integrations — the computation code already exists, this task provides the CLI entry point and terminal rendering.

## Context

- Spectral analysis exists in `scripts/analysis/` and `nat spannung spectral`
- Regime analysis exists in `nat spannung regime` and cluster tools
- Correlation analysis exists in `scripts/viz/correlations.py` and `reports/signal_correlation.json`

These commands repackage existing functionality under the unified `nat viz` interface.

## Scope

**In scope**:
- `nat viz spectral <feature>` — PSD, coherence with returns, frequency-band IC
- `nat viz regime` — current regime state, transition history, regime-conditional IC
- `nat viz correlation` — cross-algorithm signal correlation matrix

**Out of scope**:
- New spectral analysis algorithms
- New regime detection methods
- Modifying underlying analysis scripts

## Implementation

### `nat viz spectral <feature>` (~1.5h)

Wraps `nat spannung spectral` output into terminal + PNG:

```python
def cmd_viz_spectral(args):
    """PSD + coherence for a single feature."""
    # Delegate to existing spectral analysis
    # Terminal: show PSD slope, Hurst exponent, dominant frequency, IC by band
    # PNG: 3-panel (PSD, coherence, frequency-band IC bars)
```

Terminal output:
```
  Spectral Analysis — imbalance_qty_l1_last (BTC, 4h)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PSD slope:        -1.86 (brown noise)
  Hurst exponent:    0.43 (slightly mean-reverting)
  Dominant freq:     0.015 Hz (68s period)
  OU half-life:      5.7s

  IC by Frequency Band:
    Ultra-low (0.005-0.1 Hz):  +0.45  ████████████████████  ← all signal here
    Mid       (0.1-1.0 Hz):    +0.02  █
    High      (1.0-5.0 Hz):    -0.01
```

### `nat viz regime` (~1.5h)

Shows current regime classification and IC lift:

```python
def cmd_viz_regime(args):
    """Current regime state and regime-conditional IC."""
    # Load latest feature data
    # Classify current regime using ent_book_shape or GMM
    # Show: current regime, duration, IC lift, historical transitions
```

Terminal output:
```
  Regime State — BTC (current)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Current regime:    LOW ENTROPY (structured book)
  Duration:          14 minutes
  ent_book_shape:    2.81 (quintile: Q1)
  Expected IC lift:  +22% (0.45 → 0.55)

  Regime History (last 4h):
    ██████░░░░██████████░░░░░░████████░░░░██████████████
    LOW    HIGH  LOW          HIGH      LOW
```

### `nat viz correlation` (~1h)

Terminal correlation matrix:

```python
def cmd_viz_correlation(args):
    """Cross-algorithm signal correlation matrix."""
    # Load signal_correlation.json or compute from recent data
    # Rich table with color-coded cells
```

Terminal output:
```
  Signal Correlation Matrix (last 24h)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              jump   optimal  funding  3f_liq
  jump         —      0.12     0.08    0.21
  optimal    0.12       —      0.15    0.31
  funding    0.08     0.15       —     0.09
  3f_liq     0.21     0.31     0.09      —

  Max correlation: 0.31 (optimal × 3f_liq) — OK (<0.35)
```

## Files to Create / Modify

| File | Action |
|------|--------|
| `scripts/viz/spectral.py` | Create new (wraps existing analysis) |
| `scripts/viz/regime.py` | Create new |
| `scripts/viz/correlations.py` | Extend existing with `render_terminal()` |
| `nat` | Add 3 handlers + register subparsers |

## Acceptance Criteria

### Spectral:
- [ ] `nat viz spectral imbalance_qty_l1_last --symbol BTC` shows PSD slope, Hurst, dominant freq
- [ ] IC by frequency band displayed with bar chart
- [ ] `--output spectral.png` saves 3-panel matplotlib figure
- [ ] Error if feature name is invalid

### Regime:
- [ ] `nat viz regime --symbol BTC` shows current regime state and duration
- [ ] Regime history displayed as Unicode timeline
- [ ] Expected IC lift shown based on current regime
- [ ] Works even if GMM model is not trained (falls back to ent_book_shape quintiles)

### Correlation:
- [ ] `nat viz correlation` shows cross-algorithm correlation matrix
- [ ] Color-coded: green < 0.3, yellow 0.3-0.6, red > 0.6
- [ ] Max correlation pair highlighted
- [ ] `--json` exports matrix as JSON

### All three:
- [ ] Each supports `--json` for data export
- [ ] Each supports `--output <path>` for PNG export
- [ ] Runs without error on current data

## Testing / Verification

```bash
# 1. Spectral
nat viz spectral imbalance_qty_l1_last --symbol BTC
nat viz spectral raw_spread_bps_mean --symbol ETH --json

# 2. Regime
nat viz regime --symbol BTC
nat viz regime --symbol SOL --hours 8

# 3. Correlation
nat viz correlation
nat viz correlation --json

# 4. PNG export for all three
nat viz spectral imbalance_qty_l1_last --output /tmp/spectral.png
nat viz regime --output /tmp/regime.png
nat viz correlation --output /tmp/corr.png
ls -la /tmp/{spectral,regime,corr}.png
```
