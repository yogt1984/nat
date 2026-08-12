# Reproduce NAT's headline numbers

```
./reproduce.sh          # from the repo root
```

Needs `python3` with `pandas`, `pyarrow`, `matplotlib` (`pip install pandas pyarrow
matplotlib`); the project venv is used automatically when present. No network, no
API keys, no other data: the script reads only `reproduce/slice/` (frozen in this
repo) and `config/costs.toml`, and writes figures + `headlines.json` to
`reproduce/out/`.

## What it regenerates

| claim | recorded (FINDINGS) | from this slice |
|---|---|---|
| §7.2 — universe half-spread vs BTC | median 1.372 bps = 17.7× BTC | 1.428 bps = 18.6× BTC |
| §7.10 — depth at the touch | 4/177 pairs hold $5k | 4/177, median $81 |
| §4.11 — maker breakeven at BTC touch | +0.144 bps → first viable rung rebate_t2 | +0.151 bps → rebate_t2 |

The recorded values were measured over more snapshot days than the slice carries;
matching in shape and rung, not to the third decimal, is the expected outcome.
`scripts/tests/test_reproduce_slice.py` pins the slice's own values exactly, so a
drift means the slice or the computation changed — both must be deliberate.

## What is frozen, and why

- `slice/l2/2026-08-07,2026-08-10/` — two days of XS-8 order-book snapshot sweeps
  (~9.7 MB, 177 perp pairs, ~50k rows). The venue serves no historical L2, so this
  data is unreproducible from any public source: freezing it in git is the only
  way a third party can ever check these numbers.
- `slice/candles_1h/` — BTC/ETH/SOL hourly OHLCV samples from the candle archive,
  for shape/context (venue retention caps history, so these too cannot be
  re-fetched indefinitely).
- Two constants from FINDINGS §4.7 are carried in `scripts/repro/make_figures.py`
  (`ADVERSE_BPS`): expected adverse selection conditional on a fill at BTC's
  touch. They come from tick data too large to freeze; they are inputs to the
  ladder figure and are declared, not hidden.

Fees and the maker-tier ladder are **not** frozen — they load from
`config/costs.toml`, the project's single source of truth, so a fee change
reprices the ladder figure automatically.
