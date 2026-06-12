"""
Spectral / persistence process — frequency-domain description of a feature.

Operates on TICK-level data (100ms ticks, 10 Hz — the sampling rate the
spannung machinery is built around; `data_level = "ticks"`). Per feature:

  - Welch PSD: dominant oscillation periods, noise color, spectral-slope
    Hurst estimate (`compute_psd`)
  - FFT autocorrelation with Ornstein-Uhlenbeck fit: mean-reversion rate
    theta, half-life in seconds (`compute_acf`)
  - Band-pass filtered IC: which frequency bands (20-200s / 2-20s / 0.5-2s /
    0.22-0.5s) carry predictive power at 1s and 5s horizons
    (`compute_band_ic`)
  - Spectral entropy: concentrated (periodic) vs broadband (noise)

Informative iff the best band |IC| >= min_band_ic AND the signal persists at
least as long as that band's prediction horizon (OU half-life >= horizon) —
predictive power without persistence is not tradeable structure.

Wraps `scripts/exploration/spannung_spectral.py` (the analyses behind the
preprint's spectral section: IC 0.45 in the ultra-low band).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from .base import EvaluationProcess, Finding, ProcessContext, ProcessResult, make_run_id, partition_usable_columns
from .registry import register

_PRICE_PREFIXES = ("raw_midprice", "raw_microprice")


def _spannung():
    """Import the spannung spectral module.

    `scripts/exploration/` is on sys.path under pytest (conftest) and the
    editable install, but not necessarily for bare subprocess invocations —
    fall back to inserting it explicitly (the conftest trick).
    """
    try:
        import spannung_spectral
        return spannung_spectral
    except ImportError:
        exploration = Path(__file__).resolve().parent.parent / "exploration"
        if str(exploration) not in sys.path:
            sys.path.insert(0, str(exploration))
        import spannung_spectral
        return spannung_spectral


@register
class SpectralPersistenceProcess(EvaluationProcess):
    """PSD / Hurst / OU half-life / frequency-band IC on tick-level features."""

    data_level = "ticks"

    PARAMS = {
        "features": (None, "list of name prefixes to score; None = all non-meta numeric"),
        "min_band_ic": (0.05, "minimum best-band |IC| for the informative gate"),
        "min_obs": (5000, "minimum finite ticks per feature (Welch needs ~2 segments)"),
    }

    def name(self) -> str:
        return "spectral"

    def evaluate(self, ticks, ctx: ProcessContext) -> ProcessResult:
        t0 = time.time()
        ss = _spannung()
        result = ProcessResult(
            run_id=make_run_id(self.name(), ctx.symbol),
            process=self.name(), kind=self.kind,
            symbol=ctx.symbol, timeframe="tick", params=dict(self.params),
        )
        min_obs = int(self.params["min_obs"])
        min_band_ic = float(self.params["min_band_ic"])

        cols = [
            c for c in self.required_columns(list(ticks.columns))
            if not c.startswith(_PRICE_PREFIXES) and c != ctx.price_col
        ]
        usable, skipped = partition_usable_columns(ticks, cols, min_obs=min_obs)
        result.features_tested = usable
        result.features_skipped = skipped

        prices = ticks[ctx.price_col].to_numpy(dtype=np.float64, na_value=np.nan)

        for feat in usable:
            raw = ticks[feat].to_numpy(dtype=np.float64, na_value=np.nan)
            # Spectral machinery needs a gap-free series: fill NaN with the mean
            x = np.where(np.isfinite(raw), raw, np.nanmean(raw))

            freqs, pxx_db, psd = ss.compute_psd(x)
            _, acf = ss.compute_acf(x)
            band_ics = ss.compute_band_ic(x, prices)
            entropy = ss.compute_spectral_entropy(freqs, pxx_db)

            best = max(band_ics, key=lambda b: abs(b.ic_mean), default=None)
            best_ic = abs(best.ic_mean) if best else 0.0
            horizon_s = best.horizon_ticks / ctx.sample_rate_hz if best else None
            persistent = (
                best is not None and acf.ou_halflife_s >= horizon_s
            )

            result.findings.append(Finding(
                feature=feat, horizon=best.horizon_label if best else None,
                metric="band_ic_max",
                value=round(best.ic_mean, 4) if best else 0.0,
                threshold=min_band_ic,
                informative=bool(best_ic >= min_band_ic and persistent),
                extras={
                    "best_band": best.band_name if best else None,
                    "band_ics": [
                        {"band": b.band_name, "horizon": b.horizon_label,
                         "ic_mean": b.ic_mean, "ic_ir": b.ic_ir,
                         "n_windows": b.n_windows}
                        for b in band_ics
                    ],
                    "hurst": round(psd.hurst_estimate, 3),
                    "noise_slope": round(psd.noise_slope, 3),
                    "noise_color": psd.noise_color,
                    "dominant_period_s": psd.peaks[0].period_s if psd.peaks else None,
                    "ou_halflife_s": acf.ou_halflife_s,
                    "ou_theta": acf.ou_theta,
                    "acf_first_zero_s": acf.first_zero_crossing_s,
                    "spectral_entropy": round(entropy, 4),
                    "persistent_at_horizon": bool(persistent),
                },
            ))

        return result.finalize(time.time() - t0)
