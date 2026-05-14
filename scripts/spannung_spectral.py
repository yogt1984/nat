#!/usr/bin/env python3
"""
Spannung Spectral Analysis — frequency-domain characterization of imbalance signal.

Answers: At what frequencies does imbalance oscillate, and at which
frequencies does it predict returns?

Analyses:
  1. PSD (Welch) — dominant oscillation periods, noise color, Hurst estimate
  2. Coherence + Phase — imbalance→returns at 1s/5s/30s/60s, phase lead in ms
  3. Autocorrelation — OU half-life, first zero crossing
  4. Band-pass filtered IC — which frequency bands carry predictive power
  5. Spectral entropy — concentrated (periodic) vs broadband (noisy)
  6. Market-making metrics — refresh rate, half-life, SNR, assessment

Usage:
    python scripts/spannung_spectral.py --data-dir data/features/2026-05-12
    python scripts/spannung_spectral.py --data-dir data/features/2026-05-12 --symbol BTC
    nat spannung spectral --data data/features/2026-05-12

Output:
    reports/spannung/spectral_{SYM}.json
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy import stats
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from cluster_pipeline.loader import load_parquet

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("spannung_spectral")

# ── Configuration ─────────────────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]

NEEDED = [
    "timestamp_ns", "symbol", "raw_midprice",
    "imbalance_qty_l1", "illiq_composite",
]

FS = 10.0            # sampling rate: 10 Hz (1 tick = 100ms)
NYQUIST = FS / 2.0   # 5 Hz

# Welch PSD
NPERSEG = 2048       # ~205s segments, freq resolution ~0.005 Hz
NOVERLAP = 1024      # 50% overlap

# Coherence horizons (ticks)
COHERENCE_HORIZONS = {
    "1s":  10,
    "5s":  50,
    "30s": 300,
    "60s": 600,
}

# Frequency bands for band-pass filtered IC
FREQUENCY_BANDS = [
    ("ultra_low",  0.005, 0.05),   # 20–200s periods
    ("low",        0.05,  0.5),    # 2–20s periods
    ("mid",        0.5,   2.0),    # 0.5–2s periods
    ("high",       2.0,   4.5),    # 0.22–0.5s periods
]

# Band IC horizons (ticks)
BAND_IC_HORIZONS = {"1s": 10, "5s": 50}

# ACF
ACF_MAX_LAG = 600    # 60 seconds

# IC windows (same as spannung_grid.py)
IC_WINDOW = 3000
IC_MIN_OBS = 100


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PSDPeak:
    frequency_hz: float
    period_s: float
    power_db: float


@dataclass
class PSDResult:
    peaks: List[PSDPeak]
    noise_slope: float
    noise_color: str
    hurst_estimate: float


@dataclass
class CoherencePeak:
    frequency_hz: float
    period_s: float
    coherence: float
    phase_rad: float
    phase_ms: float


@dataclass
class CoherenceResult:
    horizon_label: str
    horizon_ticks: int
    peaks: List[CoherencePeak]
    mean_coherence: float
    max_coherence: float


@dataclass
class ACFResult:
    first_zero_crossing_lag: int
    first_zero_crossing_s: float
    ou_halflife_ticks: float
    ou_halflife_s: float
    ou_theta: float
    acf_at_1s: float
    acf_at_5s: float
    acf_at_10s: float


@dataclass
class BandICResult:
    band_name: str
    low_hz: float
    high_hz: float
    horizon_label: str
    horizon_ticks: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    n_windows: int


@dataclass
class MarketMakingMetrics:
    dominant_period_s: float
    suggested_refresh_hz: float
    ou_halflife_s: float
    snr_predictive_db: float
    predictive_band: str
    predictive_band_ic: float
    spectral_entropy: float
    assessment: str


@dataclass
class SpectralResult:
    timestamp: str
    data_dir: str
    symbol: str
    n_rows: int
    duration_hours: float
    sampling_rate_hz: float
    psd: PSDResult
    coherence: List[CoherenceResult]
    acf: ACFResult
    band_ic: List[BandICResult]
    market_making: MarketMakingMetrics
    psd_freqs: List[float]
    psd_power_db: List[float]
    acf_values: List[float]


# ── Core analysis functions ──────────────────────────────────────────────────

def compute_psd(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, PSDResult]:
    """Welch PSD with peak detection and noise color characterization.

    Welch estimator
    ---------------
    Divides x[n] into K overlapping segments of length L (nperseg), applies a
    Hann window w[n], and averages the periodograms:

        P_xx(f) = (1/K) * sum_{k=0}^{K-1} (1/(L * U)) * |X_k(f)|^2

    where  X_k(f) = sum_{n=0}^{L-1} w[n] x_k[n] e^{-j2pi f n / fs}
    and    U = (1/L) sum_{n=0}^{L-1} w[n]^2  (window power normalisation).

    Output units: (signal units)^2 / Hz.  Converted to dB: 10*log10(P_xx).

    Noise slope / Hurst exponent
    ----------------------------
    For fractional Brownian motion (fBm), the theoretical PSD follows a power
    law:

        P(f) ~ f^{-(2H+1)}

    where H in [0,1] is the Hurst exponent.  Taking log10 of both sides:

        log10 P = -(2H+1) * log10 f + const

    so the OLS slope beta of (log10 P) ~ (log10 f) gives:

        H = -(beta + 1) / 2

    Interpretation: H < 0.5 => mean-reverting; H > 0.5 => trending;
                    H = 0.5 => Brownian (white-noise increments).

    Ref: Mandelbrot & Van Ness (1968), SIAM Review 10(4).
    """
    nperseg = min(NPERSEG, len(signal) // 4)
    noverlap = nperseg // 2

    freqs, pxx = sig.welch(signal, fs=FS, nperseg=nperseg, noverlap=noverlap)

    # Convert to dB: P_dB = 10 * log10(P_xx)
    pxx_db = 10 * np.log10(pxx + 1e-30)

    # Find peaks (skip DC at index 0)
    peak_idx, props = sig.find_peaks(pxx_db[1:], prominence=2.0)
    peak_idx += 1  # offset for skipped DC

    # Sort by power, take top 8
    if len(peak_idx) > 0:
        order = np.argsort(pxx_db[peak_idx])[::-1]
        peak_idx = peak_idx[order[:8]]
        peaks = [
            PSDPeak(
                frequency_hz=round(float(freqs[i]), 5),
                period_s=round(1.0 / freqs[i], 2) if freqs[i] > 0 else float("inf"),
                power_db=round(float(pxx_db[i]), 1),
            )
            for i in peak_idx
        ]
    else:
        peaks = []

    # Noise color: fit log-log slope beta on 0.05–4 Hz (avoids DC and near-Nyquist)
    # Model: log10 P = beta * log10 f + c  (OLS, degree-1 polynomial)
    # Color thresholds (conventional): white beta>-0.5, pink -1.5<beta<=-0.5, brown beta<=-1.5
    mask = (freqs >= 0.05) & (freqs <= 4.0) & (pxx > 0)
    if mask.sum() > 10:
        log_f = np.log10(freqs[mask])
        log_p = np.log10(pxx[mask])
        slope, _ = np.polyfit(log_f, log_p, 1)
    else:
        slope = 0.0

    slope = float(slope)
    if slope > -0.5:
        color = "white"
    elif slope > -1.5:
        color = "pink"
    else:
        color = "brown"

    # Hurst from power-law: P(f) ~ f^{-(2H+1)}  =>  H = -(beta+1)/2, clipped to [0,1]
    hurst = max(0.0, min(1.0, -(slope + 1) / 2))

    return freqs, pxx_db, PSDResult(
        peaks=peaks,
        noise_slope=round(slope, 3),
        noise_color=color,
        hurst_estimate=round(hurst, 3),
    )


def compute_coherence_phase(
    signal: np.ndarray,
    prices: np.ndarray,
    horizon_ticks: int,
    horizon_label: str,
) -> CoherenceResult:
    """Cross-coherence and phase between imbalance and forward returns.

    Forward log returns
    -------------------
    For horizon h (ticks), the forward log return at tick t is:

        r_t^h = log( p[t+h] / p[t] )

    Squared coherence
    -----------------
    Given the Welch cross-spectral density P_xy(f) and auto-spectral
    densities P_xx(f), P_yy(f), the magnitude-squared coherence is:

        C^2(f) = |P_xy(f)|^2 / ( P_xx(f) * P_yy(f) )

    C^2 in [0,1]; 0 = no linear relationship, 1 = perfect linear coupling.

    Phase spectrum
    --------------
    The cross-phase at frequency f (argument of P_xy, in radians):

        phi(f) = arg( P_xy(f) )

    Positive phi => x leads y by phi/(2*pi*f) seconds.
    Converted to milliseconds:

        phase_ms = phi(f) / (2*pi*f) * 1000

    Ref: Carter (1987), "Coherence and Time Delay Estimation", Proc. IEEE.
    """
    n = len(signal) - horizon_ticks
    if n < NPERSEG:
        return CoherenceResult(
            horizon_label=horizon_label, horizon_ticks=horizon_ticks,
            peaks=[], mean_coherence=0.0, max_coherence=0.0,
        )

    # r_t^h = log(p[t+h] / p[t])  — forward log return over horizon_ticks
    fwd_ret = np.log(prices[horizon_ticks:] / prices[:n])
    sig_trunc = signal[:n]

    # Remove NaN
    valid = np.isfinite(sig_trunc) & np.isfinite(fwd_ret)
    if valid.sum() < NPERSEG:
        return CoherenceResult(
            horizon_label=horizon_label, horizon_ticks=horizon_ticks,
            peaks=[], mean_coherence=0.0, max_coherence=0.0,
        )

    sig_clean = sig_trunc[valid]
    ret_clean = fwd_ret[valid]

    nperseg = min(NPERSEG, len(sig_clean) // 4)
    noverlap = nperseg // 2

    # C^2(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))  via Welch-averaged CSDs
    freqs, coh = sig.coherence(sig_clean, ret_clean, fs=FS,
                                nperseg=nperseg, noverlap=noverlap)

    # phi(f) = arg(P_xy(f))  — phase of cross-spectral density
    freqs_csd, pxy = sig.csd(sig_clean, ret_clean, fs=FS,
                              nperseg=nperseg, noverlap=noverlap)
    phase = np.angle(pxy)  # radians in (-pi, pi]

    # Find coherence peaks > 0.05
    peak_idx, _ = sig.find_peaks(coh[1:], height=0.05, prominence=0.02)
    peak_idx += 1

    if len(peak_idx) > 0:
        order = np.argsort(coh[peak_idx])[::-1]
        peak_idx = peak_idx[order[:6]]
        peaks = []
        for i in peak_idx:
            f = freqs[i]
            if f < 1e-6:
                continue
            # phase_ms = phi(f) / (2*pi*f) * 1000  [ms]; positive => imbalance leads returns
            phase_ms = float(phase[i] / (2 * np.pi * f) * 1000)
            peaks.append(CoherencePeak(
                frequency_hz=round(float(f), 4),
                period_s=round(1.0 / f, 2),
                coherence=round(float(coh[i]), 4),
                phase_rad=round(float(phase[i]), 4),
                phase_ms=round(phase_ms, 1),
            ))
    else:
        peaks = []

    coh_valid = coh[1:]  # skip DC
    return CoherenceResult(
        horizon_label=horizon_label,
        horizon_ticks=horizon_ticks,
        peaks=peaks,
        mean_coherence=round(float(np.mean(coh_valid)), 4),
        max_coherence=round(float(np.max(coh_valid)), 4),
    )


def compute_acf(signal: np.ndarray) -> Tuple[np.ndarray, ACFResult]:
    """FFT-based autocorrelation with OU process fit."""
    x = signal - np.nanmean(signal)
    x = np.nan_to_num(x, nan=0.0)
    n = len(x)

    # FFT-based autocorrelation
    nfft = 2 ** int(np.ceil(np.log2(2 * n)))
    fft_x = np.fft.rfft(x, n=nfft)
    acf_raw = np.fft.irfft(np.abs(fft_x) ** 2, n=nfft)[:ACF_MAX_LAG + 1]

    if acf_raw[0] > 0:
        acf = acf_raw / acf_raw[0]
    else:
        acf = np.zeros(ACF_MAX_LAG + 1)

    # First zero crossing
    zero_cross = ACF_MAX_LAG
    for k in range(1, ACF_MAX_LAG + 1):
        if acf[k] <= 0:
            zero_cross = k
            break

    # OU fit: acf(k) = exp(-theta * k)
    fit_end = min(zero_cross, 200)
    lags = np.arange(1, fit_end)
    acf_fit = acf[1:fit_end]

    try:
        # Only fit positive ACF values
        pos_mask = acf_fit > 0.01
        if pos_mask.sum() > 5:
            def ou_model(k, theta):
                return np.exp(-theta * k)
            popt, _ = curve_fit(ou_model, lags[pos_mask], acf_fit[pos_mask],
                                p0=[0.01], bounds=(1e-6, 1.0))
            theta = float(popt[0])
        else:
            theta = -np.log(max(acf[1], 0.01))
    except Exception:
        theta = -np.log(max(acf[1], 0.01))

    halflife_ticks = np.log(2) / max(theta, 1e-8)
    halflife_s = halflife_ticks * 0.1  # ticks to seconds

    return acf, ACFResult(
        first_zero_crossing_lag=int(zero_cross),
        first_zero_crossing_s=round(zero_cross * 0.1, 1),
        ou_halflife_ticks=round(halflife_ticks, 1),
        ou_halflife_s=round(halflife_s, 2),
        ou_theta=round(theta, 6),
        acf_at_1s=round(float(acf[10]) if len(acf) > 10 else 0, 4),
        acf_at_5s=round(float(acf[50]) if len(acf) > 50 else 0, 4),
        acf_at_10s=round(float(acf[100]) if len(acf) > 100 else 0, 4),
    )


def bandpass_filter(signal: np.ndarray, low_hz: float, high_hz: float) -> Optional[np.ndarray]:
    """Butterworth bandpass, zero-phase. Returns None if band is invalid."""
    low = max(low_hz, 0.005)
    high = min(high_hz, NYQUIST - 0.2)
    if low >= high or len(signal) < 500:
        return None
    try:
        sos = sig.butter(4, [low, high], btype="bandpass", fs=FS, output="sos")
        return sig.sosfiltfilt(sos, signal)
    except Exception:
        return None


def compute_rolling_ic(signal: np.ndarray, returns: np.ndarray) -> Tuple[float, float, float, int]:
    """Non-overlapping rolling Spearman IC. Returns (mean, std, ir, n_windows)."""
    ic_values = []
    start = 0
    n = min(len(signal), len(returns))
    while start + IC_WINDOW <= n:
        end = start + IC_WINDOW
        s_win = signal[start:end]
        r_win = returns[start:end]
        valid = np.isfinite(s_win) & np.isfinite(r_win)
        if valid.sum() >= IC_MIN_OBS:
            rho, _ = stats.spearmanr(s_win[valid], r_win[valid])
            if np.isfinite(rho):
                ic_values.append(float(rho))
        start = end

    if len(ic_values) < 2:
        return 0.0, 0.0, 0.0, 0
    arr = np.array(ic_values)
    m, s = float(np.mean(arr)), float(np.std(arr))
    ir = m / s if s > 1e-8 else 0.0
    return m, s, ir, len(ic_values)


def compute_band_ic(signal: np.ndarray, prices: np.ndarray) -> List[BandICResult]:
    """Band-pass filter imbalance into frequency bands, measure IC per band."""
    results = []
    for band_name, low_hz, high_hz in FREQUENCY_BANDS:
        filtered = bandpass_filter(signal, low_hz, high_hz)
        if filtered is None:
            continue
        for hz_label, hz_ticks in BAND_IC_HORIZONS.items():
            n = len(filtered) - hz_ticks
            if n < IC_WINDOW:
                continue
            fwd_ret = np.log(prices[hz_ticks:hz_ticks + n] / prices[:n])
            m, s, ir, nw = compute_rolling_ic(filtered[:n], fwd_ret)
            results.append(BandICResult(
                band_name=band_name,
                low_hz=low_hz,
                high_hz=high_hz,
                horizon_label=hz_label,
                horizon_ticks=hz_ticks,
                ic_mean=round(m, 4),
                ic_std=round(s, 4),
                ic_ir=round(ir, 2),
                n_windows=nw,
            ))
    return results


def compute_spectral_entropy(freqs: np.ndarray, pxx_db: np.ndarray) -> float:
    """Normalized Shannon entropy of PSD (0=concentrated, 1=uniform)."""
    # Work in linear power, skip DC
    pxx_lin = 10 ** (pxx_db[1:] / 10)
    total = np.sum(pxx_lin)
    if total <= 0:
        return 1.0
    p = pxx_lin / total
    h = -np.sum(p * np.log2(p + 1e-30))
    h_max = np.log2(len(p))
    return round(float(h / h_max) if h_max > 0 else 1.0, 4)


def compute_market_making_metrics(
    psd_result: PSDResult,
    acf_result: ACFResult,
    band_ics: List[BandICResult],
    spectral_entropy: float,
) -> MarketMakingMetrics:
    """Aggregate findings into market-making assessment."""
    # Dominant period from PSD
    if psd_result.peaks:
        dom_period = psd_result.peaks[0].period_s
    else:
        dom_period = 0.0

    refresh_hz = (2.0 / dom_period) if dom_period > 0 else 0.0

    # Best predictive band (highest |IC| at 5s horizon, fallback 1s)
    best_band = ""
    best_ic = 0.0
    for bic in band_ics:
        if bic.horizon_label == "5s" and abs(bic.ic_mean) > abs(best_ic):
            best_ic = bic.ic_mean
            best_band = bic.band_name
    if not best_band:
        for bic in band_ics:
            if abs(bic.ic_mean) > abs(best_ic):
                best_ic = bic.ic_mean
                best_band = bic.band_name

    # SNR: ratio of power in predictive band to total (approximate from PSD peaks)
    snr_db = -10.0  # default
    if psd_result.peaks and best_band:
        # Find band limits
        for bn, lo, hi in FREQUENCY_BANDS:
            if bn == best_band:
                peak_powers = [p.power_db for p in psd_result.peaks
                               if lo <= p.frequency_hz <= hi]
                if peak_powers:
                    snr_db = max(peak_powers) - psd_result.peaks[0].power_db
                break

    # Assessment
    parts = []
    if spectral_entropy < 0.85:
        parts.append(f"moderate spectral concentration (entropy={spectral_entropy:.2f})")
    else:
        parts.append(f"broadband spectrum (entropy={spectral_entropy:.2f})")

    if acf_result.ou_halflife_s < 10:
        parts.append(f"fast mean-reversion (OU half-life={acf_result.ou_halflife_s:.1f}s)")
    elif acf_result.ou_halflife_s < 30:
        parts.append(f"moderate mean-reversion (OU half-life={acf_result.ou_halflife_s:.1f}s)")
    else:
        parts.append(f"slow mean-reversion (OU half-life={acf_result.ou_halflife_s:.1f}s)")

    if abs(best_ic) > 0.05:
        parts.append(f"predictive {best_band} band (IC={best_ic:.3f})")
    elif abs(best_ic) > 0.02:
        parts.append(f"weak {best_band} band signal (IC={best_ic:.3f})")
    else:
        parts.append("no clear frequency-localized prediction")

    if psd_result.hurst_estimate < 0.4:
        parts.append(f"mean-reverting (H={psd_result.hurst_estimate:.2f})")
    elif psd_result.hurst_estimate > 0.6:
        parts.append(f"trending (H={psd_result.hurst_estimate:.2f})")
    else:
        parts.append(f"near-random-walk (H={psd_result.hurst_estimate:.2f})")

    assessment = "; ".join(parts)

    return MarketMakingMetrics(
        dominant_period_s=round(dom_period, 2),
        suggested_refresh_hz=round(refresh_hz, 3),
        ou_halflife_s=acf_result.ou_halflife_s,
        snr_predictive_db=round(snr_db, 1),
        predictive_band=best_band,
        predictive_band_ic=round(best_ic, 4),
        spectral_entropy=spectral_entropy,
        assessment=assessment,
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_spectral(df: pd.DataFrame, symbol: str, data_dir: str) -> SpectralResult:
    """Run all spectral analyses on one symbol."""
    signal = df["imbalance_qty_l1"].values.astype(np.float64)
    prices = df["raw_midprice"].values.astype(np.float64)

    # Forward-fill then backward-fill NaNs
    sig_series = pd.Series(signal).ffill().bfill()
    price_series = pd.Series(prices).ffill().bfill()
    signal = sig_series.values
    prices = price_series.values

    n = len(signal)
    ts = df["timestamp_ns"].values
    duration_h = (ts[-1] - ts[0]) / 1e9 / 3600

    log.info(f"    PSD ...")
    freqs, pxx_db, psd_result = compute_psd(signal)

    log.info(f"    Coherence ...")
    coh_results = []
    for label, h_ticks in COHERENCE_HORIZONS.items():
        coh_results.append(compute_coherence_phase(signal, prices, h_ticks, label))

    log.info(f"    ACF ...")
    acf_arr, acf_result = compute_acf(signal)

    log.info(f"    Band-filtered IC ...")
    band_ics = compute_band_ic(signal, prices)

    entropy = compute_spectral_entropy(freqs, pxx_db)

    mm = compute_market_making_metrics(psd_result, acf_result, band_ics, entropy)

    return SpectralResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        data_dir=data_dir,
        symbol=symbol,
        n_rows=n,
        duration_hours=round(duration_h, 1),
        sampling_rate_hz=FS,
        psd=psd_result,
        coherence=coh_results,
        acf=acf_result,
        band_ic=band_ics,
        market_making=mm,
        psd_freqs=[round(float(f), 5) for f in freqs],
        psd_power_db=[round(float(p), 2) for p in pxx_db],
        acf_values=[round(float(a), 5) for a in acf_arr],
    )


# ── Display ──────────────────────────────────────────────────────────────────

def print_spectral(result: SpectralResult):
    """Print formatted spectral summary."""
    W, BOLD = "\033[0m", "\033[1m"
    G, Y, R = "\033[32m", "\033[33m", "\033[31m"

    w = 90
    print(f"\n{'=' * w}")
    print(f"  {BOLD}SPECTRAL ANALYSIS — {result.symbol}{W}"
          f"  ({result.n_rows:,} rows, {result.duration_hours:.1f}h, fs={result.sampling_rate_hz}Hz)")
    print(f"{'=' * w}")

    # ── PSD Peaks ──
    psd = result.psd
    print(f"\n  {BOLD}1. Power Spectral Density{W}")
    print(f"  {'─' * 60}")
    if psd.peaks:
        print(f"  {'#':>3}  {'period':>10}  {'freq(Hz)':>10}  {'power(dB)':>10}")
        print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}")
        for i, p in enumerate(psd.peaks):
            period_str = f"{p.period_s:.1f}s" if p.period_s < 200 else f"{p.period_s/60:.1f}min"
            print(f"  {i+1:>3}  {period_str:>10}  {p.frequency_hz:>10.4f}  {p.power_db:>10.1f}")
    else:
        print(f"  No significant PSD peaks found")
    print(f"\n  Noise: slope={psd.noise_slope:.2f} ({psd.noise_color})"
          f"  Hurst={psd.hurst_estimate:.3f}"
          f" ({'mean-reverting' if psd.hurst_estimate < 0.4 else 'trending' if psd.hurst_estimate > 0.6 else 'random-walk'})")

    # ── Coherence ──
    print(f"\n  {BOLD}2. Coherence (imbalance → forward returns){W}")
    print(f"  {'─' * 60}")
    for cr in result.coherence:
        if cr.peaks:
            print(f"\n  horizon={cr.horizon_label}  (mean_coh={cr.mean_coherence:.3f}, max={cr.max_coherence:.3f})")
            print(f"    {'freq(Hz)':>10}  {'period':>8}  {'coherence':>10}  {'phase(ms)':>10}")
            print(f"    {'─'*10}  {'─'*8}  {'─'*10}  {'─'*10}")
            for cp in cr.peaks[:5]:
                period_str = f"{cp.period_s:.1f}s" if cp.period_s < 200 else f"{cp.period_s/60:.1f}min"
                print(f"    {cp.frequency_hz:>10.4f}  {period_str:>8}  {cp.coherence:>10.4f}  {cp.phase_ms:>+10.1f}")
        else:
            print(f"\n  horizon={cr.horizon_label}  — no significant coherence peaks")

    # ── ACF ──
    acf = result.acf
    print(f"\n  {BOLD}3. Autocorrelation{W}")
    print(f"  {'─' * 60}")
    print(f"  First zero crossing: lag {acf.first_zero_crossing_lag}"
          f" ({acf.first_zero_crossing_s}s)")
    print(f"  OU half-life: {acf.ou_halflife_s:.2f}s"
          f" ({acf.ou_halflife_ticks:.0f} ticks, theta={acf.ou_theta:.6f})")
    print(f"  ACF(1s)={acf.acf_at_1s:.4f}"
          f"  ACF(5s)={acf.acf_at_5s:.4f}"
          f"  ACF(10s)={acf.acf_at_10s:.4f}")

    # ── Band IC ──
    print(f"\n  {BOLD}4. Band-Filtered IC{W}")
    print(f"  {'─' * 60}")
    print(f"  {'band':>12}  {'freq range':>14}  {'horizon':>8}  {'IC':>8}  {'IC_IR':>7}  {'windows':>8}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}")
    for bic in result.band_ic:
        ic_color = G if abs(bic.ic_mean) > 0.05 else Y if abs(bic.ic_mean) > 0.02 else W
        print(f"  {bic.band_name:>12}  {bic.low_hz:>5.3f}–{bic.high_hz:<5.1f}Hz"
              f"  {bic.horizon_label:>8}"
              f"  {ic_color}{bic.ic_mean:>+8.4f}{W}"
              f"  {bic.ic_ir:>7.2f}"
              f"  {bic.n_windows:>8}")

    # ── Market-Making Assessment ──
    mm = result.market_making
    print(f"\n  {BOLD}5. Market-Making Assessment{W}")
    print(f"  {'─' * 60}")
    print(f"  Spectral entropy:      {mm.spectral_entropy:.3f}"
          f" ({'concentrated' if mm.spectral_entropy < 0.85 else 'broadband'})")
    print(f"  Dominant period:       {mm.dominant_period_s:.1f}s"
          f" → refresh at {mm.suggested_refresh_hz:.3f} Hz")
    print(f"  Mean-reversion:        OU half-life = {mm.ou_halflife_s:.2f}s")
    print(f"  Predictive band:       {mm.predictive_band}"
          f" (IC={mm.predictive_band_ic:+.4f},"
          f" SNR={mm.snr_predictive_db:+.1f} dB)")
    print(f"\n  {BOLD}Assessment:{W} {mm.assessment}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spannung spectral analysis")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to parquet data")
    parser.add_argument("--symbol", type=str, default="all", help='Symbol or "all"')
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = Path(args.output) if args.output else ROOT / "reports" / "spannung"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = SYMBOLS if args.symbol == "all" else [args.symbol.upper()]

    for sym in symbols:
        log.info(f"  Loading {sym} from {data_dir} ...")
        df = load_parquet(str(data_dir), symbols=[sym], columns=NEEDED)
        if df.empty:
            log.warning(f"  No data for {sym}, skipping")
            continue
        df = df.sort_values("timestamp_ns").reset_index(drop=True)
        log.info(f"  {len(df):,} rows, running spectral analysis ...")

        t0 = time.time()
        result = run_spectral(df, sym, data_dir)
        elapsed = time.time() - t0
        log.info(f"    Done in {elapsed:.1f}s")

        # Save JSON
        result_path = out_dir / f"spectral_{sym}.json"
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        log.info(f"  Saved: {result_path}")

        # Print
        print_spectral(result)

        del df


if __name__ == "__main__":
    main()
