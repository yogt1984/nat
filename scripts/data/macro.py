"""
Macro data fetcher — daily/weekly OHLCV from public APIs.

No API key needed. Uses Hyperliquid's own candle endpoint.
Falls back to Binance public API.

Usage:
    from data.macro import fetch_candles
    df = fetch_candles("ETH", interval="1d", days=365)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import urllib.request

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "macro"


def fetch_candles(
    symbol: str,
    interval: str = "1d",
    days: int = 365,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Hyperliquid.

    Args:
        symbol: e.g. "ETH", "BTC", "SOL"
        interval: "1h", "4h", "1d", "1w"
        days: how many days back
        use_cache: cache to data/macro/ (refreshes if >1h old)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{days}d.parquet"

    # Use cache if fresh
    if use_cache and cache_file.exists():
        age_s = time.time() - cache_file.stat().st_mtime
        if age_s < 3600:  # 1 hour
            return pd.read_parquet(cache_file)

    # Try Hyperliquid first
    df = _fetch_hyperliquid(symbol, interval, days)
    if df is None or len(df) == 0:
        df = _fetch_binance(symbol, interval, days)

    if df is not None and len(df) > 0:
        df.to_parquet(cache_file, index=False)

    return df


def _fetch_hyperliquid(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fetch from Hyperliquid info API."""
    try:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000

        payload = json.dumps({
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            }
        }).encode()

        req = urllib.request.Request(
            "https://api.hyperliquid.xyz/info",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        if not data or not isinstance(data, list):
            return None

        rows = []
        for c in data:
            rows.append({
                "timestamp": pd.to_datetime(c["t"], unit="ms", utc=True),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df

    except Exception:
        return None


def _fetch_binance(symbol: str, interval: str, days: int) -> Optional[pd.DataFrame]:
    """Fallback: Binance public klines API (no auth needed)."""
    try:
        # Map interval
        iv_map = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
        bi = iv_map.get(interval, "1d")
        pair = f"{symbol}USDT"

        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000

        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={pair}&interval={bi}"
            f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
        )

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        rows = []
        for k in data:
            rows.append({
                "timestamp": pd.to_datetime(k[0], unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    except Exception:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard macro indicators to OHLCV data.

    Adds:
        - Moving averages: SMA 7, 21, 50, 100, 200
        - MA crossover signals
        - RSI (14)
        - Support/resistance (rolling high/low)
        - Psychological levels (round numbers)
        - Volume profile (relative volume)
    """
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)

    # ── Moving Averages ──
    for w in [7, 21, 50, 100, 200]:
        df[f"sma_{w}"] = pd.Series(c).rolling(w).mean().values

    # EMA
    for w in [12, 26]:
        df[f"ema_{w}"] = pd.Series(c).ewm(span=w, adjust=False).mean().values

    # ── MA Crossovers ──
    # Golden/death cross
    df["cross_50_200"] = np.where(df["sma_50"] > df["sma_200"], 1, -1)
    # Short-term trend
    df["cross_7_21"] = np.where(df["sma_7"] > df["sma_21"], 1, -1)
    # MACD-style
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = pd.Series(df["macd"]).ewm(span=9, adjust=False).mean().values
    df["macd_cross"] = np.where(df["macd"] > df["macd_signal"], 1, -1)

    # Price vs MAs (distance in %)
    for w in [50, 200]:
        sma = df[f"sma_{w}"].values
        df[f"dist_sma_{w}_pct"] = np.where(sma > 0, (c - sma) / sma * 100, np.nan)

    # ── RSI ──
    df["rsi_14"] = _rsi(c, 14)

    # ── Support / Resistance ──
    for w in [20, 50]:
        df[f"resist_{w}"] = pd.Series(h).rolling(w).max().values
        df[f"support_{w}"] = pd.Series(lo).rolling(w).min().values

    # Distance to support/resistance (%)
    df["dist_resist_pct"] = (df["resist_20"] - c) / c * 100
    df["dist_support_pct"] = (c - df["support_20"]) / c * 100

    # ── Psychological Levels ──
    # Nearest round number (100 for ETH, 1000 for BTC)
    magnitude = 10 ** max(0, int(np.log10(np.nanmedian(c))) - 1)
    nearest_round = np.round(c / magnitude) * magnitude
    df["dist_psych_pct"] = (c - nearest_round) / c * 100
    df["psych_level"] = nearest_round

    # ── Volume Profile ──
    vol_sma = pd.Series(v).rolling(20).mean().values
    df["rel_volume"] = np.where(vol_sma > 0, v / vol_sma, 1.0)

    # ── Trend Strength ──
    # ADX approximation via directional movement
    df["atr_14"] = _atr(h, lo, c, 14)
    df["atr_pct"] = df["atr_14"] / c * 100

    return df


def get_macro_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate macro trading signals from indicator data.

    Returns DataFrame with:
        - direction: +1 (bullish) / -1 (bearish) / 0 (neutral)
        - strength: 0.0 to 1.0 (conviction)
        - regime: "trending_up", "trending_down", "ranging", "breakout"
    """
    n = len(df)
    direction = np.zeros(n)
    strength = np.zeros(n)
    regime = ["unknown"] * n

    for i in range(200, n):
        score = 0.0
        factors = 0

        # MA alignment (strongest signal)
        if not np.isnan(df["sma_50"].iloc[i]) and not np.isnan(df["sma_200"].iloc[i]):
            if df["cross_50_200"].iloc[i] == 1:
                score += 2.0
            else:
                score -= 2.0
            factors += 2

        # Short-term trend
        if not np.isnan(df["sma_7"].iloc[i]):
            score += df["cross_7_21"].iloc[i]
            factors += 1

        # MACD
        score += df["macd_cross"].iloc[i]
        factors += 1

        # RSI
        rsi = df["rsi_14"].iloc[i]
        if not np.isnan(rsi):
            if rsi > 70:
                score -= 0.5  # overbought
            elif rsi < 30:
                score += 0.5  # oversold
            factors += 0.5

        # Price vs SMA200
        dist = df["dist_sma_200_pct"].iloc[i] if "dist_sma_200_pct" in df.columns else 0
        if not np.isnan(dist):
            if dist > 20:
                score -= 0.5  # extended
            elif dist < -20:
                score += 0.5  # deep value

        if factors > 0:
            normalized = score / factors
            direction[i] = np.sign(normalized)
            strength[i] = min(1.0, abs(normalized))

        # Classify regime
        atr_pct = df["atr_pct"].iloc[i] if "atr_pct" in df.columns else 0
        if abs(score) >= 3:
            regime[i] = "trending_up" if score > 0 else "trending_down"
        elif not np.isnan(atr_pct) and atr_pct > 5:
            regime[i] = "breakout"
        else:
            regime[i] = "ranging"

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "close": df["close"],
        "direction": direction,
        "strength": strength,
        "regime": regime,
    })


def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index."""
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = pd.Series(gains).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, adjust=False).mean().values

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-15, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    return pd.Series(tr).rolling(period).mean().values
