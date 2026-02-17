"""
Data Collector for NAT Validation

Collects order book snapshots and trades from Hyperliquid via WebSocket.
Computes features in real-time and saves to Parquet.

Usage:
    python data_collector.py --duration 3600  # Collect for 1 hour
    python data_collector.py --duration 86400 # Collect for 24 hours
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import argparse

import numpy as np
import pandas as pd
import websockets
from tqdm import tqdm

from config import (
    HYPERLIQUID_WS_URL, ASSETS, FEATURES_DIR,
    EMISSION_INTERVAL_MS, BOOK_LEVELS, TRADE_BUFFER_SECONDS
)


@dataclass
class PriceLevel:
    price: float
    size: float


@dataclass
class Trade:
    timestamp_ns: int
    price: float
    size: float
    side: str  # 'buy' or 'sell'


@dataclass
class OrderBook:
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    timestamp_ns: int = 0

    @property
    def mid_price(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000


class FeatureComputer:
    """Compute features from order book and trade data."""

    def __init__(self, buffer_size: int = 1000):
        self.price_buffer = deque(maxlen=buffer_size)
        self.return_buffer = deque(maxlen=buffer_size)
        self.trade_buffer: List[Trade] = []
        self.imbalance_buffer = deque(maxlen=buffer_size)
        self.entropy_buffer = deque(maxlen=600)  # 1 minute at 100ms
        self.last_price = 0.0

    def update_book(self, book: OrderBook):
        """Update with new order book snapshot."""
        mid = book.mid_price
        if mid > 0:
            if self.last_price > 0:
                ret = (mid - self.last_price) / self.last_price
                self.return_buffer.append(ret)
            self.price_buffer.append(mid)
            self.last_price = mid

            # Compute and store imbalance
            imb = self._compute_imbalance(book)
            self.imbalance_buffer.append(imb)

    def update_trades(self, trades: List[Trade]):
        """Update with new trades."""
        current_time = time.time_ns()
        cutoff = current_time - (TRADE_BUFFER_SECONDS * 1_000_000_000)

        # Add new trades
        self.trade_buffer.extend(trades)

        # Remove old trades
        self.trade_buffer = [t for t in self.trade_buffer if t.timestamp_ns > cutoff]

    def _compute_imbalance(self, book: OrderBook, levels: int = 5) -> float:
        """Compute order book imbalance at given depth."""
        bid_vol = sum(l.size for l in book.bids[:levels]) if book.bids else 0
        ask_vol = sum(l.size for l in book.asks[:levels]) if book.asks else 0
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _permutation_entropy(self, data: List[float], order: int = 3) -> float:
        """Compute permutation entropy."""
        if len(data) < order:
            return 0.0

        from math import factorial
        n_patterns = factorial(order)
        pattern_counts = {}

        for i in range(len(data) - order + 1):
            window = data[i:i+order]
            # Get ordinal pattern
            pattern = tuple(sorted(range(order), key=lambda x: window[x]))
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        total = sum(pattern_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in pattern_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)

        max_entropy = np.log(n_patterns)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _distribution_entropy(self, data: List[float], n_bins: int = 10) -> float:
        """Compute entropy of distribution."""
        if len(data) < 5:
            return 0.0

        arr = np.array(data)
        if arr.std() < 1e-10:
            return 0.0

        counts, _ = np.histogram(arr, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(n_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def compute_features(self, book: OrderBook) -> Optional[Dict[str, float]]:
        """Compute all features."""
        if len(self.return_buffer) < 32:
            return None

        returns = list(self.return_buffer)

        features = {}

        # === RAW FEATURES ===
        features['raw_mid_price'] = book.mid_price
        features['raw_spread_bps'] = book.spread_bps
        features['raw_bid_depth_l5'] = sum(l.size for l in book.bids[:5])
        features['raw_ask_depth_l5'] = sum(l.size for l in book.asks[:5])

        # === IMBALANCE FEATURES ===
        features['imbalance_l5'] = self._compute_imbalance(book, 5)
        features['imbalance_l10'] = self._compute_imbalance(book, 10)

        # Imbalance persistence (autocorrelation)
        if len(self.imbalance_buffer) >= 50:
            imb_arr = np.array(list(self.imbalance_buffer)[-50:])
            features['imbalance_persistence'] = np.corrcoef(imb_arr[:-5], imb_arr[5:])[0, 1]
        else:
            features['imbalance_persistence'] = 0.0

        # === TRADE FLOW FEATURES ===
        recent_trades = [t for t in self.trade_buffer
                        if t.timestamp_ns > time.time_ns() - 5_000_000_000]  # Last 5s

        if recent_trades:
            buy_vol = sum(t.size for t in recent_trades if t.side == 'buy')
            sell_vol = sum(t.size for t in recent_trades if t.side == 'sell')
            total_vol = buy_vol + sell_vol
            features['flow_aggressor_ratio'] = buy_vol / total_vol if total_vol > 0 else 0.5
            features['flow_volume_5s'] = total_vol
            features['flow_trade_count_5s'] = len(recent_trades)
        else:
            features['flow_aggressor_ratio'] = 0.5
            features['flow_volume_5s'] = 0.0
            features['flow_trade_count_5s'] = 0

        # Aggressor momentum
        if len(self.trade_buffer) >= 10:
            recent_aggressor = [1 if t.side == 'buy' else -1 for t in self.trade_buffer[-10:]]
            features['flow_aggressor_momentum'] = np.mean(recent_aggressor)
        else:
            features['flow_aggressor_momentum'] = 0.0

        # === VOLATILITY FEATURES ===
        ret_arr = np.array(returns[-100:])
        features['vol_realized_100'] = np.std(ret_arr) * np.sqrt(10 * 3600)  # Annualized approx

        if len(returns) >= 20:
            features['vol_realized_20'] = np.std(returns[-20:]) * np.sqrt(10 * 3600)
        else:
            features['vol_realized_20'] = 0.0

        # Volatility ratio (short/long)
        if features['vol_realized_100'] > 0:
            features['vol_ratio'] = features['vol_realized_20'] / features['vol_realized_100']
        else:
            features['vol_ratio'] = 1.0

        # === ENTROPY FEATURES (CORE) ===
        features['ent_permutation_8'] = self._permutation_entropy(returns[-8:], 3)
        features['ent_permutation_16'] = self._permutation_entropy(returns[-16:], 3)
        features['ent_permutation_32'] = self._permutation_entropy(returns[-32:], 3)

        # Book shape entropy
        if book.bids and book.asks:
            depths = [l.size for l in book.bids[:10]] + [l.size for l in book.asks[:10]]
            features['ent_book_shape'] = self._distribution_entropy(depths, 10)
        else:
            features['ent_book_shape'] = 0.0

        # Trade size entropy
        if recent_trades:
            trade_sizes = [t.size for t in recent_trades]
            features['ent_trade_size'] = self._distribution_entropy(trade_sizes, 5)
        else:
            features['ent_trade_size'] = 0.0

        # Store current entropy for rate of change calculation
        current_entropy = features['ent_permutation_16']
        self.entropy_buffer.append(current_entropy)

        # Entropy rate of change (vs 5 seconds ago = 50 samples)
        if len(self.entropy_buffer) >= 50:
            features['ent_rate_of_change'] = current_entropy - self.entropy_buffer[-50]
        else:
            features['ent_rate_of_change'] = 0.0

        # Entropy z-score vs 1-minute mean
        if len(self.entropy_buffer) >= 100:
            ent_arr = np.array(list(self.entropy_buffer)[-600:])
            mean_ent = np.mean(ent_arr)
            std_ent = np.std(ent_arr)
            if std_ent > 0:
                features['ent_zscore'] = (current_entropy - mean_ent) / std_ent
            else:
                features['ent_zscore'] = 0.0
        else:
            features['ent_zscore'] = 0.0

        # === COMPOSITE SIGNAL ===
        # Combine entropy + imbalance persistence + aggressor momentum
        # This is the key validation target
        features['composite_regime_signal'] = (
            0.4 * (1 - features['ent_permutation_16']) +  # Low entropy = trending
            0.3 * abs(features['imbalance_persistence']) +
            0.3 * abs(features['flow_aggressor_momentum'])
        )

        return features


class DataCollector:
    """Collect data from Hyperliquid WebSocket."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.computer = FeatureComputer()
        self.book = OrderBook()
        self.features_list: List[Dict] = []
        self.connected = False

    async def connect_and_collect(self, duration_seconds: int):
        """Connect to WebSocket and collect data for specified duration."""
        print(f"[{self.symbol}] Connecting to Hyperliquid WebSocket...")

        end_time = time.time() + duration_seconds
        pbar = tqdm(total=duration_seconds, desc=f"{self.symbol}", unit="s")
        last_update = time.time()

        async with websockets.connect(HYPERLIQUID_WS_URL) as ws:
            # Subscribe to order book
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {
                    "type": "l2Book",
                    "coin": self.symbol
                }
            }))

            # Subscribe to trades
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {
                    "type": "trades",
                    "coin": self.symbol
                }
            }))

            print(f"[{self.symbol}] Subscribed, collecting data...")
            self.connected = True

            last_emit = time.time()

            while time.time() < end_time:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(msg)

                    self._process_message(data)

                    # Emit features at regular interval
                    now = time.time()
                    if (now - last_emit) * 1000 >= EMISSION_INTERVAL_MS:
                        features = self.computer.compute_features(self.book)
                        if features:
                            features['timestamp'] = datetime.utcnow().isoformat()
                            features['timestamp_ns'] = time.time_ns()
                            features['symbol'] = self.symbol
                            self.features_list.append(features)
                        last_emit = now

                    # Update progress
                    if now - last_update >= 1.0:
                        pbar.update(int(now - last_update))
                        last_update = now

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"[{self.symbol}] Error: {e}")
                    await asyncio.sleep(1)

        pbar.close()
        print(f"[{self.symbol}] Collection complete: {len(self.features_list)} samples")

    def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message."""
        if 'channel' not in data:
            return

        channel = data['channel']

        if channel == 'l2Book':
            self._update_book(data.get('data', {}))
        elif channel == 'trades':
            self._update_trades(data.get('data', []))

    def _update_book(self, book_data: Dict):
        """Update order book from message."""
        if 'levels' not in book_data:
            return

        levels = book_data['levels']

        self.book.bids = []
        self.book.asks = []
        self.book.timestamp_ns = time.time_ns()

        for level in levels:
            px = float(level.get('px', 0))
            sz = float(level.get('sz', 0))
            side = level.get('side', '')

            if side == 'B':
                self.book.bids.append(PriceLevel(px, sz))
            elif side == 'A':
                self.book.asks.append(PriceLevel(px, sz))

        # Sort
        self.book.bids.sort(key=lambda x: -x.price)
        self.book.asks.sort(key=lambda x: x.price)

        # Update computer
        self.computer.update_book(self.book)

    def _update_trades(self, trades_data: List[Dict]):
        """Update trades from message."""
        trades = []
        for t in trades_data:
            trade = Trade(
                timestamp_ns=int(t.get('time', time.time_ns())),
                price=float(t.get('px', 0)),
                size=float(t.get('sz', 0)),
                side='buy' if t.get('side', 'B') == 'B' else 'sell'
            )
            trades.append(trade)

        self.computer.update_trades(trades)

    def save_to_parquet(self, output_dir: Path):
        """Save collected features to Parquet."""
        if not self.features_list:
            print(f"[{self.symbol}] No data to save")
            return None

        df = pd.DataFrame(self.features_list)

        output_path = output_dir / f"{self.symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"[{self.symbol}] Saved {len(df)} rows to {output_path}")
        return output_path


async def collect_all_symbols(duration_seconds: int, symbols: List[str]):
    """Collect data for all symbols in parallel."""
    collectors = [DataCollector(symbol) for symbol in symbols]

    tasks = [
        collector.connect_and_collect(duration_seconds)
        for collector in collectors
    ]

    await asyncio.gather(*tasks)

    # Save all data
    output_files = []
    for collector in collectors:
        path = collector.save_to_parquet(FEATURES_DIR)
        if path:
            output_files.append(path)

    return output_files


def main():
    parser = argparse.ArgumentParser(description='Collect NAT validation data')
    parser.add_argument('--duration', type=int, default=3600,
                       help='Collection duration in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--symbols', nargs='+', default=ASSETS,
                       help=f'Symbols to collect (default: {ASSETS})')
    args = parser.parse_args()

    print("=" * 60)
    print("NAT Week 1-2 Validation - Data Collection")
    print("=" * 60)
    print(f"Duration: {args.duration} seconds ({args.duration/3600:.1f} hours)")
    print(f"Symbols: {args.symbols}")
    print(f"Output: {FEATURES_DIR}")
    print("=" * 60)

    asyncio.run(collect_all_symbols(args.duration, args.symbols))

    print("\nData collection complete!")
    print(f"Next step: Run 'python regime_labeler.py' to label regimes")


if __name__ == '__main__':
    main()
