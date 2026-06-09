#!/usr/bin/env python3
"""
Limit Order Simulator — Directional and Market-Making Modes

Two simulation modes:
  directional: signal → entry fill → hold → exit fill → PnL
  mm:          simultaneous bid+ask, spread capture, inventory management

Fill model (both modes): conservative mid-cross on 100ms tick data.

Usage:
    python scripts/kalman/limit_order_sim.py --symbol BTC
    python scripts/kalman/limit_order_sim.py --all-symbols --save
    python scripts/kalman/limit_order_sim.py --sweep
    python scripts/kalman/limit_order_sim.py --mode mm --all-symbols
    python scripts/kalman/limit_order_sim.py --mode mm --max-inventory 3 --requote 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data.features import available_dates, load_features
from utils.costs import maker_bps, taker_bps
from utils.metrics import sharpe_daily

# ---------------------------------------------------------------------------
# Config & data classes
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "timestamp_ns", "symbol", "raw_midprice", "raw_spread",
    "raw_spread_bps", "imbalance_qty_l1", "ent_book_shape",
    "raw_bid_depth_5", "raw_ask_depth_5",
]
SYMBOLS = ["BTC", "ETH", "SOL"]
TRAIN_WINDOW = 3


@dataclass
class SimConfig:
    entry_threshold: float = 0.3
    regime_feature: str = "ent_book_shape"
    regime_percentile: float = 30.0
    latency_ticks: int = 2
    entry_timeout_ticks: int = 50
    exit_timeout_ticks: int = 100
    min_ticks_between: int = 20
    stop_loss_bps: float = 5.0
    signal_cancel: bool = True
    fill_model: str = "mid_cross"
    fill_prob_scale: float = 0.7
    maker_fee_bps: float = field(default_factory=maker_bps)
    taker_fee_bps: float = field(default_factory=taker_bps)


@dataclass
class RoundTrip:
    signal_tick: int
    entry_tick: int
    exit_tick: int
    side: str
    entry_price: float
    exit_price: float
    exit_type: str
    exit_reason: str
    signal_strength: float
    gross_pnl_bps: float
    fee_bps: float
    net_pnl_bps: float
    holding_ticks: int


@dataclass
class DateResult:
    date: str
    symbol: str
    n_signals: int
    n_entry_fills: int
    n_round_trips: int
    n_maker_exits: int
    n_taker_exits: int
    fill_rate: float
    maker_exit_rate: float
    mean_gross_bps: float
    mean_net_bps: float
    total_net_bps: float
    mean_holding_ticks: float
    mean_spread_bps: float


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------

def simulate_day(
    midprices: np.ndarray,
    spreads: np.ndarray,
    spread_bps: np.ndarray,
    signal: np.ndarray,
    regime_vals: np.ndarray,
    regime_thresh: float,
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
) -> tuple[list[RoundTrip], int]:
    """Simulate full round-trip trades for one day of tick data.

    Returns (trades, n_signals).
    """
    n = len(midprices)
    trades: list[RoundTrip] = []
    n_signals = 0
    last_trade_tick = -cfg.min_ticks_between
    prob_adjusted = cfg.fill_model == "prob_adjusted"
    if prob_adjusted and rng is None:
        rng = np.random.default_rng(42)

    i = 0
    while i < n - cfg.latency_ticks - 1:
        # --- FLAT: look for entry signal ---
        if not np.isfinite(signal[i]):
            i += 1
            continue
        if not np.isfinite(regime_vals[i]) or regime_vals[i] >= regime_thresh:
            i += 1
            continue
        if i - last_trade_tick < cfg.min_ticks_between:
            i += 1
            continue

        # Determine side
        if signal[i] > cfg.entry_threshold:
            side = "buy"
            direction = 1.0
        elif signal[i] < -cfg.entry_threshold:
            side = "sell"
            direction = -1.0
        else:
            i += 1
            continue

        n_signals += 1
        signal_tick = i
        signal_strength = float(signal[i])

        # Order placed after latency
        order_tick = i + cfg.latency_ticks
        if order_tick >= n:
            break

        sp = spreads[order_tick]
        if not np.isfinite(sp) or sp <= 0:
            i += 1
            continue

        # Order price at best bid (buy) or best ask (sell)
        if side == "buy":
            order_price = midprices[order_tick] - sp / 2
        else:
            order_price = midprices[order_tick] + sp / 2

        # --- WAITING_ENTRY: scan for fill ---
        entry_filled = False
        entry_tick = -1
        entry_end = min(order_tick + cfg.entry_timeout_ticks, n)

        for j in range(order_tick + 1, entry_end):
            # Signal cancel: signal reversed or dropped below threshold
            if cfg.signal_cancel and np.isfinite(signal[j]):
                if side == "buy" and signal[j] < 0:
                    break
                if side == "sell" and signal[j] > 0:
                    break

            # Fill check
            filled_this_tick = False
            if side == "buy" and midprices[j] <= order_price:
                filled_this_tick = True
            elif side == "sell" and midprices[j] >= order_price:
                filled_this_tick = True

            if filled_this_tick:
                if prob_adjusted and rng.random() > cfg.fill_prob_scale:
                    continue  # probabilistic rejection
                entry_tick = j
                entry_filled = True
                break

        if not entry_filled:
            i = entry_end
            continue

        # --- IN_POSITION: manage exit ---
        entry_price = order_price

        # Exit limit on opposite side, priced at tick after entry
        exit_price_tick = min(entry_tick + 1, n - 1)
        sp_exit = spreads[exit_price_tick]
        if not np.isfinite(sp_exit) or sp_exit <= 0:
            sp_exit = sp  # fallback to entry spread

        if side == "buy":
            exit_order_price = midprices[exit_price_tick] + sp_exit / 2
        else:
            exit_order_price = midprices[exit_price_tick] - sp_exit / 2

        exit_tick = -1
        exit_type = "taker"
        exit_reason = "timeout"
        exit_price = 0.0
        exit_end = min(entry_tick + cfg.exit_timeout_ticks, n)

        for j in range(entry_tick + 1, exit_end):
            # Stop loss check
            unrealized_bps = direction * (midprices[j] - entry_price) / entry_price * 10000
            if unrealized_bps < -cfg.stop_loss_bps:
                exit_tick = j
                exit_price = midprices[j]
                exit_type = "taker"
                exit_reason = "stop_loss"
                break

            # Maker exit fill
            if side == "buy" and midprices[j] >= exit_order_price:
                exit_tick = j
                exit_price = exit_order_price
                exit_type = "maker"
                exit_reason = "maker_fill"
                break
            elif side == "sell" and midprices[j] <= exit_order_price:
                exit_tick = j
                exit_price = exit_order_price
                exit_type = "maker"
                exit_reason = "maker_fill"
                break

        # Timeout or end-of-day: taker exit
        if exit_tick < 0:
            exit_tick = exit_end - 1 if exit_end <= n else n - 1
            exit_price = midprices[exit_tick]
            exit_type = "taker"
            exit_reason = "timeout"

        # PnL
        gross_pnl_bps = direction * (exit_price - entry_price) / entry_price * 10000

        # Fees: entry is always maker (rebate), exit depends on type
        entry_fee = -cfg.maker_fee_bps  # rebate = negative cost
        if exit_type == "maker":
            exit_fee = -cfg.maker_fee_bps
        else:
            exit_fee = cfg.taker_fee_bps
        total_fee = entry_fee + exit_fee
        net_pnl_bps = gross_pnl_bps - total_fee

        holding_ticks = exit_tick - entry_tick

        trades.append(RoundTrip(
            signal_tick=signal_tick,
            entry_tick=entry_tick,
            exit_tick=exit_tick,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_type=exit_type,
            exit_reason=exit_reason,
            signal_strength=signal_strength,
            gross_pnl_bps=round(gross_pnl_bps, 4),
            fee_bps=round(total_fee, 4),
            net_pnl_bps=round(net_pnl_bps, 4),
            holding_ticks=holding_ticks,
        ))

        last_trade_tick = exit_tick
        i = exit_tick + 1

    return trades, n_signals


# ---------------------------------------------------------------------------
# Market-Making simulator
# ---------------------------------------------------------------------------

@dataclass
class MMConfig:
    max_inventory: int = 3
    requote_ticks: int = 10          # refresh orders every N ticks
    latency_ticks: int = 2
    skew_factor: float = 0.0         # 0 = symmetric, >0 = skew away from inventory
    skew_signal: bool = True         # use imbalance to skew quotes
    skew_signal_mult: float = 0.5    # how much signal moves quote (in spreads)
    stop_loss_bps: float = 10.0      # flatten inventory if mark-to-market loss
    maker_fee_bps: float = field(default_factory=maker_bps)
    taker_fee_bps: float = field(default_factory=taker_bps)


@dataclass
class MMFill:
    tick: int
    side: str              # "buy" or "sell"
    price: float
    fee_bps: float         # negative = rebate
    inventory_after: int


@dataclass
class MMDateResult:
    date: str
    symbol: str
    n_fills: int
    n_buy_fills: int
    n_sell_fills: int
    n_round_trips: int     # matched buy+sell pairs
    n_stop_outs: int
    mean_spread_bps: float
    gross_pnl_bps: float   # from matched pairs
    total_fee_bps: float
    net_pnl_bps: float
    max_inventory: int
    mean_abs_inventory: float
    pnl_per_rt_bps: float


def simulate_day_mm(
    midprices: np.ndarray,
    spreads: np.ndarray,
    spread_bps: np.ndarray,
    signal: np.ndarray,
    cfg: MMConfig,
) -> tuple[list[MMFill], int]:
    """Market-making simulation: simultaneous bid+ask orders.

    Always maintains orders on both sides (subject to inventory limits).
    Imbalance signal optionally skews quotes.
    Returns (fills, n_stop_outs).
    """
    n = len(midprices)
    fills: list[MMFill] = []
    inventory = 0
    n_stop_outs = 0

    # Track inventory cost incrementally (FIFO cost basis)
    # inv_costs holds the entry price for each unit of inventory
    inv_costs: list[float] = []  # positive inventory = buy prices, negative = sell prices

    # Order state
    bid_price = 0.0
    ask_price = 0.0
    bid_active = False
    ask_active = False
    last_quote_tick = -cfg.requote_ticks

    i = cfg.latency_ticks
    while i < n:
        mid = midprices[i]
        sp = spreads[i]

        if not (mid == mid and sp == sp and sp > 0):  # fast NaN check
            i += 1
            continue

        # --- Stop loss: flatten if underwater ---
        if inv_costs:
            avg_entry = 0.0
            for c in inv_costs:
                avg_entry += c
            avg_entry /= len(inv_costs)

            if inventory > 0:
                unrealized = (mid - avg_entry) / avg_entry * 10000
            else:
                unrealized = (avg_entry - mid) / avg_entry * 10000

            if unrealized < -cfg.stop_loss_bps:
                side = "sell" if inventory > 0 else "buy"
                for _ in range(abs(inventory)):
                    fills.append(MMFill(
                        tick=i, side=side, price=mid,
                        fee_bps=cfg.taker_fee_bps, inventory_after=0,
                    ))
                inventory = 0
                inv_costs.clear()
                n_stop_outs += 1
                bid_active = False
                ask_active = False
                last_quote_tick = i
                i += cfg.latency_ticks
                continue

        # --- Check fills on existing orders ---
        if bid_active and mid <= bid_price:
            inventory += 1
            if inventory > 0:
                inv_costs.append(bid_price)
            elif inv_costs:
                inv_costs.pop(0)  # close one short unit
            fills.append(MMFill(
                tick=i, side="buy", price=bid_price,
                fee_bps=-cfg.maker_fee_bps, inventory_after=inventory,
            ))
            bid_active = False
            last_quote_tick = i - cfg.requote_ticks

        if ask_active and mid >= ask_price:
            inventory -= 1
            if inventory < 0:
                inv_costs.append(ask_price)
            elif inv_costs:
                inv_costs.pop(0)  # close one long unit
            fills.append(MMFill(
                tick=i, side="sell", price=ask_price,
                fee_bps=-cfg.maker_fee_bps, inventory_after=inventory,
            ))
            ask_active = False
            last_quote_tick = i - cfg.requote_ticks

        # --- Requote ---
        if i - last_quote_tick >= cfg.requote_ticks:
            last_quote_tick = i
            half_sp = sp / 2

            # Signal-based skew: shift quotes in direction of predicted move
            skew = 0.0
            if cfg.skew_signal:
                s = signal[i]
                if s == s:  # fast NaN check
                    skew = s * cfg.skew_signal_mult * sp

            # Inventory skew: push price away from accumulated inventory
            inv_skew = cfg.skew_factor * inventory * sp

            # Compute quote prices
            new_bid = mid - half_sp + skew - inv_skew
            new_ask = mid + half_sp + skew - inv_skew

            # Inventory limits
            bid_active = inventory < cfg.max_inventory
            if bid_active:
                bid_price = new_bid
            ask_active = inventory > -cfg.max_inventory
            if ask_active:
                ask_price = new_ask

        i += 1

    # End-of-day: flatten remaining inventory at mid (taker)
    if inventory != 0 and n > 0:
        final_mid = midprices[n - 1]
        if final_mid == final_mid:
            side = "sell" if inventory > 0 else "buy"
            for _ in range(abs(inventory)):
                fills.append(MMFill(
                    tick=n - 1, side=side, price=final_mid,
                    fee_bps=cfg.taker_fee_bps, inventory_after=0,
                ))

    return fills, n_stop_outs


def compute_mm_pnl(fills: list[MMFill]) -> dict:
    """Compute PnL from FIFO-matched buy/sell pairs."""
    buy_queue: list[MMFill] = []
    sell_queue: list[MMFill] = []
    matched_pnl = []
    total_fees = 0.0

    for f in fills:
        total_fees += f.fee_bps
        if f.side == "buy":
            if sell_queue:
                # Match with earliest sell (short covering)
                s = sell_queue.pop(0)
                pnl = (s.price - f.price) / f.price * 10000
                matched_pnl.append(pnl)
            else:
                buy_queue.append(f)
        else:
            if buy_queue:
                # Match with earliest buy (long closing)
                b = buy_queue.pop(0)
                pnl = (f.price - b.price) / b.price * 10000
                matched_pnl.append(pnl)
            else:
                sell_queue.append(f)

    return {
        "n_round_trips": len(matched_pnl),
        "gross_pnl_bps": float(np.sum(matched_pnl)) if matched_pnl else 0.0,
        "mean_pnl_bps": float(np.mean(matched_pnl)) if matched_pnl else 0.0,
        "total_fee_bps": total_fees,
        "net_pnl_bps": float(np.sum(matched_pnl)) + total_fees if matched_pnl else total_fees,
        "pnl_array": np.array(matched_pnl) if matched_pnl else np.array([0.0]),
        "unmatched_buys": len(buy_queue),
        "unmatched_sells": len(sell_queue),
    }


def run_mm_walk_forward(
    symbol: str,
    data_dir: Path,
    cfg: MMConfig,
    train_window: int = TRAIN_WINDOW,
) -> tuple[list[MMDateResult], list[MMFill]]:
    """Walk-forward MM simulation."""
    all_dates = [d for d in available_dates(data_dir=data_dir) if "clean" not in d]

    date_data: list[tuple[str, object]] = []
    for d in all_dates:
        df = load_day(data_dir, d, symbol)
        if df is not None:
            date_data.append((d, df))

    if len(date_data) < train_window + 1:
        return [], []

    results = []
    all_fills = []

    for idx in range(train_window, len(date_data)):
        test_date, test_df = date_data[idx]
        midprices = test_df["raw_midprice"].values.astype(np.float64)
        spreads_abs = test_df["raw_spread"].values.astype(np.float64)
        spread_bps_arr = test_df["raw_spread_bps"].values.astype(np.float64)
        signal_arr = test_df["imbalance_qty_l1"].values.astype(np.float64)
        mean_spread = float(np.nanmean(spread_bps_arr))

        fills, n_stops = simulate_day_mm(
            midprices, spreads_abs, spread_bps_arr, signal_arr, cfg,
        )

        pnl = compute_mm_pnl(fills)
        n_buys = sum(1 for f in fills if f.side == "buy")
        n_sells = len(fills) - n_buys
        inv_series = [f.inventory_after for f in fills] if fills else [0]

        results.append(MMDateResult(
            date=test_date,
            symbol=symbol,
            n_fills=len(fills),
            n_buy_fills=n_buys,
            n_sell_fills=n_sells,
            n_round_trips=pnl["n_round_trips"],
            n_stop_outs=n_stops,
            mean_spread_bps=round(mean_spread, 2),
            gross_pnl_bps=round(pnl["gross_pnl_bps"], 2),
            total_fee_bps=round(pnl["total_fee_bps"], 2),
            net_pnl_bps=round(pnl["net_pnl_bps"], 2),
            max_inventory=int(max(abs(x) for x in inv_series)),
            mean_abs_inventory=round(float(np.mean(np.abs(inv_series))), 2),
            pnl_per_rt_bps=round(pnl["mean_pnl_bps"], 3),
        ))
        all_fills.extend(fills)

    return results, all_fills


def print_mm_report(
    all_results: dict[str, list[MMDateResult]],
    all_fills: dict[str, list[MMFill]],
    cfg: MMConfig,
):
    print(f"\n{'=' * 100}")
    print(f"  MARKET-MAKING SIMULATOR")
    print(f"  MaxInv: {cfg.max_inventory} | Requote: {cfg.requote_ticks} ticks "
          f"| Latency: {cfg.latency_ticks} | StopLoss: {cfg.stop_loss_bps} bps")
    print(f"  Skew signal: {cfg.skew_signal} (mult={cfg.skew_signal_mult}) "
          f"| Inv skew: {cfg.skew_factor} "
          f"| Maker rebate: {cfg.maker_fee_bps} bps | Taker fee: {cfg.taker_fee_bps} bps")
    print(f"{'=' * 100}")

    for symbol in SYMBOLS:
        results = all_results.get(symbol, [])
        fills = all_fills.get(symbol, [])
        if not results:
            continue

        daily_pnl = np.array([r.net_pnl_bps for r in results])
        sharpe = sharpe_daily(daily_pnl)
        total = float(np.sum(daily_pnl))
        total_rt = sum(r.n_round_trips for r in results)
        total_fills = sum(r.n_fills for r in results)

        print(f"\n  [{symbol}] {len(results)} OOS dates | "
              f"Sharpe {sharpe:+.1f} | Total {total:+.0f} bps | "
              f"{total_fills} fills, {total_rt} round-trips")

        print(f"  {'Date':12s} {'Fills':>5s} {'Buy':>4s} {'Sell':>4s} "
              f"{'RTs':>4s} {'Stops':>5s} {'Gross':>8s} {'Fees':>7s} "
              f"{'Net':>8s} {'PnL/RT':>7s} {'MaxInv':>6s} {'AvgInv':>6s} {'Sprd':>5s}")
        print(f"  {'-'*12} {'-'*5} {'-'*4} {'-'*4} "
              f"{'-'*4} {'-'*5} {'-'*8} {'-'*7} "
              f"{'-'*8} {'-'*7} {'-'*6} {'-'*6} {'-'*5}")
        for r in results:
            print(f"  {r.date:12s} {r.n_fills:5d} {r.n_buy_fills:4d} {r.n_sell_fills:4d} "
                  f"{r.n_round_trips:4d} {r.n_stop_outs:5d} {r.gross_pnl_bps:+8.1f} "
                  f"{r.total_fee_bps:+7.1f} {r.net_pnl_bps:+8.1f} "
                  f"{r.pnl_per_rt_bps:+7.3f} {r.max_inventory:6d} "
                  f"{r.mean_abs_inventory:6.1f} {r.mean_spread_bps:5.1f}")

        if not fills:
            continue

        # Aggregate
        all_pnl = compute_mm_pnl(fills)
        maker_fills = [f for f in fills if f.fee_bps < 0]
        taker_fills = [f for f in fills if f.fee_bps > 0]

        print(f"\n  Aggregate:")
        print(f"    Total fills: {len(fills)} ({len(maker_fills)} maker, {len(taker_fills)} taker)")
        print(f"    Round-trips: {all_pnl['n_round_trips']}")
        print(f"    Gross PnL: {all_pnl['gross_pnl_bps']:+.1f} bps "
              f"({all_pnl['mean_pnl_bps']:+.3f} bps/RT)")
        print(f"    Total fees: {all_pnl['total_fee_bps']:+.1f} bps "
              f"(maker rebates: {len(maker_fills) * cfg.maker_fee_bps:.1f} bps earned, "
              f"taker costs: {len(taker_fills) * cfg.taker_fee_bps:.1f} bps paid)")
        print(f"    Net PnL: {all_pnl['net_pnl_bps']:+.1f} bps")
        if all_pnl["n_round_trips"] > 0:
            pa = all_pnl["pnl_array"]
            print(f"    RT PnL distribution: "
                  f"P10={np.percentile(pa,10):+.2f}  "
                  f"P25={np.percentile(pa,25):+.2f}  "
                  f"P50={np.percentile(pa,50):+.2f}  "
                  f"P75={np.percentile(pa,75):+.2f}  "
                  f"P90={np.percentile(pa,90):+.2f}")
            print(f"    Win rate: {np.mean(pa > 0):.1%}")
        total_stops = sum(r.n_stop_outs for r in results)
        print(f"    Stop-outs: {total_stops}")


# ---------------------------------------------------------------------------
# Walk-forward orchestrator
# ---------------------------------------------------------------------------

def load_day(data_dir: Path, date_str: str, symbol: str):
    """Load one day of tick data."""
    df = load_features(
        symbols=[symbol],
        date_range=(date_str, date_str),
        columns=REQUIRED_COLS,
        data_dir=data_dir,
        validate=False,
    )
    if df.empty or len(df) < 500:
        return None
    return df


def build_date_result(
    date_str: str, symbol: str, trades: list[RoundTrip],
    n_signals: int, mean_spread_bps: float,
) -> DateResult:
    """Aggregate trade list into DateResult."""
    n_rt = len(trades)
    n_maker = sum(1 for t in trades if t.exit_type == "maker")
    n_taker = n_rt - n_maker
    gross = np.array([t.gross_pnl_bps for t in trades]) if trades else np.array([0.0])
    net = np.array([t.net_pnl_bps for t in trades]) if trades else np.array([0.0])
    hold = np.array([t.holding_ticks for t in trades]) if trades else np.array([0.0])

    return DateResult(
        date=date_str,
        symbol=symbol,
        n_signals=n_signals,
        n_entry_fills=n_rt,
        n_round_trips=n_rt,
        n_maker_exits=n_maker,
        n_taker_exits=n_taker,
        fill_rate=n_rt / max(n_signals, 1),
        maker_exit_rate=n_maker / max(n_rt, 1),
        mean_gross_bps=round(float(np.mean(gross)), 3) if trades else 0.0,
        mean_net_bps=round(float(np.mean(net)), 3) if trades else 0.0,
        total_net_bps=round(float(np.sum(net)), 1) if trades else 0.0,
        mean_holding_ticks=round(float(np.mean(hold)), 1) if trades else 0.0,
        mean_spread_bps=round(mean_spread_bps, 2),
    )


def run_walk_forward(
    symbol: str,
    data_dir: Path,
    cfg: SimConfig,
    train_window: int = TRAIN_WINDOW,
) -> tuple[list[DateResult], list[RoundTrip]]:
    """Walk-forward simulation across all available dates."""
    all_dates = [d for d in available_dates(data_dir=data_dir) if "clean" not in d]

    # Pre-load all dates
    date_data: list[tuple[str, object]] = []
    for d in all_dates:
        df = load_day(data_dir, d, symbol)
        if df is not None:
            date_data.append((d, df))

    if len(date_data) < train_window + 1:
        print(f"  {symbol}: only {len(date_data)} dates, need {train_window + 1}")
        return [], []

    results = []
    all_trades = []

    for idx in range(train_window, len(date_data)):
        # Train: calibrate regime threshold
        train_regime = []
        for _, tdf in date_data[idx - train_window:idx]:
            vals = tdf[cfg.regime_feature].values
            train_regime.append(vals[np.isfinite(vals)])
        train_regime = np.concatenate(train_regime)
        regime_thresh = float(np.nanpercentile(train_regime, cfg.regime_percentile))

        # Test
        test_date, test_df = date_data[idx]
        midprices = test_df["raw_midprice"].values.astype(np.float64)
        spreads_abs = test_df["raw_spread"].values.astype(np.float64)
        spread_bps_arr = test_df["raw_spread_bps"].values.astype(np.float64)
        signal_arr = test_df["imbalance_qty_l1"].values.astype(np.float64)
        regime_arr = test_df[cfg.regime_feature].values.astype(np.float64)
        mean_spread = float(np.nanmean(spread_bps_arr))

        trades, n_signals = simulate_day(
            midprices, spreads_abs, spread_bps_arr,
            signal_arr, regime_arr, regime_thresh, cfg,
        )

        dr = build_date_result(test_date, symbol, trades, n_signals, mean_spread)
        results.append(dr)
        all_trades.extend(trades)

    return results, all_trades


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    all_results: dict[str, list[DateResult]],
    all_trades: dict[str, list[RoundTrip]],
    cfg: SimConfig,
):
    maker_f = cfg.maker_fee_bps
    taker_f = cfg.taker_fee_bps

    print(f"\n{'=' * 95}")
    print(f"  LIMIT ORDER ROUND-TRIP SIMULATOR")
    print(f"  Threshold: {cfg.entry_threshold} | Latency: {cfg.latency_ticks} ticks "
          f"| Entry timeout: {cfg.entry_timeout_ticks} | Exit timeout: {cfg.exit_timeout_ticks}")
    print(f"  Stop loss: {cfg.stop_loss_bps} bps | Fill model: {cfg.fill_model} "
          f"| Maker rebate: {maker_f} bps | Taker fee: {taker_f} bps")
    print(f"{'=' * 95}")

    for symbol in SYMBOLS:
        results = all_results.get(symbol, [])
        trades = all_trades.get(symbol, [])
        if not results:
            continue

        daily_pnl = np.array([r.total_net_bps for r in results])
        sharpe = sharpe_daily(daily_pnl)
        total = float(np.sum(daily_pnl))

        print(f"\n  [{symbol}] {len(results)} OOS dates | "
              f"Sharpe {sharpe:+.1f} | Total {total:+.0f} bps | "
              f"{len(trades)} trades")

        # Per-date table
        print(f"  {'Date':12s} {'Sigs':>5s} {'Fills':>5s} {'FR%':>5s} "
              f"{'MkrX%':>6s} {'RT':>4s} {'Net/RT':>8s} {'Total':>8s} "
              f"{'Hold':>5s} {'Sprd':>5s}")
        print(f"  {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*6} {'-'*4} "
              f"{'-'*8} {'-'*8} {'-'*5} {'-'*5}")
        for r in results:
            print(f"  {r.date:12s} {r.n_signals:5d} {r.n_entry_fills:5d} "
                  f"{r.fill_rate:4.0%} {r.maker_exit_rate:5.0%} "
                  f"{r.n_round_trips:4d} {r.mean_net_bps:+8.2f} "
                  f"{r.total_net_bps:+8.1f} {r.mean_holding_ticks:5.0f} "
                  f"{r.mean_spread_bps:5.1f}")

        if not trades:
            continue

        # Aggregate stats
        net_arr = np.array([t.net_pnl_bps for t in trades])
        gross_arr = np.array([t.gross_pnl_bps for t in trades])
        hold_arr = np.array([t.holding_ticks for t in trades])
        n_maker = sum(1 for t in trades if t.exit_type == "maker")
        n_stop = sum(1 for t in trades if t.exit_reason == "stop_loss")
        n_timeout = sum(1 for t in trades if t.exit_reason == "timeout")
        n_buy = sum(1 for t in trades if t.side == "buy")
        n_sell = len(trades) - n_buy

        print(f"\n  Aggregate:")
        print(f"    Trades: {len(trades)} ({n_buy} buy, {n_sell} sell)")
        print(f"    Fill rate: {sum(r.n_entry_fills for r in results)}"
              f" / {sum(r.n_signals for r in results)}"
              f" = {sum(r.n_entry_fills for r in results) / max(sum(r.n_signals for r in results), 1):.1%}")
        print(f"    Exit type: {n_maker} maker ({n_maker/len(trades):.0%}), "
              f"{n_stop} stop-loss, {n_timeout} timeout")
        print(f"    Gross PnL: mean {np.mean(gross_arr):+.3f} bps, "
              f"median {np.median(gross_arr):+.3f} bps")
        print(f"    Net PnL:   mean {np.mean(net_arr):+.3f} bps, "
              f"median {np.median(net_arr):+.3f} bps")
        print(f"    PnL distribution: "
              f"P10={np.percentile(net_arr,10):+.2f}  "
              f"P25={np.percentile(net_arr,25):+.2f}  "
              f"P50={np.percentile(net_arr,50):+.2f}  "
              f"P75={np.percentile(net_arr,75):+.2f}  "
              f"P90={np.percentile(net_arr,90):+.2f}")
        print(f"    Win rate: {np.mean(net_arr > 0):.1%}")
        print(f"    Holding: mean {np.mean(hold_arr):.0f} ticks "
              f"({np.mean(hold_arr)*0.1:.1f}s)")

        # Economics breakdown
        maker_exit_trades = [t for t in trades if t.exit_type == "maker"]
        taker_exit_trades = [t for t in trades if t.exit_type == "taker"]
        print(f"\n  Economics:")
        if maker_exit_trades:
            mk_gross = np.mean([t.gross_pnl_bps for t in maker_exit_trades])
            mk_net = np.mean([t.net_pnl_bps for t in maker_exit_trades])
            print(f"    Maker exits: gross {mk_gross:+.3f}, "
                  f"net {mk_net:+.3f} bps/trade "
                  f"(fee: {-2*maker_f:+.1f} bps rebate)")
        if taker_exit_trades:
            tk_gross = np.mean([t.gross_pnl_bps for t in taker_exit_trades])
            tk_net = np.mean([t.net_pnl_bps for t in taker_exit_trades])
            print(f"    Taker exits: gross {tk_gross:+.3f}, "
                  f"net {tk_net:+.3f} bps/trade "
                  f"(fee: {-maker_f + taker_f:+.1f} bps)")

        # Direction breakdown
        buy_trades = [t for t in trades if t.side == "buy"]
        sell_trades = [t for t in trades if t.side == "sell"]
        if buy_trades:
            print(f"    Buy trades:  {len(buy_trades):4d}, "
                  f"net {np.mean([t.net_pnl_bps for t in buy_trades]):+.3f} bps/trade")
        if sell_trades:
            print(f"    Sell trades: {len(sell_trades):4d}, "
                  f"net {np.mean([t.net_pnl_bps for t in sell_trades]):+.3f} bps/trade")


def print_sweep_report(
    sweep_results: list[tuple[float, dict[str, list[DateResult]], dict[str, list[RoundTrip]]]],
):
    print(f"\n{'=' * 95}")
    print(f"  THRESHOLD SWEEP SUMMARY")
    print(f"{'=' * 95}")
    print(f"  {'Thresh':>7s}", end="")
    for symbol in SYMBOLS:
        print(f" | {'Trades':>6s} {'FR%':>5s} {'MkrX%':>6s} "
              f"{'Net/RT':>7s} {'Total':>8s} {'Sharpe':>7s}", end="")
    print()
    print(f"  {'-'*7}" + (" | " + "-"*44) * len(SYMBOLS))

    for thresh, all_results, all_trades in sweep_results:
        print(f"  {thresh:7.2f}", end="")
        for symbol in SYMBOLS:
            results = all_results.get(symbol, [])
            trades = all_trades.get(symbol, [])
            if not results or not trades:
                print(f" | {'---':>6s} {'---':>5s} {'---':>6s} "
                      f"{'---':>7s} {'---':>8s} {'---':>7s}", end="")
                continue
            daily_pnl = np.array([r.total_net_bps for r in results])
            sharpe = sharpe_daily(daily_pnl)
            n_trades = len(trades)
            net_arr = np.array([t.net_pnl_bps for t in trades])
            n_sigs = sum(r.n_signals for r in results)
            n_fills = sum(r.n_entry_fills for r in results)
            fr = n_fills / max(n_sigs, 1)
            n_maker = sum(1 for t in trades if t.exit_type == "maker")
            mkr = n_maker / max(n_trades, 1)
            print(f" | {n_trades:6d} {fr:4.0%} {mkr:5.0%} "
                  f"{np.mean(net_arr):+7.2f} {np.sum(net_arr):+8.0f} "
                  f"{sharpe:+7.1f}", end="")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Limit Order Round-Trip Simulator")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "features"))
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--all-symbols", action="store_true")
    parser.add_argument("--entry-threshold", type=float, default=0.3)
    parser.add_argument("--stop-loss-bps", type=float, default=5.0)
    parser.add_argument("--exit-timeout", type=int, default=100)
    parser.add_argument("--entry-timeout", type=int, default=50)
    parser.add_argument("--latency-ticks", type=int, default=2)
    parser.add_argument("--fill-model", choices=["mid_cross", "prob_adjusted"],
                        default="mid_cross")
    parser.add_argument("--mode", choices=["directional", "mm"], default="directional",
                        help="Simulation mode: directional or market-making")
    parser.add_argument("--max-inventory", type=int, default=3,
                        help="[mm] Max inventory per side")
    parser.add_argument("--requote", type=int, default=10,
                        help="[mm] Requote interval in ticks")
    parser.add_argument("--skew-signal-mult", type=float, default=0.5,
                        help="[mm] Signal skew multiplier (in spreads)")
    parser.add_argument("--inv-skew", type=float, default=0.0,
                        help="[mm] Inventory skew factor")
    parser.add_argument("--no-signal-skew", action="store_true",
                        help="[mm] Disable imbalance-based quote skew")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep entry thresholds [0.1, 0.2, 0.3, 0.5]")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    symbols = SYMBOLS if args.all_symbols else [args.symbol.upper()]

    # --- Market-Making mode ---
    if args.mode == "mm":
        mm_cfg = MMConfig(
            max_inventory=args.max_inventory,
            requote_ticks=args.requote,
            latency_ticks=args.latency_ticks,
            skew_signal=not args.no_signal_skew,
            skew_signal_mult=args.skew_signal_mult,
            skew_factor=args.inv_skew,
            stop_loss_bps=args.stop_loss_bps,
        )
        all_results = {}
        all_fills_map = {}
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            t0 = time.time()
            results, mm_fills = run_mm_walk_forward(sym, data_dir, mm_cfg)
            print(f"{len(results)} dates, {len(mm_fills)} fills ({time.time()-t0:.1f}s)")
            all_results[sym] = results
            all_fills_map[sym] = mm_fills

        print_mm_report(all_results, all_fills_map, mm_cfg)

        if args.save:
            report = {}
            for sym in symbols:
                results = all_results.get(sym, [])
                if not results:
                    continue
                daily_pnl = np.array([r.net_pnl_bps for r in results])
                report[sym] = {
                    "n_dates": len(results),
                    "sharpe": round(sharpe_daily(daily_pnl), 2),
                    "total_net_bps": round(float(np.sum(daily_pnl)), 1),
                    "per_date": [asdict(r) for r in results],
                }
            out = ROOT / "reports" / "limit_order_mm.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump({"config": asdict(mm_cfg), "results": report}, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
            print(f"\n  Saved to {out}")
        return

    cfg = SimConfig(
        entry_threshold=args.entry_threshold,
        stop_loss_bps=args.stop_loss_bps,
        exit_timeout_ticks=args.exit_timeout,
        entry_timeout_ticks=args.entry_timeout,
        latency_ticks=args.latency_ticks,
        fill_model=args.fill_model,
    )

    if args.sweep:
        thresholds = [0.1, 0.15, 0.2, 0.3, 0.5]
        sweep_results = []
        for thresh in thresholds:
            cfg_t = SimConfig(
                entry_threshold=thresh,
                stop_loss_bps=args.stop_loss_bps,
                exit_timeout_ticks=args.exit_timeout,
                entry_timeout_ticks=args.entry_timeout,
                latency_ticks=args.latency_ticks,
                fill_model=args.fill_model,
            )
            print(f"\n  Threshold {thresh}...")
            all_results = {}
            all_trades_map = {}
            for sym in symbols:
                print(f"    {sym}...", end=" ", flush=True)
                t0 = time.time()
                results, trades = run_walk_forward(sym, data_dir, cfg_t)
                print(f"{len(results)} dates, {len(trades)} trades ({time.time()-t0:.1f}s)")
                all_results[sym] = results
                all_trades_map[sym] = trades
            sweep_results.append((thresh, all_results, all_trades_map))

        print_sweep_report(sweep_results)
        return

    # Single run
    all_results = {}
    all_trades_map = {}
    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        t0 = time.time()
        results, trades = run_walk_forward(sym, data_dir, cfg)
        print(f"{len(results)} dates, {len(trades)} trades ({time.time()-t0:.1f}s)")
        all_results[sym] = results
        all_trades_map[sym] = trades

    print_report(all_results, all_trades_map, cfg)

    if args.save:
        report = {}
        for sym in symbols:
            results = all_results.get(sym, [])
            trades = all_trades_map.get(sym, [])
            if not results:
                continue
            daily_pnl = np.array([r.total_net_bps for r in results])
            report[sym] = {
                "n_dates": len(results),
                "n_trades": len(trades),
                "sharpe": round(sharpe_daily(daily_pnl), 2),
                "total_net_bps": round(float(np.sum(daily_pnl)), 1),
                "per_date": [asdict(r) for r in results],
            }
        out = ROOT / "reports" / "limit_order_sim.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"config": asdict(cfg), "results": report}, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
        print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
