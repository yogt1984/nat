"""
Signal-to-Order Bridge — MF 3f Liquidity Signal

Daemon that reads live ingestor data, computes the 3f composite signal,
and places/manages maker orders on Hyperliquid.

Flow each cycle:
  1. Load latest bars from data/features/
  2. Train z-score params on last 3 completed dates
  3. Compute signal on today's bars
  4. Check kill switches against PnL history
  5. Reconcile desired position vs current position
  6. Place/cancel maker orders to match target
  7. Log everything to data/execution/

Modes:
  dry-run:  computes signal + logs orders, no exchange interaction (default)
  paper:    computes signal + calls exchange in dry-run mode
  live:     actually places orders (requires HL_PRIVATE_KEY)

Usage:
  python scripts/execution/signal_bridge.py --mode dry-run
  python scripts/execution/signal_bridge.py --mode live --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# `nat bridge start` / direct invocation puts scripts/execution/ on sys.path[0];
# add scripts/ so sibling packages (risk, agent, signal_lifecycle) resolve when
# the daemon is launched directly (not just imported under conftest).
import sys
_SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from data.features import load_features


from alpha.deployer import (
    ScaleSchedule,
    evaluate_kill_switches,
    compute_position_limits,
)
# NOTE: HyperliquidClient is imported lazily where it's actually used (run_bridge
# / live execution) so the dry-run daemon, `status`, and `health` don't require
# the exchange SDK (msgpack/eth-account). The type hint below is stringized via
# `from __future__ import annotations`, so no runtime import is needed for it.

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "features"
EXEC_LOG_DIR = ROOT / "data" / "execution"

# ── Config ────────────────────────────────────────────────────────────────

BAR_SECONDS = 300  # 5min
HORIZON_BARS = 20  # 100min
TRAIN_WINDOW = 3
MIN_BARS_PER_DATE = 12
P_LONG = 80
P_SHORT = 20

CYCLE_SECONDS = 300  # check every 5 min (1 bar)
try:
    from config_utils import load_symbols
except ImportError:
    from config_utils import load_symbols
SYMBOLS = load_symbols()

LOAD_COLUMNS = [
    "timestamp_ns", "symbol", "raw_midprice",
    "raw_spread_bps", "raw_ask_depth_5", "flow_vwap_deviation",
]

# Maker fee on Hyperliquid (rebate on some tiers, but assume small positive)
MAKER_FEE_BPS = 0.30


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class SignalState:
    """Current signal output for one symbol."""
    symbol: str
    timestamp: str
    direction: int  # +1 long, -1 short, 0 flat
    composite: float
    n_bars_today: int
    train_dates: list[str]


@dataclass
class TargetPosition:
    """Desired position based on signal and limits."""
    symbol: str
    direction: int  # +1, -1, 0
    size_usd: float  # target notional
    size_asset: float  # target in asset units
    entry_price: float  # current mid for reference


@dataclass
class CycleLog:
    """One execution cycle log entry."""
    timestamp: str
    symbol: str
    signal: SignalState
    target: TargetPosition | None
    current_position_size: float
    action: str  # "hold", "open_long", "open_short", "close", "reduce", "skip"
    order_placed: bool
    order_result: dict | None
    kill_switch_status: str  # "ok", "halted", "killed"
    reason: str


# ── Data loading (same as paper trader) ──────────────────────────────────

def load_date(data_dir: Path, date_str: str, symbol: str) -> pd.DataFrame | None:
    df = load_features(
        symbols=[symbol],
        date_range=(date_str, date_str),
        columns=LOAD_COLUMNS,
        data_dir=data_dir,
        validate=False,
    )
    return df if not df.empty else None


def aggregate_to_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bar_ns = BAR_SECONDS * 1_000_000_000
    ticks = ticks.copy()
    ticks["bar_id"] = ticks["timestamp_ns"].values // bar_ns
    agg = {
        "timestamp_ns": ("timestamp_ns", "first"),
        "midprice_last": ("raw_midprice", "last"),
        "spread_bps_last": ("raw_spread_bps", "last"),
        "depth_5_std": ("raw_ask_depth_5", "std"),
        "n_ticks": ("raw_midprice", "count"),
    }
    if "flow_vwap_deviation" in ticks.columns:
        agg["vwap_deviation_std"] = ("flow_vwap_deviation", "std")
    bars = ticks.groupby("bar_id").agg(**agg).reset_index(drop=True)
    bars = bars[bars["n_ticks"] >= 10].reset_index(drop=True)
    bars["depth_5_std"] = bars["depth_5_std"].fillna(0.0)
    if "vwap_deviation_std" in bars.columns:
        bars["vwap_deviation_std"] = bars["vwap_deviation_std"].fillna(0.0)
    return bars


def discover_dates(data_dir: Path) -> list[str]:
    return sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("2026-") and "clean" not in d and (data_dir / d).is_dir()
    )


# ── Signal computation ───────────────────────────────────────────────────

def compute_zscore_params_3f(train_bars_list: list[pd.DataFrame]) -> dict | None:
    spread = np.concatenate([b["spread_bps_last"].values for b in train_bars_list])
    depth = np.concatenate([b["depth_5_std"].values for b in train_bars_list])
    vwap_arrs = [b["vwap_deviation_std"].values for b in train_bars_list
                 if "vwap_deviation_std" in b.columns]
    if not vwap_arrs:
        return None
    vwap = np.concatenate(vwap_arrs)
    n = min(len(spread), len(depth), len(vwap))
    spread, depth, vwap = spread[:n], depth[:n], vwap[:n]
    mask = np.isfinite(spread) & np.isfinite(depth) & np.isfinite(vwap)
    spread, depth, vwap = spread[mask], depth[mask], vwap[mask]
    if len(spread) < 20:
        return None
    params = {
        "spread_mean": np.mean(spread), "spread_std": max(np.std(spread), 1e-10),
        "depth_mean": np.mean(depth), "depth_std": max(np.std(depth), 1e-10),
        "vwap_mean": np.mean(vwap), "vwap_std": max(np.std(vwap), 1e-10),
    }
    z_s = (spread - params["spread_mean"]) / params["spread_std"]
    z_d = (depth - params["depth_mean"]) / params["depth_std"]
    z_v = (vwap - params["vwap_mean"]) / params["vwap_std"]
    composite = (z_s + z_d + z_v) / 3.0
    params["p_long"] = np.percentile(composite, P_LONG)
    params["p_short"] = np.percentile(composite, P_SHORT)
    return params


def compute_signal(
    data_dir: Path, symbol: str, all_dates: list[str],
) -> SignalState | None:
    """Compute the current signal for a symbol using latest data."""
    # Load all dates into bars
    date_bars: list[tuple[str, pd.DataFrame]] = []
    for date_str in all_dates:
        ticks = load_date(data_dir, date_str, symbol)
        if ticks is None or len(ticks) < 100:
            continue
        bars = aggregate_to_bars(ticks)
        if len(bars) >= MIN_BARS_PER_DATE:
            date_bars.append((date_str, bars))

    if len(date_bars) < TRAIN_WINDOW + 1:
        return None

    # Train on last TRAIN_WINDOW completed dates (exclude today)
    train_bars = [b for _, b in date_bars[-(TRAIN_WINDOW + 1):-1]]
    train_dates = [d for d, _ in date_bars[-(TRAIN_WINDOW + 1):-1]]
    today_date, today_bars = date_bars[-1]

    params = compute_zscore_params_3f(train_bars)
    if params is None:
        return None

    # Score today's bars
    bars = today_bars.copy()
    z_s = (bars["spread_bps_last"] - params["spread_mean"]) / params["spread_std"]
    z_d = (bars["depth_5_std"] - params["depth_mean"]) / params["depth_std"]
    if "vwap_deviation_std" not in bars.columns:
        return None
    z_v = (bars["vwap_deviation_std"] - params["vwap_mean"]) / params["vwap_std"]
    bars["composite"] = (z_s + z_d + z_v) / 3.0

    # Use the LATEST bar's signal
    latest_composite = float(bars["composite"].iloc[-1])
    if latest_composite >= params["p_long"]:
        direction = 1
    elif latest_composite <= params["p_short"]:
        direction = -1
    else:
        direction = 0

    return SignalState(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        direction=direction,
        composite=round(latest_composite, 4),
        n_bars_today=len(bars),
        train_dates=train_dates,
    )


# ── Position sizing ──────────────────────────────────────────────────────

def compute_target(
    signal: SignalState,
    mid_price: float,
    position_limit_usd: float,
) -> TargetPosition:
    """Convert signal + limits into a target position."""
    if signal.direction == 0 or mid_price <= 0:
        return TargetPosition(
            symbol=signal.symbol,
            direction=0,
            size_usd=0.0,
            size_asset=0.0,
            entry_price=mid_price,
        )

    # Full allocation to limit when signal fires
    size_usd = position_limit_usd
    size_asset = size_usd / mid_price

    return TargetPosition(
        symbol=signal.symbol,
        direction=signal.direction,
        size_usd=round(size_usd, 2),
        size_asset=round(size_asset, 6),
        entry_price=mid_price,
    )


# ── Order management ─────────────────────────────────────────────────────

def reconcile_position(
    target: TargetPosition,
    current_size: float,  # signed: positive = long, negative = short
    client: HyperliquidClient,
    mid_price: float,
    mode: str,
) -> CycleLog | None:
    """
    Decide what order action to take based on target vs current position.
    Returns a log entry with the action taken.
    """
    symbol = target.symbol
    target_size = target.direction * target.size_asset
    delta = target_size - current_size

    # Determine action
    if abs(delta) < 1e-8:
        action = "hold"
    elif target.direction == 0 and abs(current_size) > 1e-8:
        action = "close"
    elif target.direction != 0 and abs(current_size) < 1e-8:
        action = "open_long" if target.direction == 1 else "open_short"
    elif np.sign(delta) != np.sign(current_size) and abs(current_size) > 1e-8:
        action = "close"  # flip direction: close first
    else:
        action = "reduce" if abs(target_size) < abs(current_size) else "hold"

    order_placed = False
    order_result = None

    if action == "hold":
        pass
    elif mode == "dry-run":
        log.info(f"[DRY-RUN] {symbol}: {action} delta={delta:+.6f} @ ~{mid_price:.1f}")
        order_result = {"dry_run": True, "delta": delta}
    elif action == "close":
        # Cancel existing orders, place reducing order
        client.cancel_all(symbol)
        is_buy = current_size < 0  # buy to close short, sell to close long
        # Place at mid (maker)
        price = mid_price * (1.001 if is_buy else 0.999)
        result = client.place_maker_order(
            symbol, is_buy=is_buy,
            price=round(price, 1),
            size=round(abs(current_size), 6),
            reduce_only=True,
        )
        order_placed = True
        order_result = asdict(result)
    elif action in ("open_long", "open_short"):
        is_buy = target.direction == 1
        # Place at mid (maker), slight offset for fill probability
        price = mid_price * (0.999 if is_buy else 1.001)
        result = client.place_maker_order(
            symbol, is_buy=is_buy,
            price=round(price, 1),
            size=round(target.size_asset, 6),
        )
        order_placed = True
        order_result = asdict(result)

    return action, order_placed, order_result


# ── PnL tracking ─────────────────────────────────────────────────────────

def load_pnl_history() -> list[dict]:
    """Load daily PnL history from execution logs."""
    pnl_file = EXEC_LOG_DIR / "daily_pnl.json"
    if pnl_file.exists():
        with open(pnl_file) as f:
            return json.load(f)
    return []


def compute_kill_metrics(pnl_history: list[dict]) -> dict:
    """Compute kill switch metrics from PnL history."""
    if not pnl_history:
        return {"daily_pnl_pct": 0, "weekly_dd_pct": 0, "monthly_dd_pct": 0, "ic_negative_days": 0}

    daily_pnl = pnl_history[-1].get("pnl_pct", 0) if pnl_history else 0

    # Weekly: sum of last 7 days
    week = pnl_history[-7:] if len(pnl_history) >= 7 else pnl_history
    weekly_pnls = [d.get("pnl_pct", 0) for d in week]
    weekly_cum = np.cumsum(weekly_pnls)
    weekly_dd = float(np.min(weekly_cum) - np.max(weekly_cum)) if len(weekly_cum) > 1 else 0
    weekly_dd_pct = abs(min(weekly_dd, 0))

    # Monthly: sum of last 30 days
    month = pnl_history[-30:] if len(pnl_history) >= 30 else pnl_history
    monthly_pnls = [d.get("pnl_pct", 0) for d in month]
    monthly_cum = np.cumsum(monthly_pnls)
    monthly_dd = float(np.min(monthly_cum) - np.max(monthly_cum)) if len(monthly_cum) > 1 else 0
    monthly_dd_pct = abs(min(monthly_dd, 0))

    # IC negative days: consecutive days with negative PnL at end
    ic_neg = 0
    for d in reversed(pnl_history):
        if d.get("pnl_pct", 0) < 0:
            ic_neg += 1
        else:
            break

    return {
        "daily_pnl_pct": daily_pnl,
        "weekly_dd_pct": weekly_dd_pct,
        "monthly_dd_pct": monthly_dd_pct,
        "ic_negative_days": ic_neg,
    }


# ── Logging ──────────────────────────────────────────────────────────────

def log_cycle(entry: CycleLog):
    """Append cycle log to daily file."""
    EXEC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = EXEC_LOG_DIR / f"cycles_{today}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(asdict(entry), default=str) + "\n")


# ── Main loop ────────────────────────────────────────────────────────────

def run_bridge(
    mode: str = "dry-run",
    symbols: list[str] | None = None,
    account_value_usd: float = 10000.0,
    weeks_deployed: int = 0,
    cycle_seconds: int = CYCLE_SECONDS,
):
    """Main execution loop."""
    if symbols is None:
        symbols = SYMBOLS

    # Initialize client (lazy import — keeps the dry-run daemon free of the SDK)
    from execution.hyperliquid_client import HyperliquidClient

    if mode == "live":
        client = HyperliquidClient.from_env(dry_run=False)
    elif mode == "paper":
        client = HyperliquidClient.from_env(dry_run=True)
    else:
        client = HyperliquidClient.readonly()

    scale = ScaleSchedule.get_schedule(weeks_deployed)
    limits = compute_position_limits(symbols, account_value_usd, scale.max_capital_pct)
    limit_map = {lim.symbol: lim.max_position_usd for lim in limits}

    print(f"Signal Bridge — mode={mode}")
    print(f"  Symbols: {symbols}")
    print(f"  Scale: {scale.max_capital_pct}% capital ({scale.order_type})")
    print(f"  Limits: {', '.join(f'{s}=${limit_map[s]:.0f}' for s in symbols)}")
    print(f"  Cycle: {cycle_seconds}s")
    if client.wallet:
        print(f"  Wallet: {client.wallet}")
    print()

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now(timezone.utc)
        now_str = now.strftime("%H:%M:%S")

        try:
            # 1. Discover dates
            all_dates = discover_dates(DATA_DIR)
            if len(all_dates) < TRAIN_WINDOW + 1:
                print(f"[{now_str}] Waiting for data ({len(all_dates)} dates)")
                time.sleep(cycle_seconds)
                continue

            # 2. Check kill switches
            pnl_history = load_pnl_history()
            kill_metrics = compute_kill_metrics(pnl_history)
            kills = evaluate_kill_switches(**kill_metrics)
            any_kill = any(k.triggered for k in kills)
            killed = any(k.triggered and k.action == "kill" for k in kills)

            if killed:
                kill_status = "killed"
                print(f"[{now_str}] KILLED — strategy terminated")
                break
            elif any_kill:
                kill_status = "halted"
                triggered = [k.name for k in kills if k.triggered]
                print(f"[{now_str}] HALTED — kill switches: {triggered}")
                time.sleep(cycle_seconds)
                continue
            else:
                kill_status = "ok"

            # 3. Get current prices
            if mode != "dry-run":
                prices = client.get_midprices()
            else:
                prices = {}

            # 4. Process each symbol
            for symbol in symbols:
                signal = compute_signal(DATA_DIR, symbol, all_dates)
                if signal is None:
                    continue

                # Get current mid price
                if symbol in prices:
                    mid = prices[symbol]
                else:
                    # Estimate from latest bar data
                    ticks = load_date(DATA_DIR, all_dates[-1], symbol)
                    if ticks is not None and len(ticks) > 0:
                        mid = float(ticks["raw_midprice"].iloc[-1])
                    else:
                        continue

                # Compute target position
                target = compute_target(signal, mid, limit_map.get(symbol, 0))

                # Get current position
                current_size = 0.0
                if mode != "dry-run" and client.wallet:
                    try:
                        positions = client.get_positions()
                        for p in positions:
                            if p.coin == symbol:
                                current_size = p.size
                    except Exception as e:
                        log.warning(f"Failed to get positions: {e}")

                # Reconcile and execute
                action, order_placed, order_result = reconcile_position(
                    target, current_size, client, mid, mode,
                )

                # Log
                entry = CycleLog(
                    timestamp=now.isoformat(),
                    symbol=symbol,
                    signal=signal,
                    target=target,
                    current_position_size=current_size,
                    action=action,
                    order_placed=order_placed,
                    order_result=order_result,
                    kill_switch_status=kill_status,
                    reason=f"signal={signal.direction}, composite={signal.composite}",
                )
                log_cycle(entry)

                # Print summary
                dir_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal.direction]
                tag = "▶" if order_placed else "·"
                print(f"[{now_str}] {tag} {symbol}: {dir_str} "
                      f"(c={signal.composite:+.3f}) | "
                      f"target=${target.size_usd:.0f} | "
                      f"action={action} | "
                      f"bars={signal.n_bars_today}")

        except KeyboardInterrupt:
            print("\nStopping signal bridge.")
            break
        except Exception as e:
            log.error(f"Cycle {cycle} error: {e}", exc_info=True)

        time.sleep(cycle_seconds)


# ── T17: bridge daemon (executes LIVE lifecycle signals under risk gating) ──

log = logging.getLogger("nat.signal_bridge")


def load_config(path: Path | str | None = None) -> dict:
    """Bridge daemon config (paths + poll + account). Mode defaults to dry-run."""
    defaults = {
        "poll_interval_s": CYCLE_SECONDS,
        "account_value_usd": 10000.0,
        "mode": "dry-run",  # dry-run | paper | live (live requires explicit opt-in)
        "db_path": str(ROOT / "data" / "nat.db"),
        "halt_state_path": str(ROOT / "data" / "risk" / "halt_state.json"),
        "daily_pnl_path": str(EXEC_LOG_DIR / "daily_pnl.json"),
        "fills_dir": str(EXEC_LOG_DIR),
        "pid_file": str(EXEC_LOG_DIR / "signal_bridge.pid"),
        "heartbeat_path": str(EXEC_LOG_DIR / "signal_bridge.heartbeat"),
    }
    cfg_path = Path(path) if path else ROOT / "config" / "execution.toml"
    if cfg_path.exists():
        try:
            import tomllib

            with open(cfg_path, "rb") as f:
                defaults.update(tomllib.load(f).get("bridge", {}))
        except Exception as e:  # pragma: no cover
            log.warning("Failed to read %s: %s — using defaults", cfg_path, e)
    return defaults


def healthy(config: dict | None = None, now: float | None = None) -> bool:
    """True if the daemon heartbeat is fresh (< 3 poll intervals, min 180s)."""
    cfg = config or load_config()
    hb = Path(cfg["heartbeat_path"])
    if not hb.exists():
        return False
    now = now if now is not None else time.time()
    max_age = max(180, 3 * int(cfg.get("poll_interval_s", 300)))
    return (now - hb.stat().st_mtime) < max_age


class SignalBridgeDaemon:
    """Polls LIVE lifecycle signals and executes them under risk gating.

    Hard rules: the halt check runs at the TOP of every cycle and cannot be
    skipped; sizing is portfolio-level (risk parity), never independent; the
    daemon is dry-run unless mode=='live' AND dry_run is False. The daily P&L
    rollup writes data/execution/daily_pnl.json in the shape the kill-switch
    reads — closing the kill-switch ↔ bridge loop.

    The expensive/exchange-touching steps are seams (_read_halt / _live_signals /
    _size / _signal_value / _mid / _execute / _log_fill / _rollup_daily_pnl /
    _now) so the cycle logic is unit-tested without real execution.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        lc=None,
        client=None,
        clock=None,
        dry_run: bool = True,
    ):
        self.config = config or load_config()
        self.dry_run = dry_run
        self.mode = self.config.get("mode", "dry-run")
        self.account_value = float(self.config.get("account_value_usd", 10000.0))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.client = client  # None in dry-run; a real client only for live
        self.halt_path = Path(self.config["halt_state_path"])
        self.daily_pnl_path = Path(self.config["daily_pnl_path"])
        self.fills_dir = Path(self.config["fills_dir"])
        self.pid_file = Path(self.config["pid_file"])
        self.heartbeat_path = Path(self.config["heartbeat_path"])
        self._shutdown = False
        if lc is not None:
            self.lc = lc
        else:
            from signal_lifecycle import SignalLifecycle

            self.lc = SignalLifecycle(db_path=self.config["db_path"])

    def _now(self):
        return self._clock()

    def _live(self) -> bool:
        return self.mode == "live" and not self.dry_run

    # -- seams --------------------------------------------------------------
    def _read_halt(self):
        from risk.kill_switch import read_halt_state

        return read_halt_state(self.halt_path)

    def _live_signals(self) -> list[dict]:
        return self.lc.list_signals(state="LIVE")

    def _size(self, signals: list[dict]) -> list[float]:
        """Portfolio weights via risk parity (never independent sizing)."""
        from agent.meta_portfolio import compute_risk_parity_weights

        prepared = [
            {"ic_history": (s.get("metadata") or {}).get("ic_history", [])}
            for s in signals
        ]
        return compute_risk_parity_weights(prepared)

    def _symbol(self, sig: dict) -> str:
        return (sig.get("metadata") or {}).get("symbol") or (SYMBOLS[0] if SYMBOLS else "BTC")

    def _mid(self, sig: dict) -> float:  # pragma: no cover - exchange query at live
        return 1.0

    def _signal_value(self, sig: dict):  # pragma: no cover - real feed at live
        """Latest signal value for a LIVE signal's algorithm; None if unavailable.

        Real bar-load + algorithm evaluation is wired at live deployment; until
        then this returns None (no trade), which is safe — there are no LIVE
        signals pre-paper-window anyway. The cycle plumbing (halt gate, sizing,
        fills, rollup) is fully exercised by the planted tests, which inject this.
        """
        return None

    def _execute(self, sig: dict, weight: float, now) -> dict | None:
        val = self._signal_value(sig)
        if val is None or val == 0:
            return None
        side = "BUY" if val > 0 else "SELL"
        mid = self._mid(sig)
        size_usd = self.account_value * float(weight)
        if self._live() and self.client is not None:  # pragma: no cover - live only
            self.client.place_order(
                symbol=self._symbol(sig), side=side, size_usd=size_usd, price=mid
            )
        return {
            "timestamp": now.isoformat(),
            "signal_id": sig["signal_id"],
            "signal_name": sig.get("name"),
            "symbol": self._symbol(sig),
            "side": side,
            "entry_price": float(mid),
            "entry_size": (size_usd / mid) if mid else 0.0,
            "weight": float(weight),
            "pnl": 0.0,
            "pnl_pct": 0.0,  # realized P&L is filled on close; open/dry-run => 0
            "ic": float((sig.get("metadata") or {}).get("latest_ic", 0.0) or 0.0),
            "mode": "live" if self._live() else "dry-run",
        }

    def _log_fill(self, fill: dict, now) -> None:
        self.fills_dir.mkdir(parents=True, exist_ok=True)
        path = self.fills_dir / f"fills_{now.strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(fill) + "\n")

    def _rollup_daily_pnl(self, now) -> None:
        """Aggregate fills_*.jsonl pnl_pct per date → daily_pnl.json [{date,pnl_pct}].

        This is exactly the shape kill_switch.load_pnl_history reads back.
        """
        by_date: dict[str, float] = {}
        if self.fills_dir.exists():
            for fp in sorted(self.fills_dir.glob("fills_*.jsonl")):
                for ln in fp.read_text().splitlines():
                    if not ln.strip():
                        continue
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    d = (r.get("timestamp") or "")[:10]
                    if d:
                        by_date[d] = by_date.get(d, 0.0) + float(r.get("pnl_pct", 0.0))
        out = [{"date": d, "pnl_pct": round(v, 6)} for d, v in sorted(by_date.items())]
        self.daily_pnl_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.daily_pnl_path.with_suffix(self.daily_pnl_path.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        os.replace(tmp, self.daily_pnl_path)

    # -- cycle / loop -------------------------------------------------------
    def run_cycle(self) -> dict:
        now = self._now()
        halt = self._read_halt()
        if halt.halted:  # CANNOT be skipped — gate before any signal eval/order
            log.warning("HALT active (%s) — skipping bridge cycle", halt.level)
            return {"halted": True, "picked_up": [], "fills": 0}
        sigs = self._live_signals()
        if not sigs:
            log.info("no LIVE signals — nothing to execute")
            return {"halted": False, "picked_up": [], "fills": 0}
        weights = self._size(sigs)
        picked, n_fills = [], 0
        for sig, w in zip(sigs, weights):
            picked.append(sig["signal_id"])
            fill = self._execute(sig, w, now)
            if fill:
                self._log_fill(fill, now)
                n_fills += 1
        self._rollup_daily_pnl(now)
        log.info("bridge cycle: %d LIVE signals, %d fills (%s)",
                 len(picked), n_fills, "dry-run" if not self._live() else "live")
        return {"halted": False, "picked_up": picked, "fills": n_fills}

    def status(self) -> dict:
        halt = self._read_halt()
        return {
            "mode": "live" if self._live() else "dry-run",
            "halted": halt.halted,
            "halt_level": halt.level,
            "live_signals": [s["signal_id"] for s in self._live_signals()],
            "account_value_usd": self.account_value,
        }

    def run(self) -> None:
        poll = int(self.config["poll_interval_s"])
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.write_text(str(os.getpid()))
        import signal as _signal

        _signal.signal(_signal.SIGTERM, self._handle_signal)
        _signal.signal(_signal.SIGINT, self._handle_signal)
        log.info("Signal-bridge daemon started (PID %d, poll %ds, mode %s)",
                 os.getpid(), poll, "live" if self._live() else "dry-run")
        try:
            while not self._shutdown:
                try:
                    self.run_cycle()
                except Exception as e:
                    log.error("bridge cycle error: %s", e, exc_info=True)
                self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                self.heartbeat_path.write_text(self._now().isoformat())
                for _ in range(poll):
                    if self._shutdown:
                        break
                    time.sleep(1)
        finally:
            self.pid_file.unlink(missing_ok=True)
            log.info("Signal-bridge daemon stopped")

    def _handle_signal(self, signum, frame):  # pragma: no cover
        log.info("Received signal %d, shutting down after cycle", signum)
        self._shutdown = True


def _daemon_main(argv: list[str]) -> int:
    from logging_config import setup_logging

    setup_logging("nat.signal_bridge")
    cmd = argv[0]
    if cmd == "health":
        return 0 if healthy(load_config()) else 1
    cfg = load_config()
    dry = ("--dry-run" in argv) or (cfg.get("mode", "dry-run") != "live")
    d = SignalBridgeDaemon(config=cfg, dry_run=dry)
    if cmd == "start":
        d.run()
        return 0
    if cmd == "once":
        print(json.dumps(d.run_cycle(), indent=2))
        return 0
    if cmd == "status":
        print(json.dumps(d.status(), indent=2))
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


# ── CLI ──────────────────────────────────────────────────────────────────

def _run_bridge_cli():
    parser = argparse.ArgumentParser(description="Signal-to-Order Bridge")
    parser.add_argument("--mode", choices=["dry-run", "paper", "live"],
                        default="dry-run",
                        help="dry-run: log only, paper: exchange dry-run, live: real orders")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--account-value", type=float, default=10000.0,
                        help="Account value in USD for position sizing")
    parser.add_argument("--weeks-deployed", type=int, default=0,
                        help="Weeks since deployment start (for scale schedule)")
    parser.add_argument("--cycle", type=int, default=CYCLE_SECONDS,
                        help="Cycle interval in seconds")
    args = parser.parse_args()

    from logging_config import setup_logging
    setup_logging("nat.signal_bridge")

    run_bridge(
        mode=args.mode,
        symbols=args.symbols,
        account_value_usd=args.account_value,
        weeks_deployed=args.weeks_deployed,
        cycle_seconds=args.cycle,
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch: daemon subcommands (start/status/once/health) vs one-shot bridge."""
    argv = argv if argv is not None else sys.argv[1:]
    if argv and argv[0] in ("start", "status", "once", "health"):
        return _daemon_main(argv)
    _run_bridge_cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
