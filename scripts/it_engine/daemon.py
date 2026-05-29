#!/usr/bin/env python3
"""
Information-Theoretic Alpha Discovery Engine.

Continuously computes mutual information, conditional MI, interaction
information, and transfer entropy for all ingested features against
forward returns at multiple horizons. Discovers cost-viable alpha
signals via greedy forward feature selection.

Modes:
  - live:    Subscribe to Redis nat:features:{symbol} pub/sub (100ms)
  - offline: Read from parquet files in data/features/

Usage:
  python -m scripts.it_engine.daemon start --symbol BTC
  python -m scripts.it_engine.daemon start --symbol BTC --offline --data-dir data/features
  python -m scripts.it_engine.daemon status --symbol BTC
  python -m scripts.it_engine.daemon stop --symbol BTC
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ITEngineConfig
from .estimators import ksg_mi, cmi, interaction_info, linear_te, ksg_te, min_info_bits
from .feature_selector import greedy_select
from .state import ITState

log = logging.getLogger(__name__)

# Features to exclude from IT analysis (metadata, not signals)
_META_COLUMNS = {"timestamp_ms", "symbol", "date", "hour", "minute", "second"}


class RingBuffer:
    """Fixed-size ring buffer backed by a pandas DataFrame."""

    def __init__(self, maxlen: int, columns: Optional[list[str]] = None):
        self.maxlen = maxlen
        self.columns = columns
        self._buf: deque[dict] = deque(maxlen=maxlen)

    def append(self, row: dict):
        self._buf.append(row)
        if self.columns is None and len(self._buf) > 0:
            self.columns = list(row.keys())

    def extend(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            self._buf.append(row.to_dict())
        if self.columns is None and len(df.columns) > 0:
            self.columns = list(df.columns)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._buf:
            return pd.DataFrame()
        return pd.DataFrame(list(self._buf))

    def __len__(self):
        return len(self._buf)


class ITEngine:
    """Main IT engine — computes information metrics and selects features."""

    def __init__(self, config: ITEngineConfig, symbol: str):
        self.config = config
        self.symbol = symbol
        self.buffer = RingBuffer(maxlen=config.buffer_size)
        self.state = ITState.load(symbol, config.state_dir)
        self.state.symbol = symbol
        self._running = True

        # Resolve feature columns from first data batch
        self._feature_cols: Optional[list[str]] = None
        self._entropy_cols: list[str] = []

    def _resolve_columns(self, df: pd.DataFrame):
        """Identify feature and entropy columns from data."""
        all_cols = [c for c in df.columns if c not in _META_COLUMNS]
        # Only numeric columns
        numeric = [c for c in all_cols if df[c].dtype.kind in ('f', 'i', 'u')]
        self._feature_cols = numeric

        # Find entropy conditioning columns (match prefixes from config)
        self._entropy_cols = [
            c for c in numeric
            if any(c.startswith(e.rstrip("_")) or c == e
                   for e in self.config.entropy_conditioning)
        ]
        if not self._entropy_cols:
            # Fallback: any column starting with "ent_"
            self._entropy_cols = [c for c in numeric if c.startswith("ent_")]

        log.info(
            "Resolved %d feature cols, %d entropy cols for %s",
            len(self._feature_cols), len(self._entropy_cols), self.symbol,
        )

    def _compute_forward_returns(
        self, df: pd.DataFrame, horizon: int, stride: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward returns at tick horizon with stride subsampling.

        Returns (fwd_returns, indices) where indices are the row positions
        of the subsampled points. Striding breaks the overlap autocorrelation
        that inflates MI estimates at long horizons.
        """
        if "raw_midprice" in df.columns:
            mid = df["raw_midprice"].values
        elif "raw_microprice" in df.columns:
            mid = df["raw_microprice"].values
        else:
            # Fallback: first price-like column
            price_cols = [c for c in df.columns if "price" in c.lower()]
            if not price_cols:
                idx = np.arange(0, len(df), stride)
                return np.full(len(idx), np.nan), idx
            mid = df[price_cols[0]].values

        n = len(mid)
        valid = n - horizon
        if valid <= 0:
            idx = np.arange(0, n, stride)
            return np.full(len(idx), np.nan), idx

        indices = np.arange(0, valid, stride)
        fwd = (mid[indices + horizon] - mid[indices]) / mid[indices] * 10000  # bps
        return fwd, indices

    def cycle(self):
        """Run one IT computation cycle."""
        df = self.buffer.to_dataframe()
        if len(df) < 100:
            log.debug("Buffer too small (%d rows), skipping cycle", len(df))
            return

        if self._feature_cols is None:
            self._resolve_columns(df)

        self.state.n_samples = len(df)
        k = self.config.ksg_k

        # --- Phase 1: MI and CMI for each feature × horizon ---
        mi_matrix = {}
        cmi_matrix = {}
        ii_dict = {}

        # Build entropy conditioning matrix Z (cap dimensionality)
        z_cols = [c for c in self._entropy_cols if c in df.columns]
        if len(z_cols) > self.config.cmi_max_z_dims:
            # Keep top columns by variance
            z_vars = {c: float(np.nanvar(df[c].values)) for c in z_cols}
            z_cols = sorted(z_vars, key=z_vars.get, reverse=True)[
                :self.config.cmi_max_z_dims
            ]
        Z = df[z_cols].values if z_cols else None

        for horizon in self.config.horizons:
            h_label = f"{horizon}t"
            stride = max(1, horizon // self.config.stride_divisor)
            fwd_ret, indices = self._compute_forward_returns(df, horizon, stride)
            valid_mask = np.isfinite(fwd_ret)

            if valid_mask.sum() < 50:
                continue

            r = fwd_ret[valid_mask]
            valid_idx = indices[valid_mask]
            sigma_r = float(np.std(r))

            for feat in self._feature_cols:
                if feat in _META_COLUMNS:
                    continue

                f_vals = df[feat].values[valid_idx]
                if np.std(f_vals) < 1e-12:
                    continue  # constant feature

                # MI
                mi_val = ksg_mi(f_vals, r, k=k)
                mi_matrix.setdefault(feat, {})[h_label] = mi_val

                # CMI and interaction info (if entropy columns available)
                if Z is not None and Z.shape[1] > 0:
                    if len(f_vals) < self.config.cmi_min_samples:
                        log.debug(
                            "Skipping CMI for %s at %s: %d samples (need %d)",
                            feat, h_label, len(f_vals),
                            self.config.cmi_min_samples,
                        )
                        continue
                    z_valid = Z[valid_idx]
                    cmi_val = cmi(f_vals, r, z_valid, k=k)
                    cmi_matrix.setdefault(feat, {})[h_label] = cmi_val

                    ii_val = cmi_val - mi_val
                    ii_dict[feat] = ii_val  # last horizon wins (most relevant)

        # --- Phase 2: Transfer entropy for top-N features ---
        te_dict = {}
        if mi_matrix:
            # Rank features by max MI across horizons
            max_mi = {
                feat: max(h.values()) for feat, h in mi_matrix.items() if h
            }
            top_feats = sorted(max_mi, key=max_mi.get, reverse=True)[
                :self.config.te_top_n
            ]

            # Use longest horizon for TE (most samples)
            max_horizon = max(self.config.horizons)
            te_stride = max(1, max_horizon // self.config.stride_divisor)
            fwd_ret, te_indices = self._compute_forward_returns(
                df, max_horizon, te_stride
            )
            valid_mask = np.isfinite(fwd_ret)

            if valid_mask.sum() > 50:
                r = fwd_ret[valid_mask]
                valid_idx = te_indices[valid_mask]
                te_func = ksg_te if self.config.te_method == "ksg" else linear_te
                te_kwargs = dict(
                    lag=self.config.te_lag,
                    order=self.config.te_order,
                )
                if self.config.te_method == "ksg":
                    te_kwargs["k"] = k
                for feat in top_feats:
                    f_vals = df[feat].values[valid_idx]
                    if np.std(f_vals) < 1e-12:
                        continue
                    te_val = te_func(f_vals, r, **te_kwargs)
                    te_dict.setdefault(feat, {})["returns"] = te_val

                    # Also compute TE from entropy to returns
                    for ec in z_cols[:3]:
                        e_vals = df[ec].values[valid_idx]
                        if np.std(e_vals) < 1e-12:
                            continue
                        te_ent = te_func(e_vals, r, **te_kwargs)
                        te_dict.setdefault(ec, {})["returns"] = te_ent

        # --- Phase 3: Greedy feature selection ---
        # Use the horizon with highest average MI
        best_horizon = None
        best_avg_mi = -1
        for h in self.config.horizons:
            h_label = f"{h}t"
            vals = [mi_matrix[f].get(h_label, 0) for f in mi_matrix]
            if vals:
                avg = np.mean(vals)
                if avg > best_avg_mi:
                    best_avg_mi = avg
                    best_horizon = h

        selected = []
        if best_horizon is not None:
            h_label = f"{best_horizon}t"
            gs_stride = max(1, best_horizon // self.config.stride_divisor)
            fwd_ret, gs_indices = self._compute_forward_returns(
                df, best_horizon, gs_stride
            )
            valid_mask = np.isfinite(fwd_ret)
            if valid_mask.sum() > 50:
                r = fwd_ret[valid_mask]
                valid_idx = gs_indices[valid_mask]
                sigma_r = float(np.std(r))
                feat_arrays = {
                    feat: df[feat].values[valid_idx]
                    for feat in mi_matrix
                    if feat in df.columns
                    and np.std(df[feat].values[valid_idx]) > 1e-12
                }
                selected = greedy_select(
                    features=feat_arrays,
                    returns=r,
                    fee_rt_bps=self.config.costs.default_fee_rt_bps,
                    sigma_r_bps=sigma_r,
                    max_features=self.config.max_features_greedy,
                    k=k,
                )

        # --- Phase 4: Cost viability ---
        cost_viable = {}
        for feat, horizons in mi_matrix.items():
            for h_label, mi_val in horizons.items():
                h_ticks = int(h_label.rstrip("t"))
                cv_stride = max(1, h_ticks // self.config.stride_divisor)
                fwd_ret, _ = self._compute_forward_returns(
                    df, h_ticks, cv_stride
                )
                sigma_r = float(np.nanstd(fwd_ret))
                if sigma_r > 0:
                    threshold = min_info_bits(
                        self.config.costs.default_fee_rt_bps, sigma_r
                    )
                    if mi_val > threshold:
                        cost_viable[feat] = True
                        break
            else:
                cost_viable[feat] = False

        # --- Update state ---
        self.state.mi_matrix = mi_matrix
        self.state.cmi_matrix = cmi_matrix
        self.state.interaction = ii_dict
        self.state.transfer_entropy = te_dict
        self.state.selected_features = [s["name"] for s in selected]
        self.state.cumulative_mi = [s["cumulative_mi"] for s in selected]
        self.state.cost_viable = cost_viable
        self.state.cycle_count += 1
        self.state.save(self.config.state_dir)

        # --- Summary log ---
        n_viable = sum(1 for v in cost_viable.values() if v)
        log.info(
            "Cycle %d: %d features, %d cost-viable, %d selected | %s",
            self.state.cycle_count,
            len(mi_matrix),
            n_viable,
            len(selected),
            self.symbol,
        )

    def run_offline(self, data_dir: str, dry_run: bool = False):
        """Run on saved parquet files (batch mode)."""
        from data.features import load_features

        df = load_features(
            symbols=[self.symbol],
            data_dir=Path(data_dir),
            validate=False,
        )
        if df.empty:
            log.error("No valid parquet data for %s in %s", self.symbol, data_dir)
            return

        log.info("Loaded %d rows for %s", len(df), self.symbol)

        # Fill buffer
        self.buffer.extend(df.tail(self.config.buffer_size))
        self.cycle()

        if dry_run:
            log.info("Dry run complete — 1 cycle executed")
            return

    def run_live(self):
        """Run in live mode, subscribing to Redis pub/sub."""
        try:
            import redis as redis_lib
        except ImportError:
            log.error("redis package not installed. Run: pip install redis")
            sys.exit(1)

        r = redis_lib.Redis.from_url(self.config.redis_url)
        channel = self.config.redis_feature_key.format(symbol=self.symbol)
        pubsub = r.pubsub()
        pubsub.subscribe(channel)

        log.info("Subscribed to %s", channel)

        last_compute = time.time()

        for message in pubsub.listen():
            if not self._running:
                break

            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
                features = data.get("features", data)
                self.buffer.append(features)
            except (json.JSONDecodeError, KeyError) as e:
                log.debug("Skipping malformed message: %s", e)
                continue

            now = time.time()
            if now - last_compute >= self.config.compute_interval_s:
                self.cycle()
                last_compute = now

                # Publish to Redis
                try:
                    out_key = self.config.redis_output_key.format(
                        symbol=self.symbol
                    )
                    r.hset(out_key, mapping=self.state.to_redis_dict())
                    r.expire(out_key, 120)
                except Exception as e:
                    log.warning("Redis publish failed: %s", e)

        pubsub.close()
        log.info("IT engine stopped for %s", self.symbol)

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="it-engine",
        description="Information-Theoretic Alpha Discovery Engine",
    )
    sub = p.add_subparsers(dest="command")

    # start
    ps = sub.add_parser("start", help="Start the IT engine")
    ps.add_argument("--symbol", default="BTC", help="Symbol to analyze")
    ps.add_argument("--config", default="config/it_engine.toml", help="Config file")
    ps.add_argument("--offline", action="store_true", help="Run on parquet files")
    ps.add_argument("--data-dir", default="data/features", help="Parquet data directory")
    ps.add_argument("--dry-run", action="store_true", help="Run 1 cycle and exit")
    ps.add_argument("-v", "--verbose", action="store_true")

    # status
    ss = sub.add_parser("status", help="Show engine status")
    ss.add_argument("--symbol", default="BTC")
    ss.add_argument("--config", default="config/it_engine.toml")
    ss.add_argument("--top", type=int, default=10, help="Top N features to show")

    # stop
    st = sub.add_parser("stop", help="Stop the engine (sends SIGTERM to PID file)")
    st.add_argument("--symbol", default="BTC")

    return p


def cmd_start(args):
    from logging_config import setup_logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging("nat.it_engine", level=level)

    config = ITEngineConfig.load(args.config)
    engine = ITEngine(config, args.symbol)

    # Write PID file for stop command
    pid_path = os.path.join(config.state_dir, f"pid_{args.symbol}")
    os.makedirs(config.state_dir, exist_ok=True)
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))

    def _sigterm(sig, frame):
        engine.stop()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    if args.offline or args.dry_run:
        engine.run_offline(args.data_dir, dry_run=args.dry_run)
    else:
        engine.run_live()


def cmd_status(args):
    config = ITEngineConfig.load(args.config)
    state = ITState.load(args.symbol, config.state_dir)

    if not state.last_updated:
        print(f"No state found for {args.symbol}")
        return

    print(f"Symbol:       {state.symbol}")
    print(f"Last updated: {state.last_updated}")
    print(f"Cycles:       {state.cycle_count}")
    print(f"Samples:      {state.n_samples}")
    print()

    # Top features by MI
    top_mi = state.top_features(args.top, by="mi")
    if top_mi:
        print(f"Top {args.top} features by MI (bits):")
        for feat, val in top_mi:
            viable = state.cost_viable.get(feat, False)
            ii = state.interaction.get(feat, 0)
            flag = " *VIABLE*" if viable else ""
            synergy = f" II={ii:+.4f}" if ii else ""
            print(f"  {feat:40s}  MI={val:.6f}{synergy}{flag}")
        print()

    # Selected features (greedy)
    if state.selected_features:
        print("Greedy selection:")
        for i, (feat, cum_mi) in enumerate(
            zip(state.selected_features, state.cumulative_mi)
        ):
            print(f"  {i+1}. {feat:40s}  cumulative={cum_mi:.6f} bits")
        print()

    # Cost-viable count
    n_viable = sum(1 for v in state.cost_viable.values() if v)
    print(f"Cost-viable features: {n_viable}/{len(state.cost_viable)}")


def cmd_stop(args):
    config = ITEngineConfig.load("config/it_engine.toml")
    pid_path = os.path.join(config.state_dir, f"pid_{args.symbol}")
    if not os.path.exists(pid_path):
        print(f"No PID file for {args.symbol}")
        return
    with open(pid_path) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to PID {pid}")
        os.remove(pid_path)
    except ProcessLookupError:
        print(f"Process {pid} not running, cleaning up PID file")
        os.remove(pid_path)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)


if __name__ == "__main__":
    main()
