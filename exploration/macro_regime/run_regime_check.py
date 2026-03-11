#!/usr/bin/env python3
"""
Macro Regime Detection - Quick Check Script

This script provides a simple CLI interface for:
1. Entering indicator values
2. Computing the current macro regime
3. Getting actionable recommendations

Usage:
    python run_regime_check.py                    # Interactive mode
    python run_regime_check.py --quick            # Quick check with saved data
    python run_regime_check.py --example          # Run with example data
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from config import DEFAULT_CONFIG, MacroRegime
from regime_detector import MacroRegimeDetector
from decision_engine import DecisionEngine, PreCommitmentRules
from data_manager import MacroDataManager, IndicatorInputHelper


def print_header():
    """Print script header"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " MACRO REGIME DETECTION SYSTEM ".center(68) + "║")
    print("║" + " Hierarchical Analysis: Business Cycle → Liquidity → Crypto ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()


def quick_indicators():
    """List of key indicators for quick check"""
    return [
        # Essential macro
        ("ism_pmi", "ISM Manufacturing PMI"),
        ("dxy", "US Dollar Index (DXY)"),
        ("us10y", "US 10Y Yield (%)"),
        ("credit_spreads", "HY Credit Spread (bps)"),
        # Essential crypto
        ("btc_dominance", "BTC Dominance (%)"),
        ("mvrv", "MVRV Ratio"),
        ("funding_rate", "Funding Rate (8h, %)"),
    ]


def full_indicators():
    """Full list of indicators"""
    config = DEFAULT_CONFIG
    all_indicators = {
        **config.business_cycle_indicators,
        **config.liquidity_indicators,
        **config.real_economy_indicators,
        **config.crypto_indicators,
        **config.onchain_indicators,
    }
    return [(k, v.name) for k, v in all_indicators.items()]


def interactive_quick_check():
    """Interactive quick check with essential indicators"""
    print("QUICK CHECK MODE - Enter values for key indicators\n")
    print("(Press Enter to skip, 'q' to quit)\n")

    detector = MacroRegimeDetector()
    indicators = quick_indicators()

    for key, name in indicators:
        try:
            value_str = input(f"  {name} [{key}]: ").strip()
            if value_str.lower() == 'q':
                break
            if value_str:
                value = float(value_str)
                detector.update_indicator(key, value)
        except ValueError:
            print(f"  Invalid value, skipping {key}")
        except KeyboardInterrupt:
            print("\nAborted")
            return

    print("\nComputing regime...")
    engine = DecisionEngine(detector=detector)
    engine.print_recommendation()


def interactive_full_check():
    """Interactive full check with data persistence"""
    dm = MacroDataManager()
    helper = IndicatorInputHelper(dm)

    print(helper.get_input_checklist())
    print("\n")

    print("Commands: 'info <indicator>' for details, 'compute' to run analysis, 'q' to quit\n")

    detector = MacroRegimeDetector()

    # Load any existing data
    for ind, point in dm.get_all_latest().items():
        try:
            detector.update_indicator(ind, point.value, point.timestamp)
        except Exception:
            pass

    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ('q', 'quit', 'exit'):
                break

            if user_input.lower() == 'compute':
                engine = DecisionEngine(detector=detector)
                engine.print_recommendation()
                continue

            if user_input.lower() == 'list':
                for key, name in full_indicators():
                    print(f"  {key}: {name}")
                continue

            if user_input.lower().startswith('info '):
                ind_name = user_input[5:].strip()
                print(helper.get_indicator_guidance(ind_name))
                continue

            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  <indicator> <value>  - Enter indicator value")
                print("  info <indicator>     - Get indicator guidance")
                print("  list                 - List all indicators")
                print("  compute              - Compute regime and recommendation")
                print("  q                    - Quit")
                continue

            # Parse indicator value
            parts = user_input.split()
            if len(parts) >= 2:
                indicator = parts[0]
                try:
                    value = float(parts[1])
                    detector.update_indicator(indicator, value)
                    dm.add_reading(indicator, value)
                    print(f"  ✓ {indicator} = {value}")
                except ValueError:
                    print(f"  Invalid value: {parts[1]}")
                except Exception as e:
                    print(f"  Error: {e}")
            else:
                print("  Format: <indicator> <value>")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


def run_example():
    """Run with example data to demonstrate the system"""
    print("EXAMPLE MODE - Running with sample data\n")

    engine = DecisionEngine()

    # Example: Moderate expansion scenario
    example_data = {
        # Business cycle - moderately positive
        "ism_pmi": 51.5,
        "ism_services": 52.3,
        "new_orders_inventories": 4.0,

        # Liquidity - supportive
        "dxy": 102.5,
        "us10y": 4.15,
        "yield_curve_10y2y": 0.25,
        "credit_spreads": 360,

        # Real economy
        "jobless_claims": 220000,

        # Crypto - mid-bull characteristics
        "eth_btc": 0.055,
        "btc_dominance": 50.0,

        # On-chain - healthy
        "mvrv": 1.9,
        "exchange_whale_ratio": 0.36,
        "funding_rate": 0.012,
    }

    print("Example indicator values:")
    print("-" * 40)
    for k, v in example_data.items():
        print(f"  {k}: {v}")
        engine.detector.update_indicator(k, v)
    print()

    engine.print_recommendation()

    # Also show contraction example
    print("\n\n" + "=" * 70)
    print("ALTERNATIVE SCENARIO: Contraction Signals")
    print("=" * 70 + "\n")

    engine2 = DecisionEngine()

    contraction_data = {
        "ism_pmi": 47.5,           # Below 50
        "ism_services": 48.2,
        "new_orders_inventories": -3.0,  # Destocking
        "dxy": 108.5,              # Strong dollar
        "us10y": 5.2,              # High rates
        "yield_curve_10y2y": -0.3, # Inverted
        "credit_spreads": 520,     # Widening
        "jobless_claims": 285000,  # Rising
        "eth_btc": 0.042,          # Falling
        "btc_dominance": 58.0,     # Rising (risk-off)
        "mvrv": 2.8,               # Approaching overheated
        "exchange_whale_ratio": 0.52,  # Distribution
        "funding_rate": 0.08,      # Overleveraged longs
    }

    print("Contraction indicator values:")
    print("-" * 40)
    for k, v in contraction_data.items():
        print(f"  {k}: {v}")
        engine2.detector.update_indicator(k, v)
    print()

    engine2.print_recommendation()


def main():
    parser = argparse.ArgumentParser(
        description="Macro Regime Detection CLI"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check with essential indicators only"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full interactive mode with data persistence"
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run with example data to demonstrate the system"
    )

    args = parser.parse_args()

    print_header()

    if args.example:
        run_example()
    elif args.quick:
        interactive_quick_check()
    elif args.full:
        interactive_full_check()
    else:
        # Default: quick check
        print("Choose mode:")
        print("  1. Quick check (essential indicators)")
        print("  2. Full analysis (all indicators, data saved)")
        print("  3. Example demo")
        print()

        try:
            choice = input("Enter choice [1/2/3]: ").strip()

            if choice == "1":
                interactive_quick_check()
            elif choice == "2":
                interactive_full_check()
            elif choice == "3":
                run_example()
            else:
                print("Invalid choice. Running quick check...")
                interactive_quick_check()

        except KeyboardInterrupt:
            print("\nExiting...")


if __name__ == "__main__":
    main()
