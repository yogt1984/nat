"""
Data Manager for Macro Regime Detection

Handles data persistence, loading, and historical analysis.
Supports manual input, CSV import, and API integration stubs.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import csv

from config import DEFAULT_CONFIG, IndicatorConfig


@dataclass
class DataPoint:
    """A single data point for an indicator"""
    indicator: str
    value: float
    timestamp: datetime
    source: str
    notes: Optional[str] = None


class MacroDataManager:
    """
    Manages indicator data persistence and retrieval.

    Features:
    - JSON-based storage for manual inputs
    - CSV import/export
    - Historical data management
    - Rolling statistics computation
    """

    def __init__(self, data_dir: str = "./data/macro"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.data_file = self.data_dir / "indicator_data.json"
        self.history_file = self.data_dir / "regime_history.json"

        # In-memory storage
        self.data: Dict[str, List[DataPoint]] = {}
        self.regime_history: List[Dict] = []

        # Load existing data
        self._load_data()

    def _load_data(self) -> None:
        """Load data from disk"""
        if self.data_file.exists():
            with open(self.data_file, "r") as f:
                raw_data = json.load(f)
                for indicator, points in raw_data.items():
                    self.data[indicator] = [
                        DataPoint(
                            indicator=p["indicator"],
                            value=p["value"],
                            timestamp=datetime.fromisoformat(p["timestamp"]),
                            source=p["source"],
                            notes=p.get("notes"),
                        )
                        for p in points
                    ]

        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                self.regime_history = json.load(f)

    def _save_data(self) -> None:
        """Persist data to disk"""
        raw_data = {}
        for indicator, points in self.data.items():
            raw_data[indicator] = [
                {
                    "indicator": p.indicator,
                    "value": p.value,
                    "timestamp": p.timestamp.isoformat(),
                    "source": p.source,
                    "notes": p.notes,
                }
                for p in points
            ]

        with open(self.data_file, "w") as f:
            json.dump(raw_data, f, indent=2)

    def save_regime_history(self, regime_output: Dict) -> None:
        """Save a regime detection output to history"""
        self.regime_history.append(regime_output)
        with open(self.history_file, "w") as f:
            json.dump(self.regime_history, f, indent=2)

    def add_reading(
        self,
        indicator: str,
        value: float,
        timestamp: datetime = None,
        source: str = "manual",
        notes: str = None
    ) -> None:
        """Add a new indicator reading"""
        timestamp = timestamp or datetime.now()

        point = DataPoint(
            indicator=indicator,
            value=value,
            timestamp=timestamp,
            source=source,
            notes=notes,
        )

        if indicator not in self.data:
            self.data[indicator] = []

        self.data[indicator].append(point)
        self._save_data()

    def get_latest(self, indicator: str) -> Optional[DataPoint]:
        """Get the most recent reading for an indicator"""
        if indicator not in self.data or not self.data[indicator]:
            return None
        return max(self.data[indicator], key=lambda p: p.timestamp)

    def get_all_latest(self) -> Dict[str, DataPoint]:
        """Get the most recent reading for all indicators"""
        return {
            ind: self.get_latest(ind)
            for ind in self.data
            if self.get_latest(ind) is not None
        }

    def get_history(
        self,
        indicator: str,
        lookback_days: int = 365
    ) -> List[DataPoint]:
        """Get historical readings for an indicator"""
        if indicator not in self.data:
            return []

        cutoff = datetime.now() - timedelta(days=lookback_days)
        return [
            p for p in self.data[indicator]
            if p.timestamp >= cutoff
        ]

    def get_statistics(self, indicator: str, lookback_days: int = 365) -> Dict[str, float]:
        """Compute statistics for an indicator"""
        history = self.get_history(indicator, lookback_days)
        if not history:
            return {}

        values = [p.value for p in history]

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "count": len(values),
            "current": values[-1] if values else None,
            "percentile_current": float(np.sum(np.array(values) <= values[-1]) / len(values) * 100) if values else None,
        }

    def import_from_csv(self, filepath: str) -> int:
        """
        Import indicator data from CSV.

        Expected CSV format:
        date,indicator,value,source,notes
        2024-01-15,ism_pmi,50.3,manual,
        2024-01-15,dxy,103.5,manual,

        Returns number of records imported.
        """
        count = 0
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = datetime.strptime(row["date"], "%Y-%m-%d")
                self.add_reading(
                    indicator=row["indicator"],
                    value=float(row["value"]),
                    timestamp=timestamp,
                    source=row.get("source", "csv_import"),
                    notes=row.get("notes"),
                )
                count += 1

        return count

    def export_to_csv(self, filepath: str) -> int:
        """Export all data to CSV"""
        rows = []
        for indicator, points in self.data.items():
            for p in points:
                rows.append({
                    "date": p.timestamp.strftime("%Y-%m-%d"),
                    "indicator": p.indicator,
                    "value": p.value,
                    "source": p.source,
                    "notes": p.notes or "",
                })

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        return len(rows)

    def to_dataframe(self, indicator: str = None) -> pd.DataFrame:
        """Convert data to pandas DataFrame"""
        rows = []

        indicators = [indicator] if indicator else list(self.data.keys())

        for ind in indicators:
            if ind not in self.data:
                continue
            for p in self.data[ind]:
                rows.append({
                    "timestamp": p.timestamp,
                    "indicator": p.indicator,
                    "value": p.value,
                    "source": p.source,
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

        return df

    def get_regime_history_df(self) -> pd.DataFrame:
        """Convert regime history to DataFrame"""
        if not self.regime_history:
            return pd.DataFrame()

        return pd.DataFrame(self.regime_history)


class IndicatorInputHelper:
    """
    Helper class for manual indicator input.

    Provides prompts, validation, and guidance for each indicator.
    """

    def __init__(self, data_manager: MacroDataManager):
        self.dm = data_manager
        self.config = DEFAULT_CONFIG

    def get_all_indicators(self) -> Dict[str, IndicatorConfig]:
        """Get all configured indicators"""
        return {
            **self.config.business_cycle_indicators,
            **self.config.liquidity_indicators,
            **self.config.real_economy_indicators,
            **self.config.crypto_indicators,
            **self.config.onchain_indicators,
        }

    def get_indicator_guidance(self, indicator: str) -> str:
        """Get guidance text for entering an indicator"""
        all_indicators = self.get_all_indicators()

        if indicator not in all_indicators:
            return f"Unknown indicator: {indicator}"

        cfg = all_indicators[indicator]
        latest = self.dm.get_latest(indicator)
        stats = self.dm.get_statistics(indicator)

        guidance = f"""
{cfg.name}
{'-' * len(cfg.name)}
{cfg.description}

Thresholds:
  - Expansion signal:   {'>' if cfg.direction == 1 else '<'} {cfg.expansion_threshold}
  - Contraction signal: {'<' if cfg.direction == 1 else '>'} {cfg.contraction_threshold}

"""
        if latest:
            guidance += f"Latest reading: {latest.value} ({latest.timestamp.strftime('%Y-%m-%d')})\n"

        if stats:
            guidance += f"""
Historical stats (past year):
  - Mean:   {stats.get('mean', 'N/A'):.2f}
  - Std:    {stats.get('std', 'N/A'):.2f}
  - Range:  [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]
"""

        return guidance

    def validate_input(self, indicator: str, value: float) -> Tuple[bool, str]:
        """Validate an indicator input value"""
        all_indicators = self.get_all_indicators()

        if indicator not in all_indicators:
            return False, f"Unknown indicator: {indicator}"

        cfg = all_indicators[indicator]

        # Basic sanity checks
        warnings = []

        # PMI-type indicators should be 0-100
        if "pmi" in indicator.lower():
            if value < 0 or value > 100:
                return False, f"PMI should be between 0 and 100, got {value}"
            if value < 30 or value > 70:
                warnings.append(f"Unusual PMI value: {value}")

        # DXY typical range
        if indicator == "dxy":
            if value < 80 or value > 130:
                warnings.append(f"DXY outside typical range (80-130): {value}")

        # Yield checks
        if "10y" in indicator or "2y" in indicator:
            if value < -2 or value > 15:
                warnings.append(f"Unusual yield value: {value}%")

        # MVRV checks
        if indicator == "mvrv":
            if value < 0.3 or value > 5:
                warnings.append(f"MVRV outside historical range: {value}")

        msg = "Valid"
        if warnings:
            msg = "; ".join(warnings)

        return True, msg

    def get_input_checklist(self) -> str:
        """Generate a checklist of indicators to update"""
        all_indicators = self.get_all_indicators()
        categories = {
            "Business Cycle": self.config.business_cycle_indicators.keys(),
            "Liquidity": self.config.liquidity_indicators.keys(),
            "Real Economy": self.config.real_economy_indicators.keys(),
            "Crypto": self.config.crypto_indicators.keys(),
            "On-Chain": self.config.onchain_indicators.keys(),
        }

        checklist = ["=" * 50, "INDICATOR INPUT CHECKLIST", "=" * 50, ""]

        for cat_name, indicators in categories.items():
            checklist.append(f"\n{cat_name.upper()}")
            checklist.append("-" * 30)

            for ind in indicators:
                latest = self.dm.get_latest(ind)
                cfg = all_indicators[ind]

                if latest:
                    age_days = (datetime.now() - latest.timestamp).days
                    status = f"✓ {latest.value:.2f} ({age_days}d ago)"
                    if age_days > 7:
                        status = f"⚠ {latest.value:.2f} ({age_days}d ago - STALE)"
                else:
                    status = "✗ No data"

                checklist.append(f"  [{status:30}] {cfg.name}")

        return "\n".join(checklist)

    def interactive_input(self) -> None:
        """Interactive command-line input session"""
        print(self.get_input_checklist())
        print("\n" + "=" * 50)
        print("Enter values for indicators (or 'q' to quit)")
        print("Format: indicator_name value")
        print("Example: ism_pmi 52.3")
        print("=" * 50 + "\n")

        all_indicators = self.get_all_indicators()

        while True:
            try:
                user_input = input("> ").strip()

                if user_input.lower() in ("q", "quit", "exit"):
                    break

                if user_input.lower() == "help":
                    print("\nAvailable indicators:")
                    for name in sorted(all_indicators.keys()):
                        print(f"  - {name}")
                    continue

                if user_input.lower().startswith("info "):
                    ind_name = user_input[5:].strip()
                    print(self.get_indicator_guidance(ind_name))
                    continue

                parts = user_input.split()
                if len(parts) < 2:
                    print("Format: indicator_name value")
                    continue

                indicator = parts[0]
                try:
                    value = float(parts[1])
                except ValueError:
                    print(f"Invalid number: {parts[1]}")
                    continue

                # Validate
                valid, msg = self.validate_input(indicator, value)
                if not valid:
                    print(f"Error: {msg}")
                    continue

                if msg != "Valid":
                    print(f"Warning: {msg}")

                # Save
                self.dm.add_reading(indicator, value, source="interactive")
                print(f"Saved: {indicator} = {value}")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


# =============================================================================
# Data Sources Integration Stubs
# =============================================================================

class FREDDataSource:
    """
    Stub for FRED API integration.

    To implement:
    1. Get API key from https://fred.stlouisfed.org/docs/api/api_key.html
    2. pip install fredapi
    3. Implement fetch methods
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def fetch_series(self, series_id: str, start_date: str = None) -> pd.DataFrame:
        """Fetch a data series from FRED"""
        # TODO: Implement with fredapi
        raise NotImplementedError("FRED integration not implemented. Use manual input.")


class CryptoQuantDataSource:
    """
    Stub for CryptoQuant API integration.

    To implement:
    1. Get API access from https://cryptoquant.com/
    2. Implement fetch methods for on-chain indicators
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def fetch_mvrv(self) -> float:
        """Fetch current MVRV ratio"""
        raise NotImplementedError("CryptoQuant integration not implemented. Use manual input.")

    def fetch_exchange_whale_ratio(self) -> float:
        """Fetch exchange whale ratio"""
        raise NotImplementedError("CryptoQuant integration not implemented. Use manual input.")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Initialize
    dm = MacroDataManager()
    helper = IndicatorInputHelper(dm)

    # Print checklist
    print(helper.get_input_checklist())

    # Show guidance for one indicator
    print(helper.get_indicator_guidance("ism_pmi"))

    # Start interactive session
    helper.interactive_input()
