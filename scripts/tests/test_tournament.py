"""Tests for the tournament engine — DB, signal adapter, lifecycle, rankings."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DB tests
# ---------------------------------------------------------------------------


class TestTournamentDB:
    """Test SQLite schema + CRUD operations."""

    def _make_db(self, tmp_path):
        from tournament.db import TournamentDB
        return TournamentDB(db_path=tmp_path / "test.db")

    def test_schema_creation(self, tmp_path):
        db = self._make_db(tmp_path)
        # Should not raise
        db.close()

    def test_insert_and_get_run(self, tmp_path):
        db = self._make_db(tmp_path)
        db.insert_run(
            algo_name="test_algo", algo_source="hand_coded",
            symbol="BTC", date="2026-06-01",
            n_trades=10, total_net_bps=5.5,
            net_bps_per_trade=0.55, win_rate=0.6,
            max_loss_bps=-2.0,
        )
        runs = db.get_runs("test_algo")
        assert len(runs) == 1
        assert runs[0]["symbol"] == "BTC"
        assert runs[0]["total_net_bps"] == 5.5
        db.close()

    def test_upsert_replaces(self, tmp_path):
        db = self._make_db(tmp_path)
        db.insert_run(
            algo_name="a", algo_source="hand_coded",
            symbol="BTC", date="2026-06-01",
            n_trades=5, total_net_bps=1.0,
            net_bps_per_trade=0.2, win_rate=0.5,
            max_loss_bps=-1.0,
        )
        db.insert_run(
            algo_name="a", algo_source="hand_coded",
            symbol="BTC", date="2026-06-01",
            n_trades=10, total_net_bps=2.0,
            net_bps_per_trade=0.2, win_rate=0.6,
            max_loss_bps=-1.0,
        )
        runs = db.get_runs("a")
        assert len(runs) == 1
        assert runs[0]["n_trades"] == 10
        db.close()

    def test_evaluated_dates(self, tmp_path):
        db = self._make_db(tmp_path)
        for date in ["2026-05-01", "2026-05-02", "2026-05-03"]:
            db.insert_run(
                algo_name="a", algo_source="hand_coded",
                symbol="BTC", date=date,
                n_trades=5, total_net_bps=1.0,
                net_bps_per_trade=0.2, win_rate=0.5,
                max_loss_bps=-1.0,
            )
        dates = db.get_evaluated_dates()
        assert dates == {"2026-05-01", "2026-05-02", "2026-05-03"}
        dates_a = db.get_evaluated_dates("a")
        assert len(dates_a) == 3
        db.close()

    def test_daily_pnl_aggregation(self, tmp_path):
        db = self._make_db(tmp_path)
        # Two symbols, same date
        db.insert_run(
            algo_name="a", algo_source="hand_coded",
            symbol="BTC", date="2026-06-01",
            n_trades=5, total_net_bps=10.0,
            net_bps_per_trade=2.0, win_rate=0.6,
            max_loss_bps=-1.0,
        )
        db.insert_run(
            algo_name="a", algo_source="hand_coded",
            symbol="ETH", date="2026-06-01",
            n_trades=3, total_net_bps=-5.0,
            net_bps_per_trade=-1.67, win_rate=0.33,
            max_loss_bps=-3.0,
        )
        pnl = db.get_daily_pnl("a")
        assert len(pnl) == 1
        assert pnl[0]["daily_bps"] == 5.0  # 10 + (-5)
        assert pnl[0]["total_trades"] == 8
        db.close()

    def test_rankings_crud(self, tmp_path):
        db = self._make_db(tmp_path)
        db.upsert_ranking(
            date="2026-06-01", algo_name="a", rank=1,
            composite_score=0.8, rolling_7d_sharpe=2.0,
            rolling_30d_sharpe=1.5, rolling_7d_win_rate=0.6,
        )
        db.upsert_ranking(
            date="2026-06-01", algo_name="b", rank=2,
            composite_score=0.5, rolling_7d_sharpe=1.0,
            rolling_30d_sharpe=0.8, rolling_7d_win_rate=0.5,
        )
        rankings = db.get_rankings("2026-06-01")
        assert len(rankings) == 2
        assert rankings[0]["algo_name"] == "a"
        assert rankings[0]["rank"] == 1

        latest = db.get_latest_rankings()
        assert len(latest) == 2
        db.close()

    def test_algorithm_status(self, tmp_path):
        db = self._make_db(tmp_path)
        db.upsert_status(
            algo_name="test", status="candidate",
            source="hand_coded",
        )
        s = db.get_status("test")
        assert s is not None
        assert s["status"] == "candidate"

        db.upsert_status(
            algo_name="test", status="active",
            source="hand_coded", reason="activated",
        )
        s = db.get_status("test")
        assert s["status"] == "active"

        by_status = db.get_by_status("active")
        assert len(by_status) == 1
        db.close()

    def test_rolling_sharpe(self, tmp_path):
        db = self._make_db(tmp_path)
        # Insert 10 days of data
        for i in range(10):
            pnl = 5.0 if i % 2 == 0 else -2.0  # alternating
            db.insert_run(
                algo_name="a", algo_source="hand_coded",
                symbol="BTC", date=f"2026-06-{i + 1:02d}",
                n_trades=5, total_net_bps=pnl,
                net_bps_per_trade=pnl / 5, win_rate=0.6 if pnl > 0 else 0.4,
                max_loss_bps=-1.0,
            )
        sharpe = db.compute_rolling_sharpe("a", 30)
        # Should be positive (more up days than down)
        assert sharpe > 0
        db.close()


# ---------------------------------------------------------------------------
# Signal adapter tests
# ---------------------------------------------------------------------------


class TestSignalAdapter:
    """Test SignalAlgorithm wrapping."""

    def _make_signal_spec(self, **overrides):
        spec = {
            "hypothesis_id": "HYP-SYS-abc12345",
            "features": ["imbalance_qty_l5"],
            "regime_gate": "ent_book_shape < P30",
            "extraction": "raw",
            "horizon_s": 5.0,
            "expected_ic": 0.12,
            "symbols": ["BTC", "ETH", "SOL"],
            "status": "validated",
            "_source_agent": "microstructure",
        }
        spec.update(overrides)
        return spec

    def test_basic_creation(self):
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec())
        assert sa.name() == "sig_HYP-SYS-"
        assert len(sa.alg_features()) == 2
        assert len(sa.required_columns()) == 2  # imbalance_qty_l5 + ent_book_shape
        assert sa.polarity == "high_long"
        assert sa.source_agent == "microstructure"

    def test_negative_ic_polarity(self):
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec(expected_ic=-0.08))
        assert sa.polarity == "low_long"

    def test_step_returns_correct_keys(self):
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec())
        result = sa.step({"imbalance_qty_l5": 1.5, "ent_book_shape": 0.3})
        feature_names = sa.feature_names
        assert set(result.keys()) == set(feature_names)

    def test_step_nan_input(self):
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec())
        result = sa.step({"imbalance_qty_l5": np.nan, "ent_book_shape": 0.3})
        assert all(np.isnan(v) for v in result.values())

    def test_run_batch(self):
        import pandas as pd
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec())

        n = 200
        df = pd.DataFrame({
            "imbalance_qty_l5": np.random.randn(n),
            "ent_book_shape": np.random.uniform(0, 1, n),
        })
        result = sa.run_batch(df)
        assert len(result) == n
        assert len(result.columns) == 2
        # Warmup should be NaN
        assert result.iloc[0].isna().all()
        # After warmup, some values should be finite
        assert result.iloc[150:].notna().any().any()

    def test_run_batch_missing_column(self):
        import pandas as pd
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec())

        df = pd.DataFrame({"unrelated_col": [1.0, 2.0, 3.0]})
        result = sa.run_batch(df)
        assert result.isna().all().all()

    def test_no_gate(self):
        from algorithms.signal_adapter import SignalAlgorithm
        sa = SignalAlgorithm(self._make_signal_spec(regime_gate=None))
        assert len(sa.required_columns()) == 1  # just the signal feature


class TestRegimeGateParsing:
    """Test the regime gate string parser."""

    def test_less_than(self):
        from algorithms.signal_adapter import _parse_regime_gate
        result = _parse_regime_gate("ent_book_shape < P30")
        assert result == {"feature": "ent_book_shape", "op": "<", "percentile": 30.0}

    def test_greater_than(self):
        from algorithms.signal_adapter import _parse_regime_gate
        result = _parse_regime_gate("vol_zscore > P80")
        assert result == {"feature": "vol_zscore", "op": ">", "percentile": 80.0}

    def test_decimal_percentile(self):
        from algorithms.signal_adapter import _parse_regime_gate
        result = _parse_regime_gate("entropy > P33.3")
        assert result["percentile"] == 33.3

    def test_empty(self):
        from algorithms.signal_adapter import _parse_regime_gate
        assert _parse_regime_gate("") is None
        assert _parse_regime_gate(None) is None

    def test_invalid(self):
        from algorithms.signal_adapter import _parse_regime_gate
        assert _parse_regime_gate("some random string") is None


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Test degradation and promotion state transitions."""

    def _make_db(self, tmp_path):
        from tournament.db import TournamentDB
        return TournamentDB(db_path=tmp_path / "test.db")

    def _insert_days(self, db, algo, days_data):
        """Insert (date, pnl) pairs across BTC."""
        for date, pnl in days_data:
            db.insert_run(
                algo_name=algo, algo_source="hand_coded",
                symbol="BTC", date=date,
                n_trades=5, total_net_bps=pnl,
                net_bps_per_trade=pnl / 5, win_rate=0.6 if pnl > 0 else 0.3,
                max_loss_bps=-abs(pnl) / 5,
            )

    def test_candidate_to_active(self, tmp_path):
        from tournament.lifecycle import run_lifecycle_checks
        db = self._make_db(tmp_path)
        self._insert_days(db, "algo1", [
            ("2026-06-01", 5.0),
            ("2026-06-02", 3.0),
            ("2026-06-03", 2.0),
        ])
        db.upsert_status(algo_name="algo1", status="candidate", source="hand_coded")

        config = {"degradation": {"activation_days": 3, "probation_after_days": 5,
                                  "retire_after_days": 14, "min_sharpe_7d": 0.0,
                                  "min_win_rate_7d": 0.4, "recovery_days": 3},
                  "promotion": {"min_days_active": 14, "min_sharpe_30d": 1.0,
                                "min_win_rate_30d": 0.5, "min_symbols_profitable": 2}}
        result = run_lifecycle_checks(db, config)
        assert "algo1" in result["activated"]
        assert db.get_status("algo1")["status"] == "active"
        db.close()

    def test_active_to_probation(self, tmp_path):
        from tournament.lifecycle import run_lifecycle_checks
        db = self._make_db(tmp_path)
        # 6 consecutive losing days
        self._insert_days(db, "algo1", [
            (f"2026-06-{i:02d}", -3.0) for i in range(1, 7)
        ])
        db.upsert_status(algo_name="algo1", status="active", source="hand_coded")

        config = {"degradation": {"activation_days": 3, "probation_after_days": 5,
                                  "retire_after_days": 14, "min_sharpe_7d": 0.0,
                                  "min_win_rate_7d": 0.4, "recovery_days": 3},
                  "promotion": {"min_days_active": 14, "min_sharpe_30d": 1.0,
                                "min_win_rate_30d": 0.5, "min_symbols_profitable": 2}}
        result = run_lifecycle_checks(db, config)
        assert "algo1" in result["degraded"]
        assert db.get_status("algo1")["status"] == "probation"
        db.close()

    def test_probation_to_retired(self, tmp_path):
        from tournament.lifecycle import run_lifecycle_checks
        db = self._make_db(tmp_path)
        # 15 consecutive losing days
        self._insert_days(db, "algo1", [
            (f"2026-06-{i:02d}", -2.0) for i in range(1, 16)
        ])
        db.upsert_status(algo_name="algo1", status="probation", source="hand_coded")

        config = {"degradation": {"activation_days": 3, "probation_after_days": 5,
                                  "retire_after_days": 14, "min_sharpe_7d": 0.0,
                                  "min_win_rate_7d": 0.4, "recovery_days": 3},
                  "promotion": {"min_days_active": 14, "min_sharpe_30d": 1.0,
                                "min_win_rate_30d": 0.5, "min_symbols_profitable": 2}}
        result = run_lifecycle_checks(db, config)
        assert "algo1" in result["retired"]
        assert db.get_status("algo1")["status"] == "retired"
        db.close()

    def test_probation_recovery(self, tmp_path):
        from tournament.lifecycle import run_lifecycle_checks
        db = self._make_db(tmp_path)
        # 3 losing + 3 winning (recent = winning)
        self._insert_days(db, "algo1", [
            ("2026-06-01", -3.0), ("2026-06-02", -2.0), ("2026-06-03", -1.0),
            ("2026-06-04", 5.0), ("2026-06-05", 4.0), ("2026-06-06", 3.0),
        ])
        db.upsert_status(algo_name="algo1", status="probation", source="hand_coded")

        config = {"degradation": {"activation_days": 3, "probation_after_days": 5,
                                  "retire_after_days": 14, "min_sharpe_7d": 0.0,
                                  "min_win_rate_7d": 0.4, "recovery_days": 3},
                  "promotion": {"min_days_active": 14, "min_sharpe_30d": 1.0,
                                "min_win_rate_30d": 0.5, "min_symbols_profitable": 2}}
        result = run_lifecycle_checks(db, config)
        assert "algo1" in result["recovered"]
        assert db.get_status("algo1")["status"] == "active"
        db.close()


# ---------------------------------------------------------------------------
# Rankings tests
# ---------------------------------------------------------------------------


class TestRankings:
    """Test ranking computation."""

    def _make_db(self, tmp_path):
        from tournament.db import TournamentDB
        return TournamentDB(db_path=tmp_path / "test.db")

    def test_rankings_sorted_by_composite(self, tmp_path):
        from tournament.report import compute_rankings
        db = self._make_db(tmp_path)

        # Insert data for 3 algos with different performance
        for i, (algo, pnl) in enumerate([("winner", 10.0), ("mid", 2.0), ("loser", -5.0)]):
            for d in range(1, 8):
                db.insert_run(
                    algo_name=algo, algo_source="hand_coded",
                    symbol="BTC", date=f"2026-06-{d:02d}",
                    n_trades=5, total_net_bps=pnl,
                    net_bps_per_trade=pnl / 5,
                    win_rate=0.7 if pnl > 0 else 0.3,
                    max_loss_bps=-2.0,
                )

        config = {"scoring": {"sharpe_weight": 0.4, "win_rate_weight": 0.3, "ic_weight": 0.3}}
        rankings = compute_rankings(db, "2026-06-07", config)

        assert len(rankings) == 3
        assert rankings[0]["algo_name"] == "winner"
        assert rankings[0]["rank"] == 1
        assert rankings[-1]["algo_name"] == "loser"
        assert rankings[-1]["rank"] == 3
        db.close()

    def test_rankings_persisted(self, tmp_path):
        from tournament.report import compute_rankings
        db = self._make_db(tmp_path)

        for d in range(1, 4):
            db.insert_run(
                algo_name="a", algo_source="hand_coded",
                symbol="BTC", date=f"2026-06-{d:02d}",
                n_trades=5, total_net_bps=5.0,
                net_bps_per_trade=1.0, win_rate=0.6,
                max_loss_bps=-1.0,
            )

        config = {"scoring": {"sharpe_weight": 0.4, "win_rate_weight": 0.3, "ic_weight": 0.3}}
        compute_rankings(db, "2026-06-03", config)

        stored = db.get_rankings("2026-06-03")
        assert len(stored) == 1
        assert stored[0]["algo_name"] == "a"
        db.close()


# ---------------------------------------------------------------------------
# Report formatting tests
# ---------------------------------------------------------------------------


class TestReportFormatting:
    """Test report string formatting."""

    def test_rankings_table(self):
        from tournament.report import format_rankings_table
        rankings = [
            {"rank": 1, "algo_name": "optimal_entry", "composite_score": 0.8,
             "rolling_7d_sharpe": 2.1, "rolling_30d_sharpe": 1.5, "rolling_7d_win_rate": 0.6},
            {"rank": 2, "algo_name": "jump_detector", "composite_score": 0.6,
             "rolling_7d_sharpe": 1.0, "rolling_30d_sharpe": 0.8, "rolling_7d_win_rate": 0.5},
        ]
        table = format_rankings_table(rankings)
        assert "optimal_entry" in table
        assert "jump_detector" in table

    def test_telegram_summary(self):
        from tournament.report import format_telegram_summary
        rankings = [
            {"algo_name": "a", "rolling_7d_sharpe": 2.0},
            {"algo_name": "b", "rolling_7d_sharpe": 1.5},
        ]
        lifecycle = {"activated": ["new_algo"], "promoted": [], "degraded": ["bad_algo"],
                     "retired": [], "recovered": []}
        msg = format_telegram_summary("2026-06-01", rankings, lifecycle, 50)
        assert "NAT Tournament" in msg
        assert "new_algo" in msg
        assert "bad_algo" in msg

    def test_empty_rankings(self):
        from tournament.report import format_rankings_table
        assert "No rankings" in format_rankings_table([])
