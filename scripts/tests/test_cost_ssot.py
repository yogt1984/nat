"""COST-1/2: the cost model sources fees from the SSOT (config/costs.toml via
utils.costs), and preset resolution never silently falls back to zero cost."""

import pytest

from backtest.costs import CostModel, cost_model_from_name
from utils.costs import maker_bps, taker_bps


def test_costmodel_default_fee_is_config_taker():
    # COST-1: a bare CostModel must use the config taker fee, not a hardcoded 5.0.
    assert CostModel().fee_bps == taker_bps()


def test_from_config_taker_matches_ssot():
    assert CostModel.from_config(role="taker").fee_bps == taker_bps()


def test_from_config_maker_matches_ssot():
    assert CostModel.from_config(role="maker").fee_bps == maker_bps()


def test_cost_model_from_name_known_presets():
    assert cost_model_from_name("taker").fee_bps == taker_bps()
    assert cost_model_from_name("zero").fee_bps == 0.0  # zero requires explicit opt-in


@pytest.mark.parametrize("bad", ["bogus", "", "TAKER", "0"])
def test_cost_model_from_name_unknown_raises(bad):
    # COST-2: a mistyped/empty preset must fail loudly, never a silent cost-free run.
    with pytest.raises(ValueError):
        cost_model_from_name(bad)
