"""QA-JD2 wiring contract — paper_trader_generic.ALGO_CONFIG must be consistent
with the algorithm registry, and jump_detector_v2 must be wired in.

The Q4 kill gate found jump_detector_v2 (EVT threshold, exact step/batch parity)
was never wired into the economics harness — so the one open question ("was the
Lee-Mykland family dead, or just v1's miscalibrated threshold?") was untestable.
This pins the wiring and a generic contract so future algorithms can't silently
point at nonexistent signal columns.
"""

from __future__ import annotations

import pytest

from algorithms import discover_all
from algorithms.registry import get_algorithm, list_algorithms
from alpha.paper_trader_generic import ALGO_CONFIG


@pytest.fixture(scope="module", autouse=True)
def _discover():
    discover_all()


class TestJd2Wired:
    def test_jump_detector_v2_in_config(self):
        assert "jump_detector_v2" in ALGO_CONFIG, (
            "QA-JD2: v2 must be runnable through the economics harness"
        )

    def test_jd2_primary_and_polarity(self):
        cfg = ALGO_CONFIG["jump_detector_v2"]
        assert cfg["primary"] == "alg_jd2_reversion"
        assert cfg["polarity"] == "low_long"          # same reversion semantics as v1
        assert cfg["bar_agg"] == "mean"


class TestConfigRegistryContract:
    def test_every_registry_backed_entry_points_at_real_feature(self):
        registered = set(list_algorithms())
        bad = []
        for name, cfg in ALGO_CONFIG.items():
            if name not in registered:
                continue                              # non-registry entries (e.g. composites)
            feats = {f.name for f in get_algorithm(name).alg_features()}
            if cfg["primary"] not in feats:
                bad.append(f"{name}: primary '{cfg['primary']}' not in {sorted(feats)}")
        assert not bad, "ALGO_CONFIG points at nonexistent signal columns:\n  " + "\n  ".join(bad)

    def test_polarity_and_agg_vocabulary(self):
        for name, cfg in ALGO_CONFIG.items():
            assert cfg["polarity"] in ("high_long", "low_long"), name
            assert cfg["bar_agg"] in ("mean", "last", "max", "sum"), name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
