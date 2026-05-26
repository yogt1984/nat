"""Skeptical tests for P1-5: Config Inheritance & Deduplication.

Tests cover:
- _deep_merge: nested merge, override, non-mutation, flat override
- load_agent_config: defaults inheritance, section override, missing file, precedence
- validate_config: unknown top-level keys, unknown gate keys, clean config no warnings
- End-to-end: all 3 agents get correct merged config from real agent.toml
- Backward compat: module-level load_config() still works for all 3 daemons
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from agent.base import _deep_merge, validate_config, load_agent_config


# ===========================================================================
# _deep_merge
# ===========================================================================

class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"gates": {"min_ic": 0.10, "fdr_q": 0.05}}
        override = {"gates": {"min_ic": 0.08}}
        result = _deep_merge(base, override)
        assert result["gates"]["min_ic"] == 0.08
        assert result["gates"]["fdr_q"] == 0.05  # preserved from base

    def test_does_not_mutate_inputs(self):
        base = {"gates": {"min_ic": 0.10}}
        override = {"gates": {"min_ic": 0.08}}
        base_copy = json.loads(json.dumps(base))
        override_copy = json.loads(json.dumps(override))
        _deep_merge(base, override)
        assert base == base_copy
        assert override == override_copy

    def test_override_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_override_replaces_dict_with_scalar(self):
        """If override has a scalar where base has a dict, scalar wins."""
        base = {"gates": {"min_ic": 0.10}}
        override = {"gates": "disabled"}
        assert _deep_merge(base, override) == {"gates": "disabled"}

    def test_override_replaces_scalar_with_dict(self):
        base = {"gates": "disabled"}
        override = {"gates": {"min_ic": 0.10}}
        assert _deep_merge(base, override) == {"gates": {"min_ic": 0.10}}

    def test_empty_override_returns_base(self):
        base = {"a": 1, "gates": {"x": 2}}
        assert _deep_merge(base, {}) == base

    def test_empty_base_returns_override(self):
        override = {"a": 1}
        assert _deep_merge({}, override) == override

    def test_deeply_nested_merge(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2


# ===========================================================================
# validate_config
# ===========================================================================

class TestValidateConfig:
    """Minimal valid config fixture for tests that need required keys present."""

    VALID_BASE = {
        "cycle_interval_s": 3600,
        "max_experiments_per_cycle": 10,
        "generators_enabled": ["systematic"],
    }

    def test_clean_config_no_warnings(self):
        config = {
            **self.VALID_BASE,
            "gates": {"min_ic": 0.10, "fdr_q": 0.05},
        }
        assert validate_config(config, "agent") == []

    def test_unknown_top_level_key(self):
        config = {**self.VALID_BASE, "bogus_key": True}
        warnings = validate_config(config, "agent")
        assert len(warnings) == 1
        assert "bogus_key" in warnings[0]
        assert "[agent]" in warnings[0]

    def test_unknown_gate_key(self):
        config = {**self.VALID_BASE, "gates": {"min_ic": 0.10, "bogus_gate": 0.5}}
        warnings = validate_config(config, "test")
        assert len(warnings) == 1
        assert "bogus_gate" in warnings[0]
        assert "[test.gates]" in warnings[0]

    def test_multiple_unknown_keys(self):
        config = {**self.VALID_BASE, "bogus1": 1, "bogus2": 2, "gates": {"bogus3": 3}}
        warnings = validate_config(config, "x")
        assert len(warnings) == 3

    def test_all_known_keys_pass(self):
        """Every key in _KNOWN_TOP_KEYS and _KNOWN_GATE_KEYS should pass."""
        from agent.base import _KNOWN_TOP_KEYS, _KNOWN_GATE_KEYS
        config = {k: "dummy" for k in _KNOWN_TOP_KEYS}
        config["gates"] = {k: "dummy" for k in _KNOWN_GATE_KEYS}
        assert validate_config(config, "agent") == []

    # --- Required key validation ---

    def test_missing_all_required_keys_raises(self):
        """Empty config should raise ValueError listing all missing keys."""
        with pytest.raises(ValueError, match="missing required keys"):
            validate_config({}, "agent")

    def test_missing_one_required_key_raises(self):
        """Missing just generators_enabled should raise."""
        config = {"cycle_interval_s": 3600, "max_experiments_per_cycle": 10}
        with pytest.raises(ValueError, match="generators_enabled"):
            validate_config(config, "agent")

    def test_error_message_includes_section_name(self):
        with pytest.raises(ValueError, match=r"\[my_section\]"):
            validate_config({"cycle_interval_s": 1}, "my_section")

    def test_error_lists_all_missing_keys_sorted(self):
        """All missing keys should appear in the error message."""
        with pytest.raises(ValueError) as exc_info:
            validate_config({}, "agent")
        msg = str(exc_info.value)
        assert "cycle_interval_s" in msg
        assert "generators_enabled" in msg
        assert "max_experiments_per_cycle" in msg

    def test_required_keys_present_no_raise(self):
        """Config with all required keys should not raise."""
        validate_config(self.VALID_BASE, "agent")  # no exception


# ===========================================================================
# load_agent_config
# ===========================================================================

class TestLoadAgentConfig:
    def test_missing_file_returns_base(self):
        base = {"cycle_interval_s": 999, "gates": {"min_ic": 0.42}}
        result = load_agent_config(Path("/nonexistent/agent.toml"), "agent", base)
        assert result == base

    def test_defaults_section_inherited(self, tmp_path):
        """[defaults] keys should appear in every section's config."""
        toml_content = b"""
[defaults]
max_cycle_runtime_s = 9999
max_experiments_per_cycle = 5
generators_enabled = ["test"]

[defaults.gates]
fdr_q = 0.01

[agent]
cycle_interval_s = 3600

[agent_mf]
cycle_interval_s = 7200
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)
        base = {"cycle_interval_s": 0}

        # Agent inherits defaults
        cfg_agent = load_agent_config(toml_file, "agent", base)
        assert cfg_agent["max_cycle_runtime_s"] == 9999
        assert cfg_agent["gates"]["fdr_q"] == 0.01
        assert cfg_agent["cycle_interval_s"] == 3600

        # MF also inherits same defaults
        cfg_mf = load_agent_config(toml_file, "agent_mf", base)
        assert cfg_mf["max_cycle_runtime_s"] == 9999
        assert cfg_mf["gates"]["fdr_q"] == 0.01
        assert cfg_mf["cycle_interval_s"] == 7200

    def test_section_overrides_defaults(self, tmp_path):
        """Section-specific keys beat [defaults]."""
        toml_content = b"""
[defaults]
cycle_interval_s = 3600
max_experiments_per_cycle = 5
generators_enabled = ["test"]

[defaults.gates]
min_ic = 0.10
fdr_q = 0.05

[agent_mf.gates]
min_ic = 0.08
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)

        cfg = load_agent_config(toml_file, "agent_mf", {})
        assert cfg["gates"]["min_ic"] == 0.08   # overridden
        assert cfg["gates"]["fdr_q"] == 0.05    # inherited from defaults

    def test_base_config_is_lowest_precedence(self, tmp_path):
        """BASE_CONFIG < [defaults] < [section]."""
        toml_content = b"""
[defaults]
max_experiments_per_cycle = 50
generators_enabled = ["test"]

[agent]
max_experiments_per_cycle = 10
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)

        base = {"max_experiments_per_cycle": 999, "cycle_interval_s": 42}
        cfg = load_agent_config(toml_file, "agent", base)
        assert cfg["max_experiments_per_cycle"] == 10    # section wins
        assert cfg["cycle_interval_s"] == 42             # base preserved

    def test_empty_toml_returns_base_plus_symbols(self, tmp_path):
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(b"")
        base = {
            "cycle_interval_s": 42,
            "max_experiments_per_cycle": 5,
            "generators_enabled": ["test"],
        }
        cfg = load_agent_config(toml_file, "agent", base)
        assert cfg["cycle_interval_s"] == 42
        assert cfg["max_experiments_per_cycle"] == 5
        # symbols.primary injected from config/symbols.toml
        assert cfg["symbols"]["primary"] == ["BTC", "ETH", "SOL"]

    def test_missing_section_still_gets_defaults(self, tmp_path):
        """If [agent_mf] doesn't exist, we still get [defaults]."""
        toml_content = b"""
[defaults]
max_cycle_runtime_s = 5400
cycle_interval_s = 3600
max_experiments_per_cycle = 10
generators_enabled = ["test"]

[defaults.gates]
fdr_q = 0.05
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)

        cfg = load_agent_config(toml_file, "agent_mf", {})
        assert cfg["max_cycle_runtime_s"] == 5400
        assert cfg["gates"]["fdr_q"] == 0.05

    def test_nested_defaults_preserved_under_section_override(self, tmp_path):
        """When section overrides one nested key, sibling keys from defaults survive."""
        toml_content = b"""
[defaults]
cycle_interval_s = 3600
max_experiments_per_cycle = 5
generators_enabled = ["test"]

[defaults.decay]
ic_decay_ratio = 0.5
consecutive_days_limit = 14

[agent_macro.decay]
consecutive_days_limit = 7
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)

        cfg = load_agent_config(toml_file, "agent_macro", {})
        assert cfg["decay"]["consecutive_days_limit"] == 7   # overridden
        assert cfg["decay"]["ic_decay_ratio"] == 0.5         # inherited

    def test_load_raises_on_missing_required_keys(self, tmp_path):
        """load_agent_config should raise if merged config lacks required keys."""
        toml_content = b"""
[agent]
cycle_interval_s = 3600
"""
        toml_file = tmp_path / "agent.toml"
        toml_file.write_bytes(toml_content)

        with pytest.raises(ValueError, match="missing required keys"):
            load_agent_config(toml_file, "agent", {})


# ===========================================================================
# End-to-end: real agent.toml produces correct merged configs
# ===========================================================================

class TestRealTomlInheritance:
    """Verify that the actual config/agent.toml produces correct merged values."""

    @pytest.fixture
    def toml_path(self):
        return Path(__file__).resolve().parent.parent.parent / "config" / "agent.toml"

    def test_micro_gets_defaults(self, toml_path):
        from agent.daemon import MicrostructureAgent
        cfg = load_agent_config(toml_path, "agent", MicrostructureAgent.BASE_CONFIG)
        # From [defaults]
        assert cfg["gates"]["fdr_q"] == 0.05
        assert cfg["gates"]["min_oos_dates"] == 2
        assert cfg["decay"]["ic_decay_ratio"] == 0.5
        assert cfg["symbols"]["primary"] == ["BTC", "ETH", "SOL"]
        # From [agent] specifically
        assert cfg["cycle_interval_s"] == 3600
        assert cfg["gates"]["min_ic"] == 0.10

    def test_mf_inherits_defaults_overrides_ic(self, toml_path):
        from agent.mf_daemon import MediumFrequencyAgent
        cfg = load_agent_config(toml_path, "agent_mf", MediumFrequencyAgent.BASE_CONFIG)
        # Inherited from [defaults]
        assert cfg["gates"]["fdr_q"] == 0.05
        assert cfg["gates"]["min_oos_dates"] == 2
        assert cfg["decay"]["ic_decay_ratio"] == 0.5
        assert cfg["symbols"]["primary"] == ["BTC", "ETH", "SOL"]
        # Overridden in [agent_mf]
        assert cfg["gates"]["min_ic"] == 0.08
        assert cfg["cycle_interval_s"] == 7200

    def test_macro_inherits_defaults_overrides_ic(self, toml_path):
        from agent.macro_daemon import MacroAgent
        cfg = load_agent_config(toml_path, "agent_macro", MacroAgent.BASE_CONFIG)
        # Inherited from [defaults]
        assert cfg["gates"]["fdr_q"] == 0.05
        assert cfg["gates"]["min_oos_dates"] == 2
        assert cfg["decay"]["ic_decay_ratio"] == 0.5
        # Overridden in [agent_macro]
        assert cfg["gates"]["min_ic"] == 0.07
        assert cfg["cycle_interval_s"] == 14400
        assert cfg["max_cycle_runtime_s"] == 7200  # overrides default 5400

    def test_each_agent_has_own_state_dir(self, toml_path):
        from agent.daemon import MicrostructureAgent
        from agent.mf_daemon import MediumFrequencyAgent
        from agent.macro_daemon import MacroAgent

        cfg_micro = load_agent_config(toml_path, "agent", MicrostructureAgent.BASE_CONFIG)
        cfg_mf = load_agent_config(toml_path, "agent_mf", MediumFrequencyAgent.BASE_CONFIG)
        cfg_macro = load_agent_config(toml_path, "agent_macro", MacroAgent.BASE_CONFIG)

        dirs = {
            cfg_micro["paths"]["state_dir"],
            cfg_mf["paths"]["state_dir"],
            cfg_macro["paths"]["state_dir"],
        }
        assert len(dirs) == 3  # all different

    def test_no_duplicate_keys_in_toml(self, toml_path):
        """Verify that fdr_q, min_oos_dates, etc. only appear in [defaults],
        not duplicated in [agent_mf] or [agent_macro] sections."""
        lines = [l for l in toml_path.read_text().splitlines()
                 if not l.strip().startswith("#")]
        content = "\n".join(lines)
        # These should only appear once (in [defaults]), excluding comments
        for key in ["fdr_q", "ic_decay_ratio", "consecutive_days_limit"]:
            count = content.count(key)
            assert count == 1, f"{key} appears {count} times, expected 1 (only in [defaults])"


# ===========================================================================
# Backward compat: module-level load_config() functions
# ===========================================================================

class TestModuleLevelLoadConfig:
    def test_daemon_load_config(self):
        from agent.daemon import load_config
        cfg = load_config()
        assert cfg["cycle_interval_s"] == 3600
        assert "systematic" in cfg["generators_enabled"]
        assert cfg["gates"]["fdr_q"] == 0.05

    def test_mf_daemon_load_config(self):
        from agent.mf_daemon import load_config
        cfg = load_config()
        assert cfg["cycle_interval_s"] == 7200
        assert "momentum" in cfg["generators_enabled"]
        # Inherited from [defaults]
        assert cfg["gates"]["fdr_q"] == 0.05

    def test_macro_daemon_load_config(self):
        from agent.macro_daemon import load_config
        cfg = load_config()
        assert cfg["cycle_interval_s"] == 14400
        assert "funding_meanrev" in cfg["generators_enabled"]
        # Inherited from [defaults]
        assert cfg["gates"]["fdr_q"] == 0.05

    def test_mf_config_has_defaults_symbols(self):
        """MF config should inherit symbols from [defaults] not require its own."""
        from agent.mf_daemon import load_config
        cfg = load_config()
        assert cfg["symbols"]["primary"] == ["BTC", "ETH", "SOL"]

    def test_macro_config_has_defaults_decay(self):
        """Macro config should inherit decay from [defaults]."""
        from agent.macro_daemon import load_config
        cfg = load_config()
        assert cfg["decay"]["ic_decay_ratio"] == 0.5
        assert cfg["decay"]["consecutive_days_limit"] == 14
