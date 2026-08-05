"""PROC-1 — process→algorithm compiler. Written adversarially: the refusals are the unit.

The compiler's job is NOT to emit code; it is to emit code *only* for findings that have
earned it. A generator that compiles whatever it is handed is a false-discovery factory
with a `@register` decorator on top — the Q4 failure mode (FINDINGS §4.6) automated. So
most of this file tries to make the compiler emit something it should not, and asserts it
refuses and writes nothing.

Contract encoded here:
  (a) admission — only null-calibrated (z ≥ threshold), FDR-passed (q ≤ alpha),
      `informative` findings compile, and the thresholds are IMPORTED from config, never
      written into the compiler;
  (b) polarity — MI is non-negative and carries no direction, so a finding without an
      explicit polarity cannot become a trading rule. Refuse rather than guess;
  (c) safety — a feature name is rendered into source code, so anything that is not a
      plain column identifier is rejected before it reaches a template;
  (d) no shadowing — a generated algorithm may never take the name of a registered one;
  (e) conformance — what IS emitted satisfies the algorithm contract (alg_ prefix, keys
      == alg_features(), NaN-in→NaN-out, warmup, reset);
  (f) determinism — the same finding renders byte-identically, so the diff is reviewable
      and the provenance is reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _row(**over) -> dict:
    """A surface row that SHOULD compile — each test breaks exactly one thing."""
    row = {
        "combo_id": "imbalance_qty_l1",
        "horizon": "50",
        "label_def": "tb(2.0,2.0)",
        "regime_var": "",
        "regime_bin": "all",
        "mi_bits": 0.041,
        "raw_bits": 0.052,
        "z_null": 6.2,
        "p_value": 0.0004,
        "bh_q": 0.008,
        "informative": True,
        "maturity": "PRELIM",
        "n": 21708,
        "symbol": "BTC",
        "timeframe": "15min",
        "process": "horizon_label_scan",
        "run_id": "run-abc123",
        "git_sha": "deadbeef",
        "generated_at": "2026-08-05T00:00:00+00:00",
        "polarity": 1,
    }
    row.update(over)
    return row


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "generated"
    d.mkdir()
    return d


# ── (a) admission: thresholds imported, not invented ─────────────────────────────
class TestAdmissionGate:
    def test_a_clean_finding_is_admitted(self):
        from agent.algo_synth import PromotedFinding
        ok, reasons = PromotedFinding.from_surface_row(_row()).is_compilable()
        assert ok, reasons

    @pytest.mark.parametrize("field,value,fragment", [
        ("informative", False, "informative"),
        ("z_null", 1.4, "null"),          # below it_engine.toml null_z_threshold (3.0)
        ("bh_q", 0.4, "fdr"),             # above agent.toml fdr_q / DEFAULT_FDR_ALPHA (0.05)
        ("z_null", None, "null"),
        ("bh_q", None, "fdr"),
    ])
    def test_unearned_findings_are_refused(self, field, value, fragment):
        from agent.algo_synth import PromotedFinding
        ok, reasons = PromotedFinding.from_surface_row(_row(**{field: value})).is_compilable()
        assert not ok
        assert any(fragment in r.lower() for r in reasons), reasons

    def test_thresholds_come_from_config_not_source(self):
        """The gate must read the SSOT; a literal in the compiler is a guardrail breach."""
        from it_engine.null_calibration import load_null_config
        from processes.fdr import DEFAULT_FDR_ALPHA
        from agent import algo_synth
        assert algo_synth.null_z_threshold() == load_null_config()["null_z_threshold"]
        assert algo_synth.fdr_alpha() == DEFAULT_FDR_ALPHA
        src = Path(algo_synth.__file__).read_text()
        for literal in ("3.0", "0.05"):
            assert f"= {literal}" not in src, (
                f"threshold literal {literal} hardcoded in algo_synth — import it")

    def test_gate_tracks_config_rather_than_a_snapshot(self, monkeypatch):
        """Raise the bar in config and a formerly-admissible finding must stop compiling."""
        from agent import algo_synth
        from agent.algo_synth import PromotedFinding
        monkeypatch.setattr(algo_synth, "null_z_threshold", lambda: 99.0)
        ok, reasons = PromotedFinding.from_surface_row(_row()).is_compilable()
        assert not ok and any("null" in r.lower() for r in reasons)


# ── (b) polarity: MI has no sign ─────────────────────────────────────────────────
class TestPolarityMustBeExplicit:
    """MI ≥ 0 and is direction-blind. A rule needs a sign; guessing one invents an edge."""

    @pytest.mark.parametrize("polarity", [None, 0])
    def test_missing_polarity_is_refused(self, polarity):
        from agent.algo_synth import PromotedFinding
        ok, reasons = PromotedFinding.from_surface_row(_row(polarity=polarity)).is_compilable()
        assert not ok
        assert any("polarity" in r.lower() for r in reasons), reasons

    def test_polarity_flips_the_emitted_rule(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        long_ = synthesize(PromotedFinding.from_surface_row(_row(polarity=1)), out_dir=out_dir)
        short = synthesize(PromotedFinding.from_surface_row(_row(polarity=-1)), out_dir=out_dir)
        assert long_.read_text() != short.read_text()


# ── (c) safety: feature names become source code ─────────────────────────────────
class TestUntrustedNamesNeverReachTheTemplate:
    @pytest.mark.parametrize("name", [
        "x'); import os; os.system('rm -rf /')#",
        "imbalance\nqty",
        "imbalance qty l1",
        "3_leading_digit",
        "",
        "a" * 200,
        "__class__",
    ])
    def test_non_identifier_columns_are_refused(self, name, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        finding = PromotedFinding.from_surface_row(_row(combo_id=name))
        ok, reasons = finding.is_compilable()
        assert not ok and any("name" in r.lower() or "identifier" in r.lower() for r in reasons)
        with pytest.raises(ValueError):
            synthesize(finding, out_dir=out_dir)
        assert list(out_dir.iterdir()) == [], "refused finding still wrote a file"

    def test_regime_variable_is_validated_too(self, out_dir):
        from agent.algo_synth import PromotedFinding
        bad = _row(regime_var="ent'); os.system('x", regime_bin="b2")
        ok, reasons = PromotedFinding.from_surface_row(bad).is_compilable()
        assert not ok


# ── (d) generated code may never shadow a hand-written algorithm ─────────────────
class TestNoShadowing:
    def test_refuses_to_take_a_registered_name(self, out_dir, monkeypatch):
        from agent import algo_synth
        from agent.algo_synth import PromotedFinding, synthesize
        finding = PromotedFinding.from_surface_row(_row())
        monkeypatch.setattr(algo_synth, "existing_algorithm_names",
                            lambda: {finding.algorithm_name})
        with pytest.raises(ValueError, match="already registered"):
            synthesize(finding, out_dir=out_dir)
        assert list(out_dir.iterdir()) == []

    def test_refuses_to_silently_overwrite_a_generated_file(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        finding = PromotedFinding.from_surface_row(_row())
        path = synthesize(finding, out_dir=out_dir)
        path.write_text(path.read_text() + "\n# hand edit\n")
        with pytest.raises(FileExistsError):
            synthesize(finding, out_dir=out_dir)
        assert "# hand edit" in path.read_text()
        synthesize(finding, out_dir=out_dir, overwrite=True)     # explicit only
        assert "# hand edit" not in path.read_text()


# ── (e) what IS emitted must satisfy the algorithm contract ──────────────────────
def _load(path: Path):
    """Import a generated module and return a fresh instance of its algorithm."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"gen_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ALGORITHM_CLASS()


class TestGeneratedAlgorithmConformance:
    def test_contract_shape(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        from algorithms.base import MicrostructureAlgorithm
        assert isinstance(algo, MicrostructureAlgorithm)
        names = [f.name for f in algo.alg_features()]
        assert names and all(n.startswith("alg_") for n in names)
        assert algo.required_columns() == ["imbalance_qty_l1"]
        out = algo.step({"imbalance_qty_l1": 0.4})
        assert set(out) == set(names), "step() keys must equal alg_features() exactly"

    def test_nan_in_nan_out(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        for _ in range(200):                                   # past warmup
            algo.step({"imbalance_qty_l1": float(np.random.randn())})
        out = algo.step({"imbalance_qty_l1": float("nan")})
        assert all(np.isnan(v) for v in out.values())

    def test_missing_required_column_does_not_crash(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        out = algo.step({})
        assert all(np.isnan(v) for v in out.values())

    def test_reset_clears_state(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        for x in np.linspace(-1, 1, 300):
            algo.step({"imbalance_qty_l1": float(x)})
        primed = algo.step({"imbalance_qty_l1": 0.5})
        algo.reset()
        fresh = algo.step({"imbalance_qty_l1": 0.5})
        assert not (np.isfinite(list(primed.values())[0]) and
                    np.isfinite(list(fresh.values())[0])), "reset() left state behind"

    def test_provenance_is_in_the_docstring(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        text = synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir).read_text()
        for token in ("0.041", "6.2", "0.008", "deadbeef", "run-abc123",
                      "horizon_label_scan", "PROC-1"):
            assert token in text, f"provenance token {token!r} missing from generated source"

    def test_warmup_is_declared(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        assert algo.warmup > 0, "a rolling-z rule with zero warmup is a look-ahead claim"


# ── planted round-trip: the generated rule must reproduce a known edge ───────────
class TestPlantedRoundTrip:
    """Spec §1: feed a finding whose rule is known, assert the emitted algorithm has IC."""

    @staticmethod
    def _planted_frame(n=4000, polarity=1, seed=7):
        rng = np.random.default_rng(seed)
        f = rng.normal(size=n)
        noise = rng.normal(scale=1.0, size=n)
        fwd = polarity * 0.6 * f + noise            # known linear rule, sign = polarity
        return pd.DataFrame({"imbalance_qty_l1": f, "fwd_ret": fwd})

    @pytest.mark.parametrize("polarity", [1, -1])
    def test_generated_algorithm_recovers_the_planted_sign(self, polarity, out_dir):
        from scipy.stats import spearmanr
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row(polarity=polarity)),
                                out_dir=out_dir))
        df = self._planted_frame(polarity=polarity)
        sig = algo.run_batch(df[["imbalance_qty_l1"]])
        col = [c for c in sig.columns if c.startswith("alg_")][0]
        m = np.isfinite(sig[col].to_numpy()) & np.isfinite(df["fwd_ret"].to_numpy())
        assert m.sum() > 1000
        ic = spearmanr(sig[col].to_numpy()[m], df["fwd_ret"].to_numpy()[m]).statistic
        assert ic > 0.25, f"generated rule failed to recover the planted edge (IC={ic:.3f})"

    def test_pure_noise_yields_no_edge(self, out_dir):
        """The mirror test: the same generated rule on noise must NOT print an edge."""
        from scipy.stats import spearmanr
        from agent.algo_synth import PromotedFinding, synthesize
        algo = _load(synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir))
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"imbalance_qty_l1": rng.normal(size=4000),
                           "fwd_ret": rng.normal(size=4000)})
        sig = algo.run_batch(df[["imbalance_qty_l1"]])
        col = [c for c in sig.columns if c.startswith("alg_")][0]
        m = np.isfinite(sig[col].to_numpy())
        ic = spearmanr(sig[col].to_numpy()[m], df["fwd_ret"].to_numpy()[m]).statistic
        assert abs(ic) < 0.08, f"noise produced IC={ic:.3f} — the template leaks structure"


# ── (f) determinism ──────────────────────────────────────────────────────────────
class TestDeterminism:
    def test_same_finding_renders_byte_identically(self, out_dir, tmp_path):
        from agent.algo_synth import PromotedFinding, render_source
        a = render_source(PromotedFinding.from_surface_row(_row()))
        b = render_source(PromotedFinding.from_surface_row(_row()))
        assert a == b

    def test_no_wall_clock_in_generated_source(self, out_dir):
        """Provenance is the FINDING's timestamp; now() would break reproducibility."""
        from agent.algo_synth import PromotedFinding, synthesize
        text = synthesize(PromotedFinding.from_surface_row(_row()), out_dir=out_dir).read_text()
        assert "2026-08-05T00:00:00+00:00" in text
        assert "datetime.now" not in text and "utcnow" not in text


# ── regime-gated kind ────────────────────────────────────────────────────────────
class TestRegimeGatedKind:
    def test_regime_bin_produces_a_gated_rule(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        gated = _row(regime_var="ent_book_shape_std", regime_bin="b0")
        finding = PromotedFinding.from_surface_row(gated)
        assert finding.kind == "regime_gated"
        algo = _load(synthesize(finding, out_dir=out_dir))
        assert set(algo.required_columns()) == {"imbalance_qty_l1", "ent_book_shape_std"}

    def test_unsupported_kind_refuses_rather_than_guesses(self, out_dir):
        from agent.algo_synth import PromotedFinding, synthesize
        combo = _row(combo_id="pca_combo_3")           # a combiner finding, not a column rule
        finding = PromotedFinding.from_surface_row(combo, kind="combiner")
        with pytest.raises(NotImplementedError):
            synthesize(finding, out_dir=out_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
