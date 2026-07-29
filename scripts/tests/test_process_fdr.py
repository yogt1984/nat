"""Planted test for PROC-13 — FDR/DSR on the process layer + cross-run ledger.

A surface of thousands of (combo × horizon × label × regime) cells *guarantees* a
great-looking argmax by chance. PROC-13 applies Benjamini–Hochberg across all cell
p-values (produced by PROC-12) and reports every cell WITH its q-value; the argmax
is only ever surfaced as "argmax, BH-q = …".

Contract:
  (a) an all-null grid -> ≈0 discoveries (empirical FDR ≤ q);
  (b) a grid with k planted true cells -> all k recovered, false-discovery proportion ≤ q;
  (c) every cell is annotated with a BH q-value (Finding.p_adjusted set);
  (d) informative composes: a cell stays informative only if it ALSO passes FDR;
  (e) the argmax cell is reported together with its q-value (never bare);
  (f) apply_process_fdr accepts a ProcessResult and writes q back onto its findings;
  (g) the cross-run ledger records (process, target, n_tested, git_sha) per sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from processes.base import Finding, ProcessResult
from processes.fdr import FdrReport, apply_process_fdr, read_ledger, record_sweep


def _grid(n_null: int, n_true: int, seed: int = 0):
    """A sweep's worth of cells: n_null uniform-p nulls + n_true tiny-p planted edges."""
    rng = np.random.default_rng(seed)
    findings = []
    for i in range(n_null):
        findings.append(Finding(
            feature=f"null_{i}", horizon="h1", metric="cond_mi_bits",
            value=float(rng.uniform(0.0, 0.02)),          # floor-level effect
            p_value=float(rng.uniform(0.0, 1.0)),         # honest null p
            informative=True,                             # per-cell gate "passed"
        ))
    for j in range(n_true):
        findings.append(Finding(
            feature=f"true_{j}", horizon="h1", metric="cond_mi_bits",
            value=0.30 + 0.01 * j,                        # a real regime
            p_value=1e-6,                                 # decisively significant
            informative=True,
        ))
    return findings


class TestAllNull:
    def test_all_null_grid_controls_false_discoveries(self):
        findings = _grid(500, 0, seed=1)
        rep = apply_process_fdr(findings, alpha=0.05)
        assert isinstance(rep, FdrReport)
        assert rep.n_cells == 500
        # BH controls the expected false-discovery proportion at q; a single uniform-null
        # grid should surface (essentially) nothing.
        assert rep.n_discoveries <= 0.05 * 500
        assert rep.n_discoveries == 0
        # every cell carries a q-value...
        assert all(f.p_adjusted is not None for f in findings)
        # ...and FDR strips the (false) per-cell informative flags.
        assert all(not f.informative for f in findings)


class TestPlantedPower:
    def test_recovers_true_cells_and_controls_fdp(self):
        findings = _grid(500, 10, seed=2)
        rep = apply_process_fdr(findings, alpha=0.05)
        true = [f for f in findings if f.feature.startswith("true_")]
        assert all(f.informative for f in true), "all planted regimes must be recovered"
        assert all(f.p_adjusted <= 0.05 for f in true)
        # false-discovery proportion among all discoveries is bounded by q.
        false_disc = [f for f in findings if f.informative and f.feature.startswith("null_")]
        fdp = len(false_disc) / max(rep.n_discoveries, 1)
        assert fdp <= 0.05 + 1e-9
        assert rep.n_discoveries >= 10


class TestArgmaxCarriesQ:
    def test_argmax_reported_with_its_q(self):
        findings = _grid(200, 5, seed=3)
        rep = apply_process_fdr(findings, alpha=0.05)
        assert rep.argmax is not None
        assert rep.argmax["feature"].startswith("true_")   # strongest cell is a real one
        assert rep.argmax["q_value"] is not None            # never bare
        assert rep.argmax["q_value"] <= 0.05

    def test_argmax_of_pure_null_still_carries_q(self):
        # Even when nothing survives, the headline cell is reported WITH its correction.
        findings = _grid(300, 0, seed=7)
        rep = apply_process_fdr(findings, alpha=0.05)
        assert rep.argmax is not None
        assert rep.argmax["q_value"] is not None
        assert rep.n_discoveries == 0


class TestProcessResultInput:
    def test_accepts_process_result_and_writes_back(self):
        findings = _grid(100, 5, seed=4)
        pr = ProcessResult(
            run_id="r", process="conditional_predictability", kind="evaluation",
            symbol="BTC", timeframe="bar", params={}, findings=findings,
        )
        rep = apply_process_fdr(pr, alpha=0.05)
        assert rep.n_cells == 105
        assert rep.n_discoveries >= 5
        assert all(f.p_adjusted is not None for f in pr.findings)

    def test_ignores_cells_without_p_value(self):
        findings = _grid(50, 3, seed=5)
        findings.append(Finding(feature="no_p", horizon="h1", metric="ic_mean",
                                value=0.9, p_value=None, informative=True))
        rep = apply_process_fdr(findings, alpha=0.05)
        # the p-less cell is not part of the correction and keeps its own flag untouched
        no_p = [f for f in findings if f.feature == "no_p"][0]
        assert no_p.p_adjusted is None
        assert no_p.informative is True
        assert rep.n_cells == 54          # 50 null + 3 true + 1 p-less
        assert rep.n_pvalued == 53        # the p-less cell never enters BH


class TestLedger:
    def test_records_and_reads_sweeps(self, tmp_path):
        led = tmp_path / "fdr_ledger.jsonl"
        record_sweep(led, process="conditional_predictability", target="fwd_ret_h1",
                     n_tested=505, git_sha="abc123", alpha=0.05, n_discoveries=10)
        record_sweep(led, process="mi_ksg", target="tb_label",
                     n_tested=88, git_sha="def456", alpha=0.05, n_discoveries=3)
        rows = read_ledger(led)
        assert len(rows) == 2
        assert rows[0]["process"] == "conditional_predictability"
        assert rows[0]["target"] == "fwd_ret_h1"
        assert rows[0]["n_tested"] == 505
        assert rows[0]["git_sha"] == "abc123"
        assert rows[0]["n_discoveries"] == 10
        assert rows[1]["process"] == "mi_ksg"

    def test_read_missing_ledger_is_empty(self, tmp_path):
        assert read_ledger(tmp_path / "nope.jsonl") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
