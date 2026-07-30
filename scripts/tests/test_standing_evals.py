"""PROC-5 Part C — the standing-evaluation registry + "ever run?" audit.

The 3-bar triple-barrier classifier must exist as a *standing* evaluation
(triple_barrier -> mi_ksg targeting tb_label), not an ad-hoc one-off, and we must be able
to answer "has it ever actually run?" against the persisted process-results index.

These tests pin the registry contract, the run-matching logic (a label-mode mi_ksg record
targeting tb_label counts; a forward-return mi_ksg run does NOT), the audit over the
SQLite index, and the run dispatch.
"""

from __future__ import annotations

import pytest

from processes.base import Finding, ProcessResult
from processes.standing import (
    STANDING_EVALS,
    StandingEval,
    _record_matches,
    audit_standing_evals,
    get_standing_eval,
    list_standing_evals,
    run_standing_eval,
)


# --------------------------------------------------------------------------- #
# registry contract                                                           #
# --------------------------------------------------------------------------- #

class TestRegistry:
    def test_three_bar_classifier_registered(self):
        ev = get_standing_eval("barrier_3bar_mi")
        assert isinstance(ev, StandingEval)
        assert ev.transform == "triple_barrier"
        assert ev.scorer == "mi_ksg"
        assert ev.target == "tb_label"
        assert ev.symbols                       # at least one symbol scheduled

    def test_list_includes_the_classifier(self):
        names = {e.name for e in list_standing_evals()}
        assert "barrier_3bar_mi" in names
        assert len(STANDING_EVALS) == len(names)

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_standing_eval("does_not_exist")


# --------------------------------------------------------------------------- #
# run-matching logic (what counts as "this eval was run")                      #
# --------------------------------------------------------------------------- #

def _label_record(process="mi_ksg", target="tb_label"):
    return {
        "process": process,
        "findings": [
            {"feature": "f_good", "horizon": "label", "metric": "mi_bits",
             "value": 0.2, "extras": {"target": target, "gate": "null_z"}},
        ],
    }


def _fwd_return_record():
    return {
        "process": "mi_ksg",
        "findings": [
            {"feature": "tb_label", "horizon": "h1", "metric": "mi_bits",
             "value": 0.01, "extras": {"i_min_bits": 0.03}},
        ],
    }


class TestRecordMatching:
    def test_label_mode_run_matches(self):
        ev = get_standing_eval("barrier_3bar_mi")
        assert _record_matches(_label_record(), ev) is True

    def test_forward_return_run_does_not_match(self):
        ev = get_standing_eval("barrier_3bar_mi")
        # tb_label scored as a FEATURE vs forward returns is NOT the standing eval.
        assert _record_matches(_fwd_return_record(), ev) is False

    def test_wrong_process_does_not_match(self):
        ev = get_standing_eval("barrier_3bar_mi")
        assert _record_matches(_label_record(process="ic_horizon"), ev) is False

    def test_wrong_target_does_not_match(self):
        ev = get_standing_eval("barrier_3bar_mi")
        assert _record_matches(_label_record(target="tb_ret"), ev) is False


# --------------------------------------------------------------------------- #
# audit over the persisted index                                              #
# --------------------------------------------------------------------------- #

class TestAudit:
    def test_empty_db_reports_never_run(self, tmp_path):
        rows = audit_standing_evals(db_path=tmp_path / "nat.db")
        b3 = [r for r in rows if r["name"] == "barrier_3bar_mi"][0]
        assert b3["ever_run"] is False
        assert b3["n_runs"] == 0
        assert b3["last_run"] is None

    def _save(self, tmp_path, run_id, findings):
        from processes import persistence
        r = ProcessResult(
            run_id=run_id, process="mi_ksg", kind="evaluation",
            symbol="BTC", timeframe="15min", params={},
        )
        r.findings = findings
        r.provenance = {"git_sha": "deadbeef", "generated_at": "2026-07-30T00:00:00+00:00"}
        r.finalize(1.0)
        persistence.save_result(r, out_dir=tmp_path / "json", db_path=tmp_path / "nat.db")

    def test_detects_a_real_saved_label_run(self, tmp_path):
        self._save(tmp_path, "mi_ksg_BTC_label", [
            Finding(feature="f_good", horizon="label", metric="mi_bits", value=0.2,
                    informative=True, extras={"target": "tb_label", "gate": "null_z"}),
        ])
        rows = audit_standing_evals(db_path=tmp_path / "nat.db")
        b3 = [r for r in rows if r["name"] == "barrier_3bar_mi"][0]
        assert b3["ever_run"] is True
        assert b3["n_runs"] == 1
        assert b3["last_run"] == "2026-07-30T00:00:00+00:00"
        assert "BTC" in b3["symbols_seen"]

    def test_forward_return_run_is_not_counted(self, tmp_path):
        # A plain mi_ksg run (forward returns) must NOT register as the standing eval.
        self._save(tmp_path, "mi_ksg_BTC_fwd", [
            Finding(feature="imbalance", horizon="h1", metric="mi_bits", value=0.03,
                    extras={"i_min_bits": 0.02}),
        ])
        rows = audit_standing_evals(db_path=tmp_path / "nat.db")
        b3 = [r for r in rows if r["name"] == "barrier_3bar_mi"][0]
        assert b3["ever_run"] is False


# --------------------------------------------------------------------------- #
# run dispatch                                                                 #
# --------------------------------------------------------------------------- #

class TestRunDispatch:
    def test_run_dispatches_to_runner_with_target(self, monkeypatch):
        calls = {}

        def fake_run_process(name, **kw):
            calls["name"] = name
            calls.update(kw)
            return "ok"

        monkeypatch.setattr("processes.runner.run_process", fake_run_process)
        out = run_standing_eval("barrier_3bar_mi", symbol="ETH", save=False)
        assert out == "ok"
        assert calls["name"] == "triple_barrier"       # transform runs first
        assert calls["score_with"] == "mi_ksg"
        assert calls["score_target"] == "tb_label"
        assert calls["symbol"] == "ETH"
        assert calls["save"] is False

    def test_run_unknown_name_raises(self):
        with pytest.raises(KeyError):
            run_standing_eval("nope")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
