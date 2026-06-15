"""Contract test for the signal-lifecycle spine (plan T3) — written test-first.

Encodes the state-machine + provenance contract before scripts/signal_lifecycle.py
exists: legal walk, illegal transitions raise, terminal states are terminal, every
row is git_sha-stamped, unknown signals raise, state filtering, idempotent migration.

Run: pytest scripts/tests/test_signal_lifecycle.py -q
"""

from __future__ import annotations

import pytest

from signal_lifecycle import (
    SignalLifecycle, IllegalTransition, UnknownSignal,
    DISCOVERED, VALIDATED, PAPER_TRADING, APPROVAL_PENDING, LIVE, MONITORING,
    RETIRED, REJECTED,
)


@pytest.fixture
def lc(tmp_path):
    sl = SignalLifecycle(db_path=tmp_path / "nat.db")
    yield sl
    sl.close()


def test_full_legal_walk(lc):
    lc.discover("sig1", name="jump_detector")
    assert lc.get_signal("sig1")["state"] == DISCOVERED
    lc.validate("sig1")
    lc.start_paper("sig1")
    lc.request_approval("sig1")
    lc.approve("sig1")
    assert lc.get_signal("sig1")["state"] == LIVE
    lc.monitor("sig1")
    lc.retire("sig1", reason="ic_decay")
    assert lc.get_signal("sig1")["state"] == RETIRED
    states = [h["to_state"] for h in lc.history("sig1")]
    assert states == [DISCOVERED, VALIDATED, PAPER_TRADING, APPROVAL_PENDING,
                      LIVE, MONITORING, RETIRED]


def test_illegal_transition_raises(lc):
    lc.discover("sig2")
    with pytest.raises(IllegalTransition):
        lc.approve("sig2")        # DISCOVERED -> LIVE not allowed
    with pytest.raises(IllegalTransition):
        lc.start_paper("sig2")    # DISCOVERED -> PAPER_TRADING (skips VALIDATED)


def test_terminal_states_are_terminal(lc):
    lc.discover("sig3")
    lc.reject("sig3", reason="cost_killed")
    assert lc.get_signal("sig3")["state"] == REJECTED
    with pytest.raises(IllegalTransition):
        lc.validate("sig3")       # REJECTED is terminal


def test_reject_allowed_pre_live(lc):
    lc.discover("s"); lc.validate("s"); lc.start_paper("s")
    lc.reject("s", reason="weak")
    assert lc.get_signal("s")["state"] == REJECTED


def test_every_row_carries_git_sha(lc):
    lc.discover("sigp", name="x")
    lc.validate("sigp")
    assert lc.get_signal("sigp")["git_sha"], "signal row missing git_sha"
    hist = lc.history("sigp")
    assert hist and all(h["git_sha"] for h in hist), "a history row is missing git_sha"


def test_unknown_signal_raises(lc):
    with pytest.raises(UnknownSignal):
        lc.validate("does_not_exist")


def test_duplicate_discover_raises(lc):
    lc.discover("dup")
    with pytest.raises(Exception):
        lc.discover("dup")


def test_list_filters_by_state(lc):
    lc.discover("a")
    lc.discover("b"); lc.validate("b")
    assert {s["signal_id"] for s in lc.list_signals(state=DISCOVERED)} == {"a"}
    assert {s["signal_id"] for s in lc.list_signals(state=VALIDATED)} == {"b"}


def test_migration_idempotent(tmp_path):
    db = tmp_path / "nat.db"
    SignalLifecycle(db_path=db).close()
    SignalLifecycle(db_path=db).close()   # second open must not error
