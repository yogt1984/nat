"""PROC-5 Part B — the runner must propagate a transform's declared target to the scorer.

Chaining `triple_barrier --score-with mi_ksg` is pointless unless the scorer is told to
score the derived label (`tb_label`) as its target — otherwise mi_ksg silently falls back
to forward returns and the "3-bar classifier evaluation" scores the wrong thing.

A TransformProcess now declares its primary target via ``target_column()``; the runner
resolves it (an explicit ``--score-target`` wins) and sets it on the scorer's context.
"""

from __future__ import annotations

import pytest

from processes.base import TransformProcess
from processes.registry import get_process
from processes.runner import _resolve_score_target


class TestTargetDeclaration:
    def test_transform_default_target_is_none(self):
        # A generic transform declares no target (e.g. pca_combo emits pc_1..pc_k).
        pca = get_process("pca_combo")
        assert isinstance(pca, TransformProcess)
        assert pca.target_column() is None

    def test_triple_barrier_declares_tb_label(self):
        tb = get_process("triple_barrier")
        assert isinstance(tb, TransformProcess)
        assert tb.target_column() == "tb_label"


class TestResolveScoreTarget:
    def test_auto_resolves_from_transform(self):
        tb = get_process("triple_barrier")
        assert _resolve_score_target(tb, None) == "tb_label"

    def test_explicit_override_wins(self):
        tb = get_process("triple_barrier")
        assert _resolve_score_target(tb, "tb_ret") == "tb_ret"

    def test_none_when_transform_declares_nothing(self):
        pca = get_process("pca_combo")
        assert _resolve_score_target(pca, None) is None

    def test_explicit_target_on_untargeted_transform(self):
        pca = get_process("pca_combo")
        assert _resolve_score_target(pca, "pc_1") == "pc_1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
