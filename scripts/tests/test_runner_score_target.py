"""PROC-5 Part B — the runner must propagate a transform's declared target to the scorer.

Chaining `triple_barrier --score-with mi_ksg` is pointless unless the scorer is told to
score the derived label (`tb_label`) as its target — otherwise mi_ksg silently falls back
to forward returns and the "3-bar classifier evaluation" scores the wrong thing.

A TransformProcess now declares its primary target via ``target_column()``; the runner
resolves it (an explicit ``--score-target`` wins) and sets it on the scorer's context.
"""

from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from processes.base import TransformProcess
from processes.registry import get_process
from processes.runner import _build_score_frame, _chain_load_columns, _resolve_score_target


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


class TestBuildScoreFrame:
    """Label mode must hand the scorer the ORIGINAL features + the derived label.

    Scoring tb_label against only its own tb_* siblings is vacuous — mi_ksg excludes
    them as leakage, leaving nothing to score. The real-data smoke exposed exactly this.
    """

    def _frames(self):
        n = 50
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "raw_midprice": 100.0 + rng.standard_normal(n).cumsum(),
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        })
        derived = pd.DataFrame({
            "tb_label": rng.integers(-1, 2, n).astype(float),
            "tb_ret": rng.standard_normal(n),
            "tb_hit_bars": rng.integers(1, 16, n).astype(float),
        }, index=frame.index)
        return frame, derived

    def test_label_mode_includes_original_features_and_label(self):
        frame, derived = self._frames()
        sf = _build_score_frame(frame, derived, "raw_midprice", "tb_label")
        for col in ("feat_a", "feat_b", "tb_label", "raw_midprice"):
            assert col in sf.columns, col
        assert len(sf) == len(frame)

    def test_feature_mode_is_derived_plus_price_only(self):
        frame, derived = self._frames()
        sf = _build_score_frame(frame, derived, "raw_midprice", None)
        assert "raw_midprice" in sf.columns          # for forward returns
        assert "tb_label" in sf.columns
        assert "feat_a" not in sf.columns            # original features NOT re-scored
        assert "feat_b" not in sf.columns

    def test_label_mode_derived_wins_on_column_clash(self):
        frame, derived = self._frames()
        frame["tb_label"] = 99.0                     # stale same-named column
        sf = _build_score_frame(frame, derived, "raw_midprice", "tb_label")
        assert (sf["tb_label"].to_numpy() == derived["tb_label"].to_numpy()).all()


class TestChainLoadColumns:
    """The loader must load the UNION of the transform's and the chained scorer's columns.

    triple_barrier needs only price, so pruning to ITS required_columns loads zero
    features — the scorer then has nothing to score. The real-data smoke exposed this:
    proc_mi_ksg_BTC run with features_tested=[].
    """

    AVAILABLE = ["timestamp_ns", "symbol", "raw_midprice",
                 "imbalance_qty_l1", "toxic_vpin_50", "ent_book_shape_mean"]

    def test_union_includes_scorer_features(self):
        tb = get_process("triple_barrier")
        mi = get_process("mi_ksg")
        cols = _chain_load_columns(tb, mi, self.AVAILABLE)
        assert "imbalance_qty_l1" in cols
        assert "toxic_vpin_50" in cols

    def test_no_scorer_keeps_transform_pruning(self):
        tb = get_process("triple_barrier")
        cols = _chain_load_columns(tb, None, self.AVAILABLE)
        assert "imbalance_qty_l1" not in cols        # triple_barrier alone needs no features

    def test_scorer_feature_filter_respected(self):
        tb = get_process("triple_barrier")
        mi = get_process("mi_ksg", features=["toxic_"])
        cols = _chain_load_columns(tb, mi, self.AVAILABLE)
        assert "toxic_vpin_50" in cols
        assert "imbalance_qty_l1" not in cols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
