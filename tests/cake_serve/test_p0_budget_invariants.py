"""
P0 budget invariants for CakeLayerLevel page group budget mapping.

Run: pytest tests/cake_serve/test_p0_budget_invariants.py -v
"""
import numpy as np
import pytest
import torch

from vllm.v1.attention.compression.selection_level import CakeLayerLevel, SelectionContext


class TestCakeLayerBudgetInvariants:
    """Verify CakeLayerLevel budget allocation correctness."""

    def test_basic_ratios(self):
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 32, 4, 2048
        np.random.seed(42)
        prefs = np.random.uniform(5, 100, num_layers).astype(np.float32)
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)

        for ratio in [0.1, 0.25, 0.5, 0.75]:
            ctx = SelectionContext(layer_preferences=torch.tensor(prefs))
            scores = torch.zeros(num_layers, num_groups, eval_len)
            counts = level.compute_counts(scores, ratio, dummy_cluster,
                                          num_layers, 8, num_groups, ctx)
            expected = eval_len * num_layers * num_groups * ratio
            actual = counts.sum()
            err = num_layers * num_groups
            assert abs(actual - expected) <= err, f"ratio={ratio}: {actual} vs {expected}"

    def test_group_counts(self):
        level = CakeLayerLevel()
        eval_len, num_layers, ratio = 2048, 8, 0.25
        prefs = np.random.uniform(5, 100, num_layers).astype(np.float32)
        ctx = SelectionContext(layer_preferences=torch.tensor(prefs))

        for num_groups in [1, 2, 4, 8]:
            dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
            scores = torch.zeros(num_layers, num_groups, eval_len)
            counts = level.compute_counts(scores, ratio, dummy_cluster,
                                          num_layers, 8, num_groups, ctx)
            expected = eval_len * num_layers * num_groups * ratio
            assert abs(counts.sum() - expected) <= num_layers * num_groups
            # All groups within a layer share the same budget
            for l in range(num_layers):
                assert np.all(counts[l] == counts[l, 0])

    def test_zero_prefs_uniform_fallback(self):
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 16, 4, 2048
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        ctx = SelectionContext(layer_preferences=torch.zeros(num_layers))
        scores = torch.zeros(num_layers, num_groups, eval_len)
        counts = level.compute_counts(scores, 0.25, dummy_cluster,
                                      num_layers, 8, num_groups, ctx)
        assert counts.sum() > 0, "Zero prefs should fallback to uniform"

    def test_extreme_pref_capped(self):
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 16, 4, 2048
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        prefs = np.ones(num_layers, dtype=np.float32)
        prefs[7] = 1000.0
        ctx = SelectionContext(layer_preferences=torch.tensor(prefs))
        scores = torch.zeros(num_layers, num_groups, eval_len)
        counts = level.compute_counts(scores, 0.25, dummy_cluster,
                                      num_layers, 8, num_groups, ctx)
        assert np.all(counts <= eval_len), "Budget exceeds eval_len"
        assert counts[7, 0] >= counts[0, 0], "High-pref layer should get more budget"

    def test_nan_prefs_fallback(self):
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 16, 4, 2048
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        prefs = np.full(num_layers, np.nan, dtype=np.float32)
        prefs[0] = 5.0
        ctx = SelectionContext(layer_preferences=torch.tensor(prefs))
        scores = torch.zeros(num_layers, num_groups, eval_len)
        counts = level.compute_counts(scores, 0.25, dummy_cluster,
                                      num_layers, 8, num_groups, ctx)
        assert counts.sum() > 0, "NaN prefs should fallback to uniform"

    def test_skewed_pref_waterfill(self):
        """Skewed prefs that previously overflowed max_iter=10000."""
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 32, 4, 8192
        ratio = 0.75
        dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
        prefs = np.array([12000]*8 + [8000]*12 + [384]*12, dtype=np.float32)
        ctx = SelectionContext(layer_preferences=torch.tensor(prefs))
        scores = torch.zeros(32, 4, 8192)
        counts = level.compute_counts(scores, ratio, dummy, 32, 8, 4, context=ctx)
        expected = int(8192 * 32 * 4 * ratio)
        assert abs(counts.sum() - expected) <= 32 * 4, (
            f"Skewed pref water-fill failed: {counts.sum()} vs {expected}")
        assert counts.max() <= eval_len

    def test_negative_prefs_fallback(self):
        level = CakeLayerLevel()
        num_layers, num_groups, eval_len = 16, 4, 2048
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        prefs = np.random.uniform(-10, -1, num_layers).astype(np.float32)
        ctx = SelectionContext(layer_preferences=torch.tensor(prefs))
        scores = torch.zeros(num_layers, num_groups, eval_len)
        counts = level.compute_counts(scores, 0.25, dummy_cluster,
                                      num_layers, 8, num_groups, ctx)
        assert counts.sum() > 0, "Negative prefs should fallback to uniform"