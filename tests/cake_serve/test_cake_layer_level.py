"""CakeLayerLevel unit tests (pytest).

Run: pytest tests/cake_serve/test_cake_layer_level.py -v
"""
import numpy as np
import torch

from vllm.v1.attention.compression.selection_level import (
    CakeLayerLevel,
    SelectionContext,
    SELECTION_LEVELS,
)


class TestCakeLayerLevel:
    """Unit tests for CakeLayerLevel budget allocation."""

    def test_registered(self):
        assert "cake_layer" in SELECTION_LEVELS
        level = CakeLayerLevel()
        assert level.name == "cake_layer"

    def test_uniform_fallback(self):
        level = CakeLayerLevel()
        scores = torch.zeros(32, 4, 1000)
        dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
        counts = level.compute_counts(scores, 0.5, dummy, 32, 8, 4, context=None)
        assert counts.shape == (32, 4)

    def test_prefs_affect_budget(self):
        level = CakeLayerLevel()
        scores = torch.zeros(32, 4, 1000)
        dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
        prefs = torch.ones(32, dtype=torch.float32)
        prefs[0] = 100.0  # Layer 0 is very important
        ctx = SelectionContext(layer_preferences=prefs)
        counts = level.compute_counts(scores, 0.5, dummy, 32, 8, 4, context=ctx)
        # Layer 0 should get at least average budget
        assert counts[0].sum() >= counts.mean(axis=(0, 1))

    def test_budget_sum_matches(self):
        level = CakeLayerLevel()
        for eval_len in [500, 1000, 2000]:
            for ratio in [0.25, 0.5, 0.75]:
                dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
                prefs = torch.rand(32) * 10 + 1
                ctx = SelectionContext(layer_preferences=prefs)
                counts = level.compute_counts(
                    torch.randn(32, 4, eval_len), ratio, dummy, 32, 8, 4, context=ctx)
                expected = int(eval_len * 32 * 4 * ratio)
                assert abs(counts.sum() - expected) <= 32 * 4

    def test_budget_within_eval_len(self):
        level = CakeLayerLevel()
        for eval_len in [100, 500]:
            dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
            prefs = torch.rand(32) * 100
            prefs[0] = 10000.0
            ctx = SelectionContext(layer_preferences=prefs)
            counts = level.compute_counts(
                torch.randn(32, 4, eval_len), 0.9, dummy, 32, 8, 4, context=ctx)
            assert counts.max() <= eval_len

    def test_monotonic(self):
        level = CakeLayerLevel()
        scores = torch.randn(32, 4, 1000)
        dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
        ctx = SelectionContext(layer_preferences=torch.rand(32) * 10 + 1)
        hi = level.compute_counts(scores, 0.75, dummy, 32, 8, 4, context=ctx)
        lo = level.compute_counts(scores, 0.25, dummy, 32, 8, 4, context=ctx)
        for l in range(32):
            assert hi[l].sum() >= lo[l].sum()

    def test_extreme_prefs(self):
        level = CakeLayerLevel()
        dummy = torch.zeros(32 * 4 * 8, dtype=torch.long)
        prefs = torch.zeros(32)
        prefs[15] = 1.0
        ctx = SelectionContext(layer_preferences=prefs)
        counts = level.compute_counts(
            torch.randn(32, 4, 1000), 0.5, dummy, 32, 8, 4, context=ctx)
        assert counts.min() >= 0
        assert counts.sum() > 0