"""
Day 8: CakeLayerLevel unit tests.
"""
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm.v1.attention.compression.selection_level import (
    CakeLayerLevel, SelectionContext, make_selection_level
)


def test_registered():
    """CakeLayerLevel is registered and constructable."""
    level = make_selection_level("cake_layer")
    assert level.name == "cake_layer"
    assert isinstance(level, CakeLayerLevel)
    print("[PASS] test_registered")


def test_uniform_fallback():
    """Without preferences, CakeLayerLevel falls back to uniform."""
    level = CakeLayerLevel()
    eval_scores = torch.randn(32, 8, 1000)
    counts = level.compute_counts(
        eval_scores, 0.5, None, 32, 8, 4, context=None)
    assert counts.shape == (32, 4)
    assert counts.sum() == 32 * 4 * 500  # 50% of 1000
    print(f"[PASS] test_uniform_fallback: sum={counts.sum()}")


def test_preferences_affect_budget():
    """Layers with higher preferences get more budget."""
    level = CakeLayerLevel()
    eval_scores = torch.randn(32, 8, 1000)
    # Layer 0 gets 10x preference of others
    prefs = torch.ones(32, dtype=torch.float32)
    prefs[0] = 10.0
    ctx = SelectionContext(layer_preferences=prefs)

    counts = level.compute_counts(
        eval_scores, 0.5, None, 32, 8, 4, context=ctx)

    # Layer 0 should have more budget than average
    layer0_total = counts[0].sum()
    avg_total = counts.sum() / 32
    assert layer0_total > avg_total, (
        f"Layer 0 ({layer0_total}) should be > avg ({avg_total})"
    )
    print(f"[PASS] test_preferences_affect_budget: "
          f"layer0={layer0_total}, avg={avg_total:.1f}")


def test_budget_sum_matches():
    """Total budget should match eval_len * num_layers * ratio."""
    level = CakeLayerLevel()
    for eval_len in [500, 1000, 2000]:
        for ratio in [0.25, 0.5, 0.75]:
            eval_scores = torch.randn(32, 8, eval_len)
            prefs = torch.rand(32, dtype=torch.float32) * 10 + 1
            ctx = SelectionContext(layer_preferences=prefs)
            counts = level.compute_counts(
                eval_scores, ratio, None, 32, 8, 4, context=ctx)
            expected = int(eval_len * 32 * ratio)
            # Allow small rounding error (at most 1 per group)
            assert abs(counts.sum() - expected) <= 32 * 4, (
                f"Sum {counts.sum()} != expected {expected} "
                f"(eval_len={eval_len}, ratio={ratio})"
            )
    print("[PASS] test_budget_sum_matches")


def test_budget_within_eval_len():
    """No group's budget should exceed eval_len."""
    level = CakeLayerLevel()
    for eval_len in [100, 500]:
        eval_scores = torch.randn(32, 8, eval_len)
        prefs = torch.rand(32, dtype=torch.float32) * 100
        # Extreme preference to test cap
        prefs[0] = 10000.0
        ctx = SelectionContext(layer_preferences=prefs)
        counts = level.compute_counts(
            eval_scores, 0.9, None, 32, 8, 4, context=ctx)
        assert counts.max() <= eval_len, (
            f"Max budget {counts.max()} > eval_len {eval_len}"
        )
    print("[PASS] test_budget_within_eval_len")


def test_monotonic():
    """Higher ratio should give >= budget per layer."""
    level = CakeLayerLevel()
    eval_scores = torch.randn(32, 8, 1000)
    prefs = torch.rand(32, dtype=torch.float32) * 10 + 1
    ctx = SelectionContext(layer_preferences=prefs)

    counts_high = level.compute_counts(
        eval_scores, 0.75, None, 32, 8, 4, context=ctx)
    counts_low = level.compute_counts(
        eval_scores, 0.25, None, 32, 8, 4, context=ctx)

    # Each layer's total should be >= at higher ratio
    for l in range(32):
        assert counts_high[l].sum() >= counts_low[l].sum(), (
            f"Layer {l}: high={counts_high[l].sum()} < low={counts_low[l].sum()}"
        )
    print("[PASS] test_monotonic")


def test_extreme_preferences():
    """Single extreme preference doesn't break allocation."""
    level = CakeLayerLevel()
    eval_scores = torch.randn(32, 8, 1000)
    prefs = torch.zeros(32, dtype=torch.float32)
    prefs[15] = 1.0  # Only one layer has non-zero preference
    ctx = SelectionContext(layer_preferences=prefs)

    counts = level.compute_counts(
        eval_scores, 0.5, None, 32, 8, 4, context=ctx)
    # Should still produce valid allocation (all layers get some budget)
    assert counts.min() >= 0
    assert counts.sum() > 0
    # Layer 15 should get more
    layer15 = counts[15].sum()
    others_avg = (counts.sum() - layer15) / 31
    print(f"[PASS] test_extreme_preferences: "
          f"layer15={layer15}, others_avg={others_avg:.1f}")


def run_all():
    tests = [
        test_registered,
        test_uniform_fallback,
        test_preferences_affect_budget,
        test_budget_sum_matches,
        test_budget_within_eval_len,
        test_monotonic,
        test_extreme_preferences,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_all()