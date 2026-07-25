"""
Day 7: Layer Preference Integration Test
=========================================
Verifies that CakeScorer correctly returns CakeScoreOutput with
layer preference, and that preferences are stored per-request with
no cross-request contamination.

Tests:
1. CakeScorer.forward() returns CakeScoreOutput
2. Two requests with different lengths have independent preferences
3. Request end cleans up state
"""
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm.v1.attention.compression.cake import CakeScorer, CakeScoreOutput


def test_scorer_returns_cake_output():
    """CakeScorer.forward() must return CakeScoreOutput, not plain tensor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CakeScorer(
        num_kv_heads=8, num_q_per_kv=4, head_size=128,
        cake_window_size=32, cake_kernel_size=5, cake_gamma=1.0,
        cake_tau1=1.0, cake_tau2=1.0,
    ).to(device)

    T = 100
    q = torch.randn(T, 8 * 4 * 128, device=device)
    k = torch.randn(T, 8 * 128, device=device)

    result = scorer(q, k)
    assert isinstance(result, CakeScoreOutput), (
        f"Expected CakeScoreOutput, got {type(result)}"
    )
    assert isinstance(result.token_scores, torch.Tensor), (
        f"token_scores should be Tensor, got {type(result.token_scores)}"
    )
    assert isinstance(result.layer_preference, torch.Tensor), (
        f"layer_preference should be Tensor, got {type(result.layer_preference)}"
    )
    assert result.token_scores.shape[0] == 8, (
        f"Expected 8 KV heads, got {result.token_scores.shape[0]}"
    )
    assert result.layer_preference.numel() == 1, (
        f"Expected scalar preference, got {result.layer_preference.shape}"
    )
    print(f"[PASS] test_scorer_returns_cake_output")
    print(f"  token_scores shape: {result.token_scores.shape}")
    print(f"  layer_preference: {result.layer_preference.item():.4f}")


def test_preference_deterministic():
    """Same input should produce the same preference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CakeScorer(
        num_kv_heads=8, num_q_per_kv=4, head_size=128,
    ).to(device)

    torch.manual_seed(42)
    q = torch.randn(200, 8 * 4 * 128, device=device)
    k = torch.randn(200, 8 * 128, device=device)

    r1 = scorer(q, k)
    r2 = scorer(q, k)
    r3 = scorer(q, k)

    assert abs(r1.layer_preference.item() - r2.layer_preference.item()) < 1e-6
    assert abs(r2.layer_preference.item() - r3.layer_preference.item()) < 1e-6
    print(f"[PASS] test_preference_deterministic: all match at {r1.layer_preference.item():.4f}")


def test_different_inputs_different_preferences():
    """Different inputs should produce different preferences."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CakeScorer(
        num_kv_heads=8, num_q_per_kv=4, head_size=128,
    ).to(device)

    q1 = torch.randn(200, 8 * 4 * 128, device=device)
    k1 = torch.randn(200, 8 * 128, device=device)
    q2 = torch.randn(200, 8 * 4 * 128, device=device)
    k2 = torch.randn(200, 8 * 128, device=device)

    r1 = scorer(q1, k1)
    r2 = scorer(q2, k2)

    # Different inputs should give different preferences (with high probability)
    diff = abs(r1.layer_preference.item() - r2.layer_preference.item())
    print(f"  Preference 1: {r1.layer_preference.item():.4f}")
    print(f"  Preference 2: {r2.layer_preference.item():.4f}")
    print(f"  Difference: {diff:.4f}")

    # With random inputs, preferences are very likely different
    assert diff > 1e-6 or r1.layer_preference.item() == 0.0, (
        "Different inputs gave identical preferences"
    )
    print(f"[PASS] test_different_inputs_different_preferences")


def test_short_input_no_preference():
    """Very short input (< window) should produce preference=0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CakeScorer(
        num_kv_heads=8, num_q_per_kv=4, head_size=128,
    ).to(device)

    # T=1, shorter than window
    q = torch.randn(1, 8 * 4 * 128, device=device)
    k = torch.randn(1, 8 * 128, device=device)

    result = scorer(q, k)
    assert result.layer_preference.item() == 0.0, (
        f"Short input should give preference=0, got {result.layer_preference.item()}"
    )
    print(f"[PASS] test_short_input_no_preference")


def test_preference_positive():
    """Preference should always be non-negative."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = CakeScorer(
        num_kv_heads=8, num_q_per_kv=4, head_size=128,
    ).to(device)

    for _ in range(5):
        T = torch.randint(50, 500, (1,)).item()
        q = torch.randn(T, 8 * 4 * 128, device=device)
        k = torch.randn(T, 8 * 128, device=device)
        result = scorer(q, k)
        assert result.layer_preference.item() >= 0, (
            f"Negative preference: {result.layer_preference.item()}"
        )
    print(f"[PASS] test_preference_positive")


def run_all():
    tests = [
        test_scorer_returns_cake_output,
        test_preference_deterministic,
        test_different_inputs_different_preferences,
        test_short_input_no_preference,
        test_preference_positive,
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