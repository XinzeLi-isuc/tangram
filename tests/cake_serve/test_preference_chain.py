"""
P0: Execution-chain integration test — verify CAKE preference survives
the compressor lifecycle (receive_score → prepare_keep_decision → CakeLayerLevel).

Without this fix, pending_preference was cleared by _collect_layer_tensors
before _collect_preferences could read it, causing cake_layer to silently
fallback to uniform.

Run: pytest tests/cake_serve/test_preference_chain.py -v
"""
import numpy as np
import pytest
import torch

from vllm.v1.attention.compression.compressor import (
    KVCompressor,
)
from vllm.v1.attention.compression.cake import CakeScoreOutput
from vllm.v1.attention.compression.selection_level import (
    CakeLayerLevel,
    UniformLevel,
    SelectionContext,
)
from vllm.v1.attention.compression.qk_scorer_base import QKScorer


class _FakeCakeScorer(QKScorer):
    """Fake QK scorer that returns pre-set CakeScoreOutput."""
    consumes = "qk"

    def __init__(self, scores, prefs):
        super().__init__()
        self._scores = scores
        self._prefs = prefs

    def forward(self, query, key, value=None, *, module=None, position_offset=0):
        return CakeScoreOutput(self._scores, self._prefs)


class TestPreferenceChain:
    """Verify CAKE preference survives the full compressor lifecycle."""

    def test_preference_from_receiver_to_cake_layer(self):
        """receive_score → _collect_layer_tensors → CakeLayerLevel receives prefs."""
        num_layers = 4
        num_kv = 2
        num_groups = 2
        head_size = 128
        page_group_size = 1
        block_size = 16
        eval_len = 100

        compressor = KVCompressor(
            num_layers=num_layers,
            num_kv_heads=num_kv,
            page_group_size=page_group_size,
            head_size=head_size,
            hidden_dim=head_size * num_kv * num_layers,
            block_size=block_size,
            dtype=torch.float32,
            device="cpu",
        )
        compressor.compress_active = True
        compressor.level = CakeLayerLevel()

        # Set up identity cluster maps
        compressor.set_cluster_map(None)

        # Simulate receive_score with non-uniform preferences
        req_id = "test_1"
        compressor.begin_request(req_id)
        for layer_idx in range(num_layers):
            score = torch.ones(num_kv, eval_len, dtype=torch.float32)
            # Layer 0 gets high preference, layer 1 gets low
            pref_val = 100.0 if layer_idx == 0 else (1.0 if layer_idx == 1 else 0.0)
            pref = torch.tensor(pref_val, dtype=torch.float32)
            compressor.receive_score(
                req_id, layer_idx, score, layer_preference=pref)

        # Run prepare_keep_decision
        compressor.scorer_consumes = "qk"
        prev_seq = torch.zeros(num_layers, num_groups, dtype=torch.long)
        decision = compressor.prepare_keep_decision(
            req_id=req_id,
            prev_seq_lens_per_layer=prev_seq,
            chunk_len=eval_len,
            ratio=0.25,
            window_size=0,
            n_sink_tokens=0,
            total_prompt_tokens=eval_len,
        )

        # Get the cached budgets
        k_new = compressor.req_state[req_id].cached_k_new_cpu
        assert k_new is not None, "cached_k_new_cpu should not be None"
        assert k_new.shape == (num_layers, num_groups), f"shape={k_new.shape}"

        # Layer 0 (pref=100) should get more budget than layer 1 (pref=1)
        assert k_new[0, 0] > k_new[1, 0], (
            f"Layer-adaptive budget failed: "
            f"layer0={k_new[0, 0]}, layer1={k_new[1, 0]}")
        # Layer 3 (pref=0) should get minimal budget
        assert k_new[3, 0] <= k_new[0, 0], (
            f"Zero-pref layer should get ≤ high-pref layer: "
            f"layer3={k_new[3, 0]}, layer0={k_new[0, 0]}")

        compressor.end_request(req_id)

    def test_no_preference_falls_back_to_uniform(self):
        """SnapKV scorer (no CakeScoreOutput) should produce uniform budget."""
        num_layers = 4
        num_kv = 2
        num_groups = 2
        head_size = 128
        page_group_size = 1
        block_size = 16
        eval_len = 100

        compressor = KVCompressor(
            num_layers=num_layers,
            num_kv_heads=num_kv,
            page_group_size=page_group_size,
            head_size=head_size,
            hidden_dim=head_size * num_kv * num_layers,
            block_size=block_size,
            dtype=torch.float32,
            device="cpu",
        )
        compressor.compress_active = True
        compressor.level = CakeLayerLevel()
        compressor.set_cluster_map(None)

        req_id = "test_2"
        compressor.begin_request(req_id)
        for layer_idx in range(num_layers):
            score = torch.ones(num_kv, eval_len, dtype=torch.float32)
            compressor.receive_score(req_id, layer_idx, score)

        compressor.scorer_consumes = "qk"
        prev_seq = torch.zeros(num_layers, num_groups, dtype=torch.long)
        compressor.prepare_keep_decision(
            req_id=req_id,
            prev_seq_lens_per_layer=prev_seq,
            chunk_len=eval_len,
            ratio=0.25,
            window_size=0,
            n_sink_tokens=0,
            total_prompt_tokens=eval_len,
        )

        k_new = compressor.req_state[req_id].cached_k_new_cpu
        assert k_new is not None
        # Without CAKE preferences, all layers should have same budget
        layer_budgets = k_new.sum(axis=1)
        assert np.allclose(layer_budgets, layer_budgets[0]), (
            f"Without CAKE prefs, budgets should be uniform: {layer_budgets}")

        compressor.end_request(req_id)