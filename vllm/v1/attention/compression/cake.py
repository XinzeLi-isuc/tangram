# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CAKE scorer — temporal-aware attention-based importance score (axis 2).

Produces ``[num_kv_heads, chunk_len]`` scores using CAKE's mean + temporal
variance formulation, smoothed with ``avg_pool1d``. Also computes per-layer
preference scores for non-uniform budget allocation.

Reference: CAKE (ICLR 2025) — https://github.com/antgroup/cakekv

Ported from the CAKE reference ``modify_llama.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vllm.v1.attention.compression.qk_scorer_base import QKScorer


@dataclass
class CakeScoreOutput:
    """Extended scorer output carrying both token scores and layer preference.

    ``token_scores`` is the standard ``[num_kv_heads, T]`` score tensor consumed
    by the shared chunk machinery (sink / window / lock-in / rank).

    ``layer_preference`` is a scalar float32 used by ``CakeLayerLevel`` to
    allocate non-uniform KV budgets across layers.
    """
    token_scores: torch.Tensor
    layer_preference: torch.Tensor  # scalar fp32


class CakeScorer(QKScorer):
    """One (stateless) instance shared across all compressible layers.

    Input:  ``query [T, num_kv_heads * num_q_per_kv * head_size]`` and
            ``key   [T, num_kv_heads * head_size]`` (post-RoPE, token-major
            flatten) for one request's chunk.

    Output: ``CakeScoreOutput`` with:
        - token_scores ``[num_kv_heads, T]`` (float32), higher = more important.
        - layer_preference ``scalar`` (float32), higher = more budget needed.

    CAKE scoring formula:
        S_{l,i} = Mean_q(A_{l,q,i}) + gamma * Var_q(A_{l,q,i})
    followed by avg_pool1d(kernel_size=cake_kernel_size) smoothing.

    Layer preference:
        P_l = Entropy(A_hist)^(1/tau1) * TemporalVar(A_hist)^(1/tau2)
    """

    # Axis-2 dispatch: this scorer reads the inner ``Attention``'s q/k,
    # not the outer block's hidden_states.
    consumes = "qk"
    name = "cake"

    def __init__(
        self,
        num_kv_heads: int,
        num_q_per_kv: int,
        head_size: int,
        cake_window_size: int = 32,
        cake_kernel_size: int = 5,
        cake_gamma: float = 1.0,
        cake_tau1: float = 1.0,
        cake_tau2: float = 1.0,
        cake_eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_q_per_kv
        self.head_size = head_size
        self.cake_window_size = cake_window_size
        self.cake_kernel_size = cake_kernel_size
        self.cake_gamma = cake_gamma
        self.cake_tau1 = cake_tau1
        self.cake_tau2 = cake_tau2
        self.cake_eps = cake_eps
        self._scale = math.sqrt(head_size)

    @torch.no_grad()
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        module: nn.Module | None = None,
        position_offset: int = 0,
    ) -> CakeScoreOutput:
        """Compute CAKE token scores and layer preference.

        Returns ``CakeScoreOutput`` with token scores ``[num_kv_heads, T]``
        and a scalar layer preference.
        """
        del value, module, position_offset
        num_kv_heads = self.num_kv_heads
        num_q_per_kv = self.num_q_per_kv
        head_size = self.head_size
        gamma = self.cake_gamma
        kernel_size = self.cake_kernel_size
        tau1 = self.cake_tau1
        tau2 = self.cake_tau2
        eps = self.cake_eps

        chunk_len = query.shape[0]

        # Reshape: [T, num_kv_heads, num_q_per_kv, head_size] / [T, num_kv_heads, d]
        q = query.reshape(chunk_len, num_kv_heads, num_q_per_kv, head_size)
        k = key.reshape(chunk_len, num_kv_heads, head_size)

        # Observation window: trailing queries only.
        window = min(self.cake_window_size, chunk_len)

        # Use the last `window` queries: [num_kv_heads, num_q_per_kv, window, d]
        q_obs = q[chunk_len - window:].permute(1, 2, 0, 3)

        # Key transpose: [num_kv_heads, d, T]
        k_t = k.permute(1, 2, 0)

        # Attention scores: [num_kv_heads, num_q_per_kv, window, T]
        attn = torch.matmul(q_obs, k_t.unsqueeze(1)) / self._scale

        # Apply causal mask: each query can only attend to keys at or before its position.
        # The observation window queries are at positions [chunk_len-window, chunk_len-1]
        # in the original sequence. Keys at positions [0, chunk_len-1].
        q_indices = torch.arange(chunk_len - window, chunk_len,
                                 device=attn.device, dtype=torch.long)
        k_indices = torch.arange(chunk_len, device=attn.device, dtype=torch.long)
        causal_mask = (q_indices[:, None] >= k_indices[None, :])  # [window, T]
        attn = attn.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

        # Softmax over key positions
        attn_soft = F.softmax(attn, dim=-1, dtype=torch.float32)

        # --- Layer Preference ---
        # Compute preference from the history region (excluding the observation window itself)
        hist_len = chunk_len - window
        if hist_len > 0:
            # Attention on history: [num_kv_heads, num_q_per_kv, window, hist_len]
            attn_hist = attn_soft[:, :, :, :hist_len]

            # Entropy: -sum(p * log(p + eps))
            # CAKE: calculate_entropy(attention)
            entropy = -torch.sum(attn_hist * torch.log(attn_hist + eps))

            # Temporal variance: var over query positions (dim=-2), then sum all
            # CAKE: var = torch.var(attention, dim=-2).sum(0).sum(0).sum(0)
            # Use correction=0 (population variance) to match CAKE reference
            # and avoid NaN when the observation window has only 1 query.
            temporal_var = torch.var(attn_hist, dim=-2, correction=0).sum()

            # Layer preference: P_l = entropy^(1/tau1) * var^(1/tau2)
            layer_pref = (entropy ** (1.0 / tau1) * temporal_var ** (1.0 / tau2)).to(
                dtype=torch.float32
            )
        else:
            layer_pref = torch.tensor(0.0, device=attn.device, dtype=torch.float32)

        # --- Token Scores ---
        # CAKE: S = Mean_q(A) + gamma * Var_q(A), then avg_pool1d
        attn_mean = attn_soft.mean(dim=-2)       # [num_kv_heads, num_q_per_kv, T]
        attn_var = attn_soft.var(dim=-2, correction=0)         # [num_kv_heads, num_q_per_kv, T]

        raw_score = attn_mean + gamma * attn_var  # [num_kv_heads, num_q_per_kv, T]

        # Average over GQA groups: [num_kv_heads, T]
        score_gqa = raw_score.mean(dim=-2)

        # Smooth with avg_pool1d
        score = F.avg_pool1d(
            score_gqa.unsqueeze(1),          # [num_kv_heads, 1, T]
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        ).squeeze(1)                          # [num_kv_heads, T]

        return CakeScoreOutput(
            token_scores=score,
            layer_preference=layer_pref,
        )