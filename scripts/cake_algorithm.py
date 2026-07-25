"""
CAKE algorithm standalone functions (pure PyTorch).

These functions implement the core CAKE algorithm for later integration
into Tangram's scorer.py and selection_level.py. They operate on raw
PyTorch tensors with no vLLM dependencies.

Usage:
    from cake_algorithm import compute_cake_scores, allocate_cake_budgets
    
    scores, pref = compute_cake_scores(query, key, ...)
    budgets = allocate_cake_budgets(pref_scores, total_budget, eval_len, ...)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CAKE token scorer (maps to Tangram's CakeScorer.forward)
# ---------------------------------------------------------------------------

def compute_cake_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_size: int,
    window_size: int = 32,
    kernel_size: int = 5,
    gamma: float = 1.0,
    tau1: float = 1.0,
    tau2: float = 1.0,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute CAKE token scores and layer preference from post-RoPE Q/K.

    Args:
        query: [T, num_kv_heads * num_q_per_kv * head_size] post-RoPE, token-major.
        key:   [T, num_kv_heads * head_size] post-RoPE, token-major.
        num_kv_heads: Number of KV heads.
        num_q_per_kv: GQA ratio (Q heads per KV head).
        head_size: Dimension per head.
        window_size: Observation window (last N queries).
        kernel_size: Smoothing kernel for token scores.
        gamma: Weight for temporal variance in token score.
        tau1: Entropy exponent in layer preference.
        tau2: Variance exponent in layer preference.
        eps: Small constant for log stability.

    Returns:
        token_scores:   [num_kv_heads, T] float32, higher = more important.
        layer_pref:     scalar float32, layer preference score.
    """
    device = query.device
    dtype = query.dtype
    chunk_len = query.shape[0]

    # Reshape to [T, num_kv_heads, num_q_per_kv, head_size] / [T, num_kv_heads, head_size]
    q = query.reshape(chunk_len, num_kv_heads, num_q_per_kv, head_size)
    k = key.reshape(chunk_len, num_kv_heads, head_size)

    # Observation window: at most window_size, but at least 1
    window = min(window_size, chunk_len)

    # Take last `window` queries: [window, num_kv_heads, num_q_per_kv, head_size]
    q_obs = q[chunk_len - window:]

    # Permute for matmul: [num_kv_heads, num_q_per_kv, window, head_size]
    q_obs = q_obs.permute(1, 2, 0, 3)

    # Key transpose: [num_kv_heads, head_size, T]
    k_t = k.permute(1, 2, 0)

    # Attention scores: [num_kv_heads, num_q_per_kv, window, T]
    scale = math.sqrt(head_size)
    attn = torch.matmul(q_obs, k_t.unsqueeze(1)) / scale

    # --- Layer Preference ---
    # CAKE uses attention on the history region (excluding window) for preference
    hist_len = chunk_len - window
    if hist_len > 0:
        # Attention on history: [num_kv_heads, num_q_per_kv, window, hist_len]
        attn_hist = attn[:, :, :, :hist_len]

        # Softmax over key positions (history)
        attn_soft = F.softmax(attn_hist, dim=-1, dtype=torch.float32)

        # Entropy: sum over all elements of -p*log(p)
        # CAKE formula: calculate_entropy(attention) = -sum(p * log(p))
        entropy = -torch.sum(attn_soft * torch.log(attn_soft + eps))

        # Temporal variance: var over query positions (dim=-2), then sum all
        # CAKE formula: var = torch.var(attention, dim=-2).sum(0).sum(0).sum(0)
        temporal_var = torch.var(attn_soft.float(), dim=-2).sum()

        # Layer preference: P_l = entropy^(1/tau1) * var^(1/tau2)
        layer_pref = (entropy ** (1.0 / tau1) * temporal_var ** (1.0 / tau2)).to(
            dtype=torch.float32
        )
    else:
        layer_pref = torch.tensor(0.0, device=device, dtype=torch.float32)

    # --- Token Scores ---
    # CAKE: S = Mean_q(A) + gamma * Var_q(A), then avg_pool1d
    if hist_len > 0:
        # Softmax over all key positions (including window for token scores)
        attn_full = F.softmax(attn, dim=-1, dtype=torch.float32)

        # Mean and variance over query positions (dim=-2)
        # attn_full: [num_kv_heads, num_q_per_kv, window, T]
        attn_mean = attn_full.mean(dim=-2)  # [num_kv_heads, num_q_per_kv, T]
        attn_var = attn_full.var(dim=-2)  # [num_kv_heads, num_q_per_kv, T]

        # Token score = mean + gamma * var
        raw_score = attn_mean + gamma * attn_var  # [num_kv_heads, num_q_per_kv, T]

        # Remove the observation window (keep only history)
        if chunk_len > window:
            raw_score = raw_score[:, :, :hist_len]  # [num_kv_heads, num_q_per_kv, hist_len]
        else:
            raw_score = raw_score[:, :, :0]

        if raw_score.shape[-1] > 0:
            # Smoothing: avg_pool1d with kernel_size
            hist_len = raw_score.shape[-1]
            # Flatten heads for pooling: [num_kv_heads * num_q_per_kv, 1, hist_len]
            flat = raw_score.reshape(num_kv_heads * num_q_per_kv, 1, hist_len)
            score_smooth = F.avg_pool1d(
                flat,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1,
            )  # [num_kv_heads * num_q_per_kv, 1, hist_len]
            score_smooth = score_smooth.reshape(
                num_kv_heads, num_q_per_kv, hist_len
            )

            # Average over GQA groups: [num_kv_heads, hist_len]
            token_scores = score_smooth.mean(dim=-2)
        else:
            token_scores = torch.zeros(
                num_kv_heads, 0, device=device, dtype=torch.float32
            )
    else:
        token_scores = torch.zeros(
            num_kv_heads, 0, device=device, dtype=torch.float32
        )

    return token_scores, layer_pref


# ---------------------------------------------------------------------------
# CAKE budget allocator (maps to Tangram's CakeLayerLevel)
# ---------------------------------------------------------------------------

def allocate_cake_budgets(
    pref_scores: list[float] | np.ndarray | torch.Tensor,
    total_budget: int,
    eval_len: int,
    num_layers: int,
    block_size: int | None = None,
) -> np.ndarray:
    """Allocate per-layer budgets proportional to CAKE layer preferences.

    This mirrors CAKE's adjust_budgets with additional Page-alignment
    for Tangram integration.

    Args:
        pref_scores: Per-layer preference scores [num_layers].
        total_budget: Total number of tokens to keep across all layers
                      (excluding sink/window).
        eval_len: Length of the evictable region (seq_len - window_size).
        num_layers: Number of layers.
        block_size: Optional block alignment (for Tangram).

    Returns:
        budgets: [num_layers] int64, kept tokens per layer.
    """
    pref = np.asarray(pref_scores, dtype=np.float64)

    if pref.sum() <= 0:
        # Uniform fallback
        budgets = np.full(num_layers, total_budget // num_layers, dtype=np.int64)
    else:
        # Proportional allocation
        raw = pref / pref.sum() * total_budget
        budgets = np.floor(raw).astype(np.int64)

        # Distribute remainder to layers with largest fractional parts
        remainder = int(total_budget - budgets.sum())
        if remainder > 0:
            fractional = raw - budgets
            top = np.argsort(-fractional)[:remainder]
            budgets[top] += 1

    # Cap each budget at eval_len
    capped = np.clip(budgets, 0, eval_len)
    excess = (budgets - capped).sum()  # total tokens removed by capping
    budgets = capped

    # Redistribute excess from capped layers to uncapped layers
    if excess > 0:
        under_target = eval_len - budgets
        valid = under_target > 0
        if valid.any():
            num_valid = valid.sum()
            per_layer = excess // num_valid
            extra = excess % num_valid
            budgets[valid] += per_layer
            if extra > 0:
                valid_indices = np.where(valid)[0]
                top_extra = valid_indices[np.argsort(-under_target[valid])[:extra]]
                budgets[top_extra] += 1
            # Re-clip after redistribution to prevent overflow
            budgets = np.minimum(budgets, eval_len)

    # Final adjustment: ensure sum matches total_budget exactly
    diff = int(total_budget - budgets.sum())
    max_iter = 10000  # safety limit
    while diff != 0 and max_iter > 0:
        max_iter -= 1
        if diff > 0:
            # Add to layer with most room to grow
            room = eval_len - budgets
            candidates = np.where(room > 0)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(room[candidates])]
            budgets[idx] += 1
            diff -= 1
        else:
            # Remove from layer with most budget (and budget > 0)
            candidates = np.where(budgets > 0)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(budgets[candidates])]
            budgets[idx] -= 1
            diff += 1

    # Block alignment (for Tangram integration)
    if block_size is not None and block_size > 1:
        # Align up to nearest block_size
        aligned = ((budgets + block_size - 1) // block_size) * block_size
        # Clamp to eval_len
        aligned = np.minimum(aligned, eval_len)
        # The aligned budget may exceed total_budget, which is expected
        # in Tangram (block alignment inflates the physical budget)
        budgets = aligned

    return budgets.astype(np.int64)


# ---------------------------------------------------------------------------
# Convenience: end-to-end CAKE pipeline (for testing)
# ---------------------------------------------------------------------------

def cake_pipeline(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_size: int,
    num_layers: int,
    total_budget: int,
    window_size: int = 32,
    kernel_size: int = 5,
    gamma: float = 1.0,
    tau1: float = 1.0,
    tau2: float = 1.0,
    block_size: int | None = None,
) -> tuple[torch.Tensor, np.ndarray, float]:
    """Run full CAKE pipeline: scores + budgets.

    Returns:
        token_scores: [num_layers, num_kv_heads, hist_len] or averaged.
        budgets: [num_layers] int64.
        avg_pref: average layer preference (for logging).
    """
    # Compute scores (single layer call)
    token_scores, layer_pref = compute_cake_scores(
        query, key,
        num_kv_heads=num_kv_heads,
        num_q_per_kv=num_q_per_kv,
        head_size=head_size,
        window_size=window_size,
        kernel_size=kernel_size,
        gamma=gamma,
        tau1=tau1,
        tau2=tau2,
    )

    # For single-layer call, we need to aggregate preferences across layers
    # This is a placeholder for the multi-layer version

    return token_scores, None, layer_pref.item()