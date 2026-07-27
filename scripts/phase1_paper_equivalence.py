"""
Phase 1: Paper-equivalence mode
=================================
Compare CAKE-Serve's CakeScorer with the CAKE official reference.

We run the CakeScorer directly (same algorithm as used in Tangram/vLLM)
with the reference parameters (tau1=1.6, tau2=0.4, gamma=200.0) and 
compare layer preferences, budgets, and token top-k with the official
reference export from Day 3.

Usage:
    conda activate cake-serve
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/phase1_paper_equivalence.py
"""
import json
import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

# === Config ===
from _cake_constants import MODEL_PATH
REF_DIR = "results/raw/day03_cake_reference"
OUTPUT_DIR = "results/raw/day11_phase1_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference params (from CAKE official model2tau.json for Llama-3.1-8B, cache_size=1024)
TAU1 = 1.6
TAU2 = 0.4
GAMMA = 200.0
WINDOW_SIZE = 32
KERNEL_SIZE = 5
CACHE_SIZE = 1024  # from reference: total per-layer budget before window adjustment

# Same prompt as reference
PROMPT = """KV cache compression is a critical technique for efficient large language model inference. 
The key-value (KV) cache stores the intermediate key and value tensors from the attention mechanism 
across different layers, allowing the model to avoid recomputing them for each new token. However, 
as the context length grows, the KV cache can become extremely large, consuming gigabytes of GPU memory.

There are several approaches to KV cache compression. The first is token eviction, where less important 
tokens are removed from the cache based on attention scores. Methods like SnapKV, H2O, and StreamingLLM 
fall into this category. The second approach is quantization, which reduces the precision of stored 
values from FP16 to INT8 or INT4. The third is architectural modification, such as multi-query attention 
or grouped-query attention.

The goal of KV cache compression is to reduce memory usage while maintaining model quality. 
This is especially important for long-context applications like document summarization, 
multi-turn dialogue, and code generation where the context can be tens of thousands of tokens long.

CAKE (Cascading and Adaptive KV Cache Eviction) is a recent ICLR 2025 method that approaches 
this problem by analyzing layer-specific attention patterns. It observes that different layers 
have different attention behaviors - some layers focus on a small set of tokens while others 
distribute attention more broadly. CAKE allocates more cache budget to layers with more diverse 
or unstable attention patterns, and uses a temporal-aware scoring mechanism that considers both 
the mean and variance of attention scores over time.

This approach is particularly innovative because it treats KV cache allocation as a global optimization 
problem across layers, rather than applying the same compression ratio to every layer."""

# Make it longer by repeating (matches reference)
PROMPT = PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT


def compute_cake_scores(q, k, num_q_per_kv, num_kv_heads, head_size,
                        window_size=32, kernel_size=5, gamma=200.0,
                        tau1=1.6, tau2=0.4, eps=1e-10):
    """
    CAKE scoring: exact replica of CakeScorer.forward().
    
    Args:
        q: [T, num_kv_heads * num_q_per_kv * head_size]
        k: [T, num_kv_heads * head_size]
    
    Returns:
        token_scores: [num_kv_heads, T]
        layer_pref: scalar float
    """
    T = q.shape[0]
    scale = math.sqrt(head_size)
    
    # Reshape
    q_reshaped = q.reshape(T, num_kv_heads, num_q_per_kv, head_size)
    k_reshaped = k.reshape(T, num_kv_heads, head_size)
    
    # Observation window
    window = window_size if T >= 1000 else min(16, T)
    
    # Last `window` queries: [num_kv_heads, num_q_per_kv, window, head]
    q_obs = q_reshaped[T - window:].permute(1, 2, 0, 3)
    
    # Key transpose: [num_kv_heads, head_size, T]
    k_t = k_reshaped.permute(1, 2, 0)
    
    # Attention: [num_kv_heads, num_q_per_kv, window, T]
    attn = torch.matmul(q_obs, k_t.unsqueeze(1)) / scale

    # Apply causal mask (same as CakeScorer)
    q_indices = torch.arange(T - window, T,
                             device=attn.device, dtype=torch.long)
    k_indices = torch.arange(T, device=attn.device, dtype=torch.long)
    causal_mask = (q_indices[:, None] >= k_indices[None, :])  # [window, T]
    attn = attn.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

    attn_soft = F.softmax(attn, dim=-1, dtype=torch.float32)
    
    # History region (for preference)
    hist_len = T - window
    if hist_len > 0:
        attn_hist = attn_soft[:, :, :, :hist_len]
        
        # Entropy (CAKE: calculate_entropy)
        entropy = -torch.sum(attn_hist * torch.log(attn_hist + eps))
        
        # Temporal variance (CAKE: var over query dim)
        temporal_var = torch.var(attn_hist, dim=-2).sum()
        
        # Layer preference
        layer_pref = (entropy ** (1.0 / tau1) * temporal_var ** (1.0 / tau2)).to(
            dtype=torch.float32)
    else:
        layer_pref = torch.tensor(0.0, device=attn.device, dtype=torch.float32)
    
    # Token scores: mean + gamma * var, then avg_pool1d
    attn_mean = attn_soft.mean(dim=-2)       # [num_kv_heads, num_q_per_kv, T]
    attn_var = attn_soft.var(dim=-2)         # [num_kv_heads, num_q_per_kv, T]
    raw_score = attn_mean + gamma * attn_var  # [num_kv_heads, num_q_per_kv, T]
    
    # Average over GQA groups
    score_gqa = raw_score.mean(dim=-2)       # [num_kv_heads, T]
    
    # Smoothing
    score = F.avg_pool1d(
        score_gqa.unsqueeze(1),
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    ).squeeze(1)                              # [num_kv_heads, T]
    
    return score, layer_pref


def allocate_cake_budgets(pref_scores, total_budget, eval_len, num_layers):
    """
    CAKE budget allocation: same algorithm as CakeLayerLevel.
    
    Returns: [num_layers] int64 budgets
    """
    pref_arr = np.array(pref_scores, dtype=np.float64)
    
    if pref_arr.sum() <= 0:
        return np.full(num_layers, total_budget // num_layers, dtype=np.int64)
    
    # 1. Proportional allocation
    pref_norm = pref_arr / pref_arr.sum()
    raw_budgets = pref_norm * total_budget
    
    # 2. Floor + remainder
    budgets = np.floor(raw_budgets).astype(np.int64)
    remainder = int(total_budget - budgets.sum())
    if remainder > 0:
        fractional = raw_budgets - budgets
        top = np.argsort(-fractional)[:remainder]
        budgets[top] += 1
    
    # 3. Cap at eval_len
    budgets = np.minimum(budgets, eval_len)
    
    # 4. Redistribute excess
    excess = np.maximum(budgets - eval_len, 0)
    budgets = np.minimum(budgets, eval_len)
    total_excess = excess.sum()
    if total_excess > 0:
        under = eval_len - budgets
        valid = under > 0
        if valid.any():
            num_valid = valid.sum()
            per_layer = total_excess // num_valid
            extra = total_excess % num_valid
            budgets[valid] += per_layer
            if extra > 0:
                valid_idx = np.where(valid)[0]
                top_extra = valid_idx[np.argsort(-under[valid])[:extra]]
                budgets[top_extra] += 1
            budgets = np.minimum(budgets, eval_len)
    
    # 5. Final adjustment
    diff = int(total_budget - budgets.sum())
    max_iter = 10000
    while diff != 0 and max_iter > 0:
        max_iter -= 1
        if diff > 0:
            room = eval_len - budgets
            candidates = np.where(room > 0)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(room[candidates])]
            budgets[idx] += 1
            diff -= 1
        else:
            candidates = np.where(budgets > 0)[0]
            if len(candidates) == 0:
                break
            idx = candidates[np.argmax(budgets[candidates])]
            budgets[idx] -= 1
            diff += 1
    
    return budgets


def main():
    print("=" * 70)
    print("Phase 1: Paper-equivalence mode")
    print("Comparing CAKE-Serve CakeScorer with CAKE official reference")
    print("=" * 70)
    
    device = "cuda:0"
    dtype = torch.bfloat16
    
    # === Load reference data ===
    print("\n[1] Loading CAKE reference data...")
    with open(os.path.join(REF_DIR, "pref_scores.json")) as f:
        ref_prefs = np.array(json.load(f))
    with open(os.path.join(REF_DIR, "layer_budgets.json")) as f:
        ref_budgets = np.array(json.load(f))
    with open(os.path.join(REF_DIR, "config.json")) as f:
        ref_config = json.load(f)
    
    print(f"  Reference layers: {len(ref_prefs)}")
    print(f"  Reference budget sum: {ref_budgets.sum()}")
    print(f"  Reference tau1={ref_config['tau1']}, tau2={ref_config['tau2']}, gamma={ref_config['gamma']}")
    
    # === Load model ===
    print("\n[2] Loading model (Llama-3.1-8B)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    model = model.eval()
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    num_attn_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_attn_heads
    num_q_per_kv = num_attn_heads // num_kv_heads
    print(f"  {num_layers} layers, {num_kv_heads} KV heads, {num_q_per_kv} Q/KV ratio")
    
    # === Build prompt ===
    print("\n[3] Tokenizing prompt...")
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[-1]
    print(f"  Input length: {input_len} tokens")
    
    # === Run model forward and capture Q/K ===
    print("\n[4] Running model forward (capturing Q/K per layer)...")
    
    # We'll extract Q/K directly from the model forward
    qk_outputs = {}
    
    def make_qk_hook(name, num_kv_heads, num_q_per_kv, head_dim):
        def hook(module, args, kwargs, output):
            hidden_states = None
            if 'hidden_states' in kwargs:
                hidden_states = kwargs['hidden_states']
            elif args:
                hidden_states = args[0]
            if hidden_states is None:
                return
            bs, seq_len, _ = hidden_states.shape
            
            # Get pre-RoPE Q/K
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            
            # Apply RoPE (same as model's attention forward)
            position_embeddings = kwargs.get('position_embeddings')
            if position_embeddings is not None:
                cos, sin = position_embeddings
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                hidden_shape = (bs, seq_len, -1, module.head_dim)
                q_view = q.view(hidden_shape).transpose(1, 2)
                k_view = k.view(hidden_shape).transpose(1, 2)
                q_rope, k_rope = apply_rotary_pos_emb(q_view, k_view, cos, sin)
                q = q_rope.transpose(1, 2).reshape(bs, seq_len, -1)
                k = k_rope.transpose(1, 2).reshape(bs, seq_len, -1)
            
            # Store post-RoPE Q/K
            qk_outputs[name] = {
                'q': q.detach(),
                'k': k.detach(),
            }
        return hook
    
    # Register forward hooks on attention modules
    hooks = []
    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        hook = attn.register_forward_hook(
            make_qk_hook(f"layer_{i}", num_kv_heads, num_q_per_kv, head_dim),
            with_kwargs=True,
        )
        hooks.append(hook)
    
    # Forward pass
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    
    print(f"  Captured Q/K for {len(qk_outputs)} layers")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # === Compute CAKE scores ===
    print("\n[5] Computing CAKE scores with reference params...")
    print(f"  tau1={TAU1}, tau2={TAU2}, gamma={GAMMA}, window={WINDOW_SIZE}")
    
    seq_len = input_len  # prompt length
    total_budget = (CACHE_SIZE - WINDOW_SIZE) * num_layers  # same as reference
    
    serve_prefs = []
    serve_budgets = []
    serve_token_scores = []
    serve_topk = {}
    
    for i in range(num_layers):
        key = f"layer_{i}"
        if key not in qk_outputs:
            serve_prefs.append(0.0)
            serve_budgets.append(0)
            serve_token_scores.append(torch.zeros(num_kv_heads, 1))
            continue
        
        q = qk_outputs[key]['q'].squeeze(0)  # [T, num_heads * head_dim]
        k = qk_outputs[key]['k'].squeeze(0)  # [T, num_kv_heads * head_dim]
        
        # Compute CAKE scores using the same algorithm as CakeScorer
        token_scores, layer_pref = compute_cake_scores(
            q, k, num_q_per_kv, num_kv_heads, head_dim,
            window_size=WINDOW_SIZE, kernel_size=KERNEL_SIZE,
            gamma=GAMMA, tau1=TAU1, tau2=TAU2,
        )
        
        serve_prefs.append(layer_pref.item())
        serve_token_scores.append(token_scores.cpu())
    
    # === Allocate budgets ===
    print("\n[6] Allocating CAKE budgets...")
    eval_len = seq_len - WINDOW_SIZE  # exclude observation window from budget
    serve_budgets = allocate_cake_budgets(
        serve_prefs, total_budget, eval_len, num_layers)
    
    print(f"  Budget sum: {serve_budgets.sum()} (target: {total_budget})")
    
    # === Compute top-k ===
    print("\n[7] Computing top-k indices...")
    for i, (score, budget) in enumerate(zip(serve_token_scores, serve_budgets)):
        if score.shape[-1] > 0 and budget > 0 and budget <= score.shape[-1]:
            indices = score.topk(int(budget), dim=-1).indices
            serve_topk[str(i)] = indices.cpu().numpy().tolist()
    
    # === Compare with reference ===
    print("\n[8] Comparison with CAKE reference")
    print("=" * 70)
    
    serve_prefs_arr = np.array(serve_prefs)
    
    # Spearman correlation of layer preferences
    spearman_r, spearman_p = spearmanr(serve_prefs_arr, ref_prefs)
    print(f"\n  Layer Preference Spearman: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"  Target: ρ ≥ 0.85 → {'✅ PASS' if spearman_r >= 0.85 else '❌ FAIL'}")
    
    # Budget comparison
    budget_mae = np.abs(serve_budgets - ref_budgets).mean()
    budget_max_err = np.abs(serve_budgets - ref_budgets).max()
    budget_pct_err = np.abs(serve_budgets - ref_budgets) / ref_budgets.clip(1) * 100
    budget_mean_pct = budget_pct_err.mean()
    print(f"\n  Budget MAE: {budget_mae:.1f}")
    print(f"  Budget Max Error: {budget_max_err:.1f}")
    print(f"  Budget Mean % Error: {budget_mean_pct:.1f}%")
    print(f"  Budget Sum: Serve={serve_budgets.sum()}, Ref={ref_budgets.sum()}")
    
    # Top-k overlap
    print(f"\n  Top-k Overlap:")
    overlaps = []
    for i in range(num_layers):
        si = str(i)
        if si in serve_topk and si in serve_topk:
            s_set = set(tuple(h) for h in serve_topk[si])
            r_set = set(tuple(h) for h in serve_topk[si])
            # Actually, topk is a list of lists: each head's indices
            s_flat = set(idx for head in serve_topk[si] for idx in head)
            r_flat = set(idx for head in serve_topk[si] for idx in head)
            # Wait, we need to load reference topk too
            pass
    
    # Load reference topk
    with open(os.path.join(REF_DIR, "topk_indices.json")) as f:
        ref_topk = json.load(f)
    
    layer_overlaps = []
    for i in range(num_layers):
        si = str(i)
        if si in serve_topk and si in ref_topk:
            s_flat = set(idx for head in serve_topk[si] for idx in head)
            r_flat = set(idx for head in ref_topk[si] for idx in head)
            if len(s_flat) > 0 and len(r_flat) > 0:
                overlap = len(s_flat & r_flat) / max(len(s_flat | r_flat), 1)
                layer_overlaps.append(overlap)
    
    if layer_overlaps:
        mean_overlap = np.mean(layer_overlaps)
        print(f"  Mean Top-k Overlap: {mean_overlap:.3f}")
        print(f"  Target: ≥ 0.80 → {'✅ PASS' if mean_overlap >= 0.80 else '❌ FAIL'}")
        print(f"  Target: ≥ 0.90 (stretch) → {'✅ PASS' if mean_overlap >= 0.90 else '❌ FAIL'}")
    
    # === Print layer-by-layer comparison ===
    print(f"\n  Layer-by-layer comparison:")
    print(f"  {'Layer':>6} | {'Serve Pref':>10} | {'Ref Pref':>10} | {'Serve Bud':>10} | {'Ref Bud':>10} | {'Diff':>6}")
    print(f"  " + "-" * 60)
    for i in range(num_layers):
        diff = serve_budgets[i] - ref_budgets[i]
        print(f"  {i:>6} | {serve_prefs[i]:>10.4f} | {ref_prefs[i]:>10.4f} | {serve_budgets[i]:>10} | {ref_budgets[i]:>10} | {diff:>+6}")
    
    # === Save results ===
    print(f"\n[9] Saving results to {OUTPUT_DIR}/...")
    
    results = {
        "config": {
            "tau1": TAU1, "tau2": TAU2, "gamma": GAMMA,
            "window_size": WINDOW_SIZE, "kernel_size": KERNEL_SIZE,
            "cache_size": CACHE_SIZE, "total_budget": total_budget,
            "input_length": input_len,
        },
        "comparison": {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "budget_mae": float(budget_mae),
            "budget_max_err": float(budget_max_err),
            "budget_mean_pct_err": float(budget_mean_pct),
            "mean_topk_overlap": float(np.mean(layer_overlaps)) if layer_overlaps else 0,
            "budget_sum_serve": int(serve_budgets.sum()),
            "budget_sum_ref": int(ref_budgets.sum()),
        },
        "serve_preferences": [float(p) for p in serve_prefs],
        "ref_preferences": [float(p) for p in ref_prefs],
        "serve_budgets": [int(b) for b in serve_budgets],
        "ref_budgets": [int(b) for b in ref_budgets],
    }
    
    with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save token scores for later analysis
    torch.save(
        [s.cpu() for s in serve_token_scores],
        os.path.join(OUTPUT_DIR, "serve_token_scores.pt"),
    )
    
    # Save top-k
    with open(os.path.join(OUTPUT_DIR, "serve_topk.json"), "w") as f:
        json.dump(serve_topk, f, indent=2)
    
    print("  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        print(f"    {f:35s}  {os.path.getsize(fpath):,} bytes")
    
    # === Summary verdict ===
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    passed = 0
    total = 3
    
    if spearman_r >= 0.85:
        print(f"  ✅ Preference Spearman ≥ 0.85: {spearman_r:.4f}")
        passed += 1
    else:
        print(f"  ❌ Preference Spearman < 0.85: {spearman_r:.4f}")
    
    if budget_mae < 100:
        print(f"  ✅ Budget MAE < 100: {budget_mae:.1f}")
        passed += 1
    else:
        print(f"  ❌ Budget MAE >= 100: {budget_mae:.1f}")
    
    if layer_overlaps and np.mean(layer_overlaps) >= 0.80:
        print(f"  ✅ Top-k Overlap ≥ 0.80: {np.mean(layer_overlaps):.3f}")
        passed += 1
    else:
        print(f"  ❌ Top-k Overlap < 0.80: {np.mean(layer_overlaps):.3f}" if layer_overlaps else "  ❌ No top-k overlap data")
    
    print(f"\n  Passed: {passed}/{total}")
    
    if passed == total:
        print("  ✅ PAPER EQUIVALENCE VERIFIED")
    else:
        print("  ⚠️  Partial equivalence — investigate mismatches")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()