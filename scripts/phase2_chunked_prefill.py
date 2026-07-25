"""
Phase 2: Chunked Prefill Verification
======================================
Simulate chunked prefill by splitting Q/K into chunks and computing CAKE
scores per chunk. Verify that token-weighted preference aggregation converges
to the one-shot (full prompt) preferences.

This tests the CAKE-Chunk aggregation logic without running the full vLLM
stack. The key hypothesis: as we process more chunks, the accumulated
preference should converge to the one-shot preference.

Usage:
    conda activate cake-serve
    cd ~/cake-serve
    CUDA_VISIBLE_DEVICES=0 python scripts/phase2_chunked_prefill.py
"""
import json
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

MODEL_PATH = "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "results/raw/day11_phase2_chunked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference params (same as phase1)
TAU1 = 1.6
TAU2 = 0.4
GAMMA = 200.0
WINDOW_SIZE = 32
KERNEL_SIZE = 5
CACHE_SIZE = 1024

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

# Make it longer for chunked prefill testing
PROMPT = PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT


def compute_cake_scores(q, k, num_q_per_kv, num_kv_heads, head_size,
                        window_size=32, kernel_size=5, gamma=200.0,
                        tau1=1.6, tau2=0.4, eps=1e-10):
    """Same as CakeScorer.forward()."""
    T = q.shape[0]
    scale = math.sqrt(head_size)
    
    q_reshaped = q.reshape(T, num_kv_heads, num_q_per_kv, head_size)
    k_reshaped = k.reshape(T, num_kv_heads, head_size)
    
    window = window_size if T >= 1000 else min(16, T)
    window = min(window, T - 1) if T > 1 else 1
    
    q_obs = q_reshaped[T - window:].permute(1, 2, 0, 3)
    k_t = k_reshaped.permute(1, 2, 0)
    
    attn = torch.matmul(q_obs, k_t.unsqueeze(1)) / scale
    
    # Causal mask
    q_indices = torch.arange(T - window, T, device=attn.device, dtype=torch.long)
    k_indices = torch.arange(T, device=attn.device, dtype=torch.long)
    causal_mask = (q_indices[:, None] >= k_indices[None, :])
    attn = attn.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
    
    attn_soft = F.softmax(attn, dim=-1, dtype=torch.float32)
    
    # Layer preference
    hist_len = T - window
    if hist_len > 0:
        attn_hist = attn_soft[:, :, :, :hist_len]
        entropy = -torch.sum(attn_hist * torch.log(attn_hist + eps))
        temporal_var = torch.var(attn_hist, dim=-2).sum()
        layer_pref = (entropy ** (1.0 / tau1) * temporal_var ** (1.0 / tau2)).to(
            dtype=torch.float32)
    else:
        layer_pref = torch.tensor(0.0, device=attn.device, dtype=torch.float32)
    
    # Token scores
    attn_mean = attn_soft.mean(dim=-2)
    attn_var = attn_soft.var(dim=-2)
    raw_score = attn_mean + gamma * attn_var
    score_gqa = raw_score.mean(dim=-2)
    score = F.avg_pool1d(
        score_gqa.unsqueeze(1), kernel_size=kernel_size,
        padding=kernel_size // 2, stride=1,
    ).squeeze(1)
    
    return score, layer_pref


def allocate_cake_budgets(pref_scores, total_budget, eval_len, num_layers):
    """Same as CakeLayerLevel.compute_counts()."""
    pref_arr = np.array(pref_scores, dtype=np.float64)
    if pref_arr.sum() <= 0:
        return np.full(num_layers, total_budget // num_layers, dtype=np.int64)
    
    pref_norm = pref_arr / pref_arr.sum()
    raw_budgets = pref_norm * total_budget
    budgets = np.floor(raw_budgets).astype(np.int64)
    remainder = int(total_budget - budgets.sum())
    if remainder > 0:
        fractional = raw_budgets - budgets
        top = np.argsort(-fractional)[:remainder]
        budgets[top] += 1
    
    budgets = np.minimum(budgets, eval_len)
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
    print("Phase 2: Chunked Prefill Verification")
    print("=" * 70)
    
    device = "cuda:0"
    dtype = torch.bfloat16
    
    # Load model
    print("\n[1] Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    num_attn_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_attn_heads
    num_q_per_kv = num_attn_heads // num_kv_heads
    
    # Tokenize
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[-1]
    print(f"  Input length: {input_len} tokens")
    
    # Run forward pass and capture post-RoPE Q/K
    print("\n[2] Capturing post-RoPE Q/K...")
    qk_outputs = {}
    
    def make_qk_hook(name):
        def hook(module, args, kwargs, output):
            hidden_states = None
            if 'hidden_states' in kwargs:
                hidden_states = kwargs['hidden_states']
            elif args:
                hidden_states = args[0]
            if hidden_states is None:
                return
            q = module.q_proj(hidden_states).detach()
            k = module.k_proj(hidden_states).detach()
            position_embeddings = kwargs.get('position_embeddings')
            if position_embeddings is not None:
                cos, sin = position_embeddings
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                bs, seq_len = hidden_states.shape[:2]
                hidden_shape = (bs, seq_len, -1, module.head_dim)
                q_view = q.view(hidden_shape).transpose(1, 2)
                k_view = k.view(hidden_shape).transpose(1, 2)
                q_rope, k_rope = apply_rotary_pos_emb(q_view, k_view, cos, sin)
                q = q_rope.transpose(1, 2).reshape(bs, seq_len, -1)
                k = k_rope.transpose(1, 2).reshape(bs, seq_len, -1)
            qk_outputs[name] = {'q': q, 'k': k}
        return hook
    
    hooks = []
    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        hook = attn.register_forward_hook(
            make_qk_hook(f"layer_{i}"), with_kwargs=True)
        hooks.append(hook)
    
    with torch.inference_mode():
        model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
              use_cache=False, return_dict=True)
    
    for hook in hooks:
        hook.remove()
    
    print(f"  Captured {len(qk_outputs)} layers")
    
    # === One-shot (full prompt) ===
    print("\n[3] Computing one-shot CAKE scores (full prompt)...")
    T = input_len
    oneshot_prefs = []
    for i in range(num_layers):
        q_i = qk_outputs[f'layer_{i}']['q'].squeeze(0)
        k_i = qk_outputs[f'layer_{i}']['k'].squeeze(0)
        _, pref = compute_cake_scores(
            q_i, k_i, num_q_per_kv, num_kv_heads, head_dim,
            window_size=WINDOW_SIZE, kernel_size=KERNEL_SIZE,
            gamma=GAMMA, tau1=TAU1, tau2=TAU2)
        oneshot_prefs.append(pref.item())
    
    oneshot_prefs = np.array(oneshot_prefs)
    print(f"  One-shot preferences range: [{oneshot_prefs.min():.4f}, {oneshot_prefs.max():.4f}]")
    
    # === Chunked prefill simulation ===
    print("\n[4] Simulating chunked prefill...")
    
    chunk_sizes = [128, 256, 512, 1024, 2048]
    # Use chunk sizes that divide evenly into the prompt length
    valid_chunks = []
    for cs in [128, 256, 512, 1024, 2048]:
        n_chunks = (T + cs - 1) // cs  # ceil division
        if n_chunks >= 2:  # at least 2 chunks
            valid_chunks.append(cs)
    # Always include the full prompt as reference
    valid_chunks = valid_chunks + [T]
    
    results = {}
    
    for chunk_size in valid_chunks:
        n_chunks = max(1, T // chunk_size)
        actual_chunk = T // n_chunks
        print(f"\n  Chunk size: {actual_chunk}, {n_chunks} chunks")
        
        # Accumulate preferences using token-weighted average
        chunk_prefs = np.zeros(num_layers)
        total_tokens = 0
        
        for chunk_idx in range(n_chunks):
            start = chunk_idx * actual_chunk
            end = min((chunk_idx + 1) * actual_chunk, T)
            chunk_len = end - start
            
            for layer_idx in range(num_layers):
                q_i = qk_outputs[f'layer_{layer_idx}']['q'].squeeze(0)[start:end]
                k_i = qk_outputs[f'layer_{layer_idx}']['k'].squeeze(0)[start:end]
                _, pref = compute_cake_scores(
                    q_i, k_i, num_q_per_kv, num_kv_heads, head_dim,
                    window_size=WINDOW_SIZE, kernel_size=KERNEL_SIZE,
                    gamma=GAMMA, tau1=TAU1, tau2=TAU2)
                
                # Token-weighted average
                if total_tokens == 0 and chunk_idx == 0:
                    chunk_prefs[layer_idx] = pref.item() * chunk_len
                else:
                    chunk_prefs[layer_idx] += pref.item() * chunk_len
            
            total_tokens += chunk_len
            
            # Normalize at the end of each chunk
            current_prefs = chunk_prefs / total_tokens
            
            # Compute Spearman with one-shot so far
            sr, _ = spearmanr(current_prefs, oneshot_prefs)
            budget_mae = np.abs(current_prefs - oneshot_prefs).mean()
            
            print(f"    Chunk {chunk_idx+1}/{n_chunks} "
                  f"(tokens {start}-{end}, cum={total_tokens}): "
                  f"Spearman ρ={sr:.4f}, MAE={budget_mae:.4f}")
        
        # Final comparison
        final_prefs = chunk_prefs / total_tokens
        sr, sp = spearmanr(final_prefs, oneshot_prefs)
        mae = np.abs(final_prefs - oneshot_prefs).mean()
        max_err = np.abs(final_prefs - oneshot_prefs).max()
        
        results[f"chunk_{actual_chunk}"] = {
            "chunk_size": actual_chunk,
            "n_chunks": n_chunks,
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "mae": float(mae),
            "max_err": float(max_err),
            "final_prefs": final_prefs.tolist(),
        }
        
        print(f"    FINAL: Spearman ρ={sr:.4f}, MAE={mae:.4f}, MaxErr={max_err:.4f}")
    
    # === Budget comparison ===
    print("\n[5] Budget comparison...")
    eval_len = T - WINDOW_SIZE
    total_budget = (CACHE_SIZE - WINDOW_SIZE) * num_layers
    
    oneshot_budgets = allocate_cake_budgets(
        oneshot_prefs, total_budget, eval_len, num_layers)
    
    for cs_name, cs_data in results.items():
        chunk_prefs_arr = np.array(cs_data["final_prefs"])
        chunk_budgets = allocate_cake_budgets(
            chunk_prefs_arr, total_budget, eval_len, num_layers)
        budget_mae = np.abs(chunk_budgets - oneshot_budgets).mean()
        cs_data["budget_mae"] = float(budget_mae)
        print(f"  {cs_name}: budget MAE={budget_mae:.1f}")
    
    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  One-shot preferences: Spearman=1.0 (reference)")
    print(f"\n  Chunked prefill convergence:")
    for cs_name, cs_data in sorted(results.items()):
        print(f"    {cs_name}: ρ={cs_data['spearman_r']:.4f}, "
              f"MAE={cs_data['mae']:.4f}, "
              f"budget MAE={cs_data['budget_mae']:.1f}")
    
    # Check which chunk sizes pass
    print(f"\n  Verification targets:")
    for cs_name, cs_data in sorted(results.items()):
        n = cs_data["n_chunks"]
        sr = cs_data["spearman_r"]
        passed = "✅" if sr >= 0.85 else "❌"
        print(f"    {passed} {cs_name}: {n} chunks, Spearman ρ={sr:.4f}")
    
    # === Save ===
    print(f"\n[6] Saving to {OUTPUT_DIR}/...")
    
    output = {
        "config": {
            "tau1": TAU1, "tau2": TAU2, "gamma": GAMMA,
            "window_size": WINDOW_SIZE, "input_length": T,
        },
        "oneshot_preferences": oneshot_prefs.tolist(),
        "oneshot_budgets": oneshot_budgets.tolist(),
        "chunked_results": results,
    }
    
    with open(os.path.join(OUTPUT_DIR, "chunked_prefill_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f_name)
        print(f"    {f_name:40s}  {os.path.getsize(fpath):,} bytes")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()