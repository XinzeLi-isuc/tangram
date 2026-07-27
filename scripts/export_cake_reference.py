"""
CAKE Reference Export Script v3 (Day 3)
========================================
Direct computation of CAKE scores using manual forward pass.
No flash-attn needed. Uses eager attention with output_attentions=True.

Output: results/raw/day03_cake_reference/
"""
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# === Configuration ===
from _cake_constants import MODEL_PATH
OUTPUT_DIR = "results/raw/day03_cake_reference"

# CAKE hyperparameters (from model2tau.json for Llama-3.1-8B, cache_size=1024)
CACHE_SIZE = 1024
WINDOW_SIZE = 32
TAU1 = 1.6
TAU2 = 0.4
GAMMA = 200.0
KERNEL_SIZE = 5
MAX_NEW_TOKENS = 128
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

# Make it longer by repeating
PROMPT = PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
    json.dump({
        "model": MODEL_PATH, "cache_size": CACHE_SIZE, "window_size": WINDOW_SIZE,
        "tau1": TAU1, "tau2": TAU2, "gamma": GAMMA, "max_new_tokens": MAX_NEW_TOKENS,
        "prompt": PROMPT, "note": "Direct computation via manual forward pass",
    }, f, indent=2)
print("[Config saved]")

# === Step 1: Load model ===
print("\n[Step 1] Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = "cuda:0"
dtype = torch.bfloat16

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
print(f"  {num_layers} layers, {num_attn_heads} Q heads, {num_kv_heads} KV heads, head_dim={head_dim}")

# === Step 2: Build prompt ===
print("\n[Step 2] Building prompt...")
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_len = inputs.input_ids.shape[-1]
print(f"  Input length: {input_len} tokens")

# === Step 3: Register hooks to capture attention ===
print("\n[Step 3] Registering hooks...")
attention_outputs = {}

def make_attn_hook(name):
    # Use forward hook (not forward_hook with kwargs) for simplicity
    def hook(module, input_args, output):
        # output is (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attention_outputs[name] = output[1].detach()
    return hook

hooks = []
for i in range(num_layers):
    attn = model.model.layers[i].self_attn
    hook = attn.register_forward_hook(make_attn_hook(f"layer_{i}"))
    hooks.append(hook)

# === Step 4: Manual forward pass (prefill) ===
print("\n[Step 4] Running forward pass (prefill)...")
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    # Forward pass with output_attentions=True
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
    )

past_key_values = outputs.past_key_values
logits = outputs.logits
print(f"  Prefill done. KV cache seq_len: {past_key_values[0][0].shape[-2]}")
print(f"  Layers with attention captured: {len(attention_outputs)}")

# === Step 5: Compute CAKE scores ===
print("\n[Step 5] Computing CAKE scores...")

total_budget = (CACHE_SIZE - WINDOW_SIZE) * num_layers  # (1024-32)*32 = 31744
seq_len = None  # will be set from the first layer's attention

pref_scores = []
layer_budgets = []
token_scores_list = []
topk_indices = {}

for i in range(num_layers):
    key = f"layer_{i}"
    if key not in attention_outputs:
        print(f"  WARNING: no attention for layer {i}")
        pref_scores.append(0.0)
        layer_budgets.append(0)
        token_scores_list.append(torch.zeros(num_kv_heads, 1))
        continue
    
    attn = attention_outputs[key]  # [bsz, num_heads, q_len, kv_len]
    bsz, num_heads, q_len, kv_len = attn.shape
    if seq_len is None:
        seq_len = kv_len
    
    # Use the prefill attention (q_len == kv_len for causal attention)
    # Reshape for GQA: [bsz, num_heads, q_len, kv_len] -> [bsz, num_kv_heads, num_q_per_kv, q_len, kv_len]
    attn_reshaped = attn.view(bsz, num_kv_heads, num_q_per_kv, q_len, kv_len)
    
    window = min(WINDOW_SIZE, q_len)
    
    # --- Layer Preference ---
    # CAKE uses attention[:, :, -window:, :kv_len-window] for entropy/var
    # The history region (excluding window) is used for preference
    hist_len = kv_len - window
    if hist_len > 0 and q_len >= window:
        # Attention on history region from observation window queries
        obs_on_hist = attn_reshaped[:, :, :, -window:, :hist_len]  # [bsz, num_kv, num_q, window, hist_len]
        
        # Flatten batch and heads for entropy calculation
        flat_attn = obs_on_hist.reshape(-1, window, hist_len)
        
        # Entropy
        entropy = -torch.sum(flat_attn * torch.log(flat_attn + 1e-10))
        
        # Temporal variance (over query positions)
        var = torch.var(flat_attn, dim=-2).sum()
        
        pref = (entropy ** (1.0 / TAU1) * var ** (1.0 / TAU2)).item()
    else:
        pref = 0.0
        print(f"  Layer {i}: hist_len={hist_len}, window={window}, setting pref=0")
    
    pref_scores.append(pref)
    
    # --- Token Score ---
    if hist_len > 0 and q_len >= window:
        # attn_mean = mean over query dimension
        # attn_var = var over query dimension
        attn_mean = obs_on_hist.mean(dim=-2)  # [bsz, num_kv, num_q, hist_len]
        attn_var = obs_on_hist.var(dim=-2)    # [bsz, num_kv, num_q, hist_len]
        
        raw_score = attn_mean + GAMMA * attn_var  # [bsz, num_kv, num_q, hist_len]
        
        # Smoothing
        hist_len = raw_score.shape[-1]
        score_smooth = F.avg_pool1d(
            raw_score.reshape(bsz * num_kv_heads * num_q_per_kv, 1, hist_len),
            kernel_size=KERNEL_SIZE, padding=KERNEL_SIZE // 2, stride=1,
        )
        score_smooth = score_smooth.view(bsz, num_kv_heads, num_q_per_kv, hist_len)
        
        # Average over GQA groups
        hh_score = score_smooth.mean(dim=-2)  # [bsz, num_kv_heads, hist_len]
    else:
        hh_score = torch.zeros(bsz, num_kv_heads, 0, device=attn.device)
    
    token_scores_list.append(hh_score.squeeze(0))  # [num_kv_heads, hist_len]
    
    # --- Layer Budget ---
    # Use the proper CAKE approach: normalize, then apply adjust_budgets
    if sum(pref_scores) > 0:
        budget = pref / sum(pref_scores) * total_budget
    else:
        budget = total_budget / num_layers
    layer_budgets.append(budget)

# === Step 6: Compute layer budgets (CAKE-style) ===
print("\n[Step 6] Computing layer budgets...")

# Normalize preference scores to get proportional budgets
pref_arr = np.array(pref_scores, dtype=np.float64)
if pref_arr.sum() > 0:
    # Initial budget proportional to preference
    raw_budgets = pref_arr / pref_arr.sum() * total_budget
else:
    raw_budgets = np.full(num_layers, total_budget / num_layers)

# Convert to int (floor), then redistribute remainder
floor_budgets = np.floor(raw_budgets).astype(np.int64)
remainder = int(total_budget - floor_budgets.sum())

# Distribute remainder to layers with largest fractional parts
fractional = raw_budgets - floor_budgets
if remainder > 0:
    top_remainder = np.argsort(-fractional)[:remainder]
    floor_budgets[top_remainder] += 1

# Cap each budget at seq_len - WINDOW_SIZE
max_budget = seq_len - WINDOW_SIZE
excess = np.maximum(floor_budgets - max_budget, 0)
floor_budgets = np.minimum(floor_budgets, max_budget)

# Redistribute excess
total_excess = excess.sum()
if total_excess > 0:
    under_target = max_budget - floor_budgets
    valid = under_target > 0
    if valid.any():
        distribute = total_excess // valid.sum()
        remainder_e = total_excess % valid.sum()
        floor_budgets[valid] += distribute
        # Give remainder to the first `remainder_e` valid layers
        valid_indices = np.where(valid)[0]
        floor_budgets[valid_indices[:remainder_e]] += 1

# Final adjustment to match total exactly
diff = total_budget - floor_budgets.sum()
if diff != 0:
    if diff > 0:
        # Need to add: give to layers that can take more
        can_take = max_budget - floor_budgets
        valid = can_take > 0
        if valid.any():
            sorted_idx = np.argsort(-can_take)  # most capacity first
            for idx in sorted_idx:
                if diff <= 0:
                    break
                if can_take[idx] > 0:
                    give = min(diff, can_take[idx])
                    floor_budgets[idx] += give
                    diff -= give
    else:
        # Need to remove: take from layers with enough budget
        can_give = floor_budgets - 1  # keep at least 1
        valid = can_give > 0
        if valid.any():
            sorted_idx = np.argsort(can_give)  # least to spare first
            for idx in sorted_idx:
                if diff >= 0:
                    break
                if can_give[idx] > 0:
                    take = min(-diff, can_give[idx])
                    floor_budgets[idx] -= take
                    diff += take

adjusted_budgets = floor_budgets.tolist()
print(f"  Budget sum: {sum(adjusted_budgets)} (target: {total_budget})")

# === Step 7: Compute top-k indices ===
print("\n[Step 7] Computing top-k indices...")
for i, (score, budget) in enumerate(zip(token_scores_list, adjusted_budgets)):
    if score.shape[-1] > 0 and budget > 0 and budget <= score.shape[-1]:
        indices = score.topk(budget, dim=-1).indices
        topk_indices[str(i)] = indices.cpu().numpy().tolist()

# === Step 8: Generate output text ===
print("\n[Step 8] Generating output...")
with torch.inference_mode():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
output_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
print(f"  Output: {output_text[:200]}")

with open(os.path.join(OUTPUT_DIR, "generation.txt"), "w") as f:
    f.write(output_text)

# === Step 9: Save all results ===
print(f"\n[Step 9] Saving to {OUTPUT_DIR}/...")

# Preference scores
with open(os.path.join(OUTPUT_DIR, "pref_scores.json"), "w") as f:
    json.dump(pref_scores, f, indent=2)
print(f"  pref_scores.json: {len(pref_scores)} layers")

# Layer budgets
with open(os.path.join(OUTPUT_DIR, "layer_budgets.json"), "w") as f:
    json.dump(adjusted_budgets, f, indent=2)
print(f"  layer_budgets.json: {len(adjusted_budgets)} layers, sum={sum(adjusted_budgets)}")

# Token scores (as list of tensors)
torch.save([s.cpu() for s in token_scores_list], os.path.join(OUTPUT_DIR, "token_scores.pt"))
print(f"  token_scores.pt: {len(token_scores_list)} tensors")

# Top-k indices
with open(os.path.join(OUTPUT_DIR, "topk_indices.json"), "w") as f:
    json.dump(topk_indices, f, indent=2)

# Summary
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump({
        "input_length": input_len,
        "seq_length": seq_len,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "pref_scores": pref_scores,
        "layer_budgets": adjusted_budgets,
        "budget_sum": sum(adjusted_budgets),
        "budget_target": total_budget,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }, f, indent=2)

print(f"\n  Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    {f:30s}  {size:,} bytes")

# Remove hooks
for hook in hooks:
    hook.remove()

print("\n[DONE]")