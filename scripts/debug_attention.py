"""
Debug: Compare raw attention values between model output and CakeScorer computation.
"""
import json
import torch
import torch.nn.functional as F
import numpy as np
import math
import os

MODEL_PATH = "/home/lixinze/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
OUTPUT_DIR = "results/raw/day11_debug_attention"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda:0"
dtype = torch.bfloat16

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
scale = math.sqrt(head_dim)

# Same prompt as phase1
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

PROMPT = PROMPT + "\n\n" + PROMPT + "\n\n" + PROMPT
prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
input_len = inputs.input_ids.shape[-1]
print(f"Input length: {input_len}")

# Collect Q/K and attention from model forward
qk_outputs = {}
attn_outputs = {}

def make_qk_hook(name):
    def hook(module, args, kwargs, output):
        hidden_states = None
        if 'hidden_states' in kwargs:
            hidden_states = kwargs['hidden_states']
        elif args:
            hidden_states = args[0]
        if hidden_states is None:
            return
        # Get pre-RoPE Q/K
        q = module.q_proj(hidden_states).detach()
        k = module.k_proj(hidden_states).detach()
        
        # Apply RoPE to match model's attention computation
        # RoPE is applied inside the attention forward, so we need to replicate it
        position_embeddings = kwargs.get('position_embeddings')
        if position_embeddings is not None:
            cos, sin = position_embeddings
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            bs, seq_len = hidden_states.shape[:2]
            hidden_shape = (bs, seq_len, -1, module.head_dim)
            q_view = q.view(hidden_shape).transpose(1, 2)  # [bs, num_heads, seq, head]
            k_view = k.view(hidden_shape).transpose(1, 2)  # [bs, num_kv_heads, seq, head]
            q_rope, k_rope = apply_rotary_pos_emb(q_view, k_view, cos, sin)
            q = q_rope.transpose(1, 2).reshape(bs, seq_len, -1).detach()
            k = k_rope.transpose(1, 2).reshape(bs, seq_len, -1).detach()
        
        qk_outputs[name] = {'q': q, 'k': k}
    return hook

hooks = []
for i in range(num_layers):
    attn = model.model.layers[i].self_attn
    hook = attn.register_forward_hook(make_qk_hook(f"layer_{i}"), with_kwargs=True)
    hooks.append(hook)

with torch.inference_mode():
    outputs = model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

for hook in hooks:
    hook.remove()

# Compare layer 0 attention
print(f"\n--- Layer 0 Attention Comparison ---")
T = input_len
WINDOW = min(32, max(1, T - 1))  # can't exceed T-1
if T >= 1000:
    WINDOW = min(32, T - 1)
else:
    WINDOW = min(16, T - 1)
print(f"Using window={WINDOW} for T={T}")

# Model's attention
model_attn = outputs.attentions[0]  # [bsz, num_heads, q_len, kv_len]
print(f"Model attention shape: {model_attn.shape}")

# My CakeScorer computation
q = qk_outputs['layer_0']['q'].squeeze(0)  # [T, num_heads * head_dim]
k = qk_outputs['layer_0']['k'].squeeze(0)  # [T, num_kv_heads * head_dim]

# My attention computation (same as CakeScorer)
q_reshaped = q.reshape(T, num_kv_heads, num_q_per_kv, head_dim)
k_reshaped = k.reshape(T, num_kv_heads, head_dim)
q_obs = q_reshaped[T - WINDOW:].permute(1, 2, 0, 3)  # [num_kv, num_q, window, head]
k_t = k_reshaped.permute(1, 2, 0)  # [num_kv, head, T]
my_attn = torch.matmul(q_obs, k_t.unsqueeze(1)) / scale

# Apply causal mask
q_indices = torch.arange(T - WINDOW, T, device=my_attn.device, dtype=torch.long)
k_indices = torch.arange(T, device=my_attn.device, dtype=torch.long)
causal_mask = (q_indices[:, None] >= k_indices[None, :])
my_attn = my_attn.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
my_attn_soft = F.softmax(my_attn, dim=-1, dtype=torch.float32)

# Model's attention for the last WINDOW queries
# Model attention: [1, num_heads, T, T]
model_last = model_attn[0, :, -WINDOW:, :]  # [num_heads, window, T]

# Reshape model's attention to [num_kv, num_q, window, T]
model_reshaped = model_last.reshape(num_kv_heads, num_q_per_kv, WINDOW, T)

# Compare a single head, single GQA query, first query in window
h = 0
g = 0
my_slice = my_attn_soft[h, g, 0, :10]  # first 10 key positions
model_slice = model_reshaped[h, g, 0, :10]

print(f"\nMy attention (head=0, gqa=0, query=0, first 10 keys):")
print(f"  {my_slice.cpu().numpy()}")
print(f"\nModel attention (head=0, gqa=0, query=0, first 10 keys):")
model_slice_f32 = model_slice.float()
print(f"  {model_slice_f32.cpu().numpy()}")
print(f"\nMax diff: {(my_attn_soft - model_reshaped).abs().max().item()}")
print(f"Mean diff: {(my_attn_soft - model_reshaped).abs().mean().item()}")

# Compare the full attention for the last window
corr = np.corrcoef(
    my_attn_soft.flatten().cpu().numpy(),
    model_reshaped.flatten().cpu().numpy()
)[0, 1]
print(f"\nPearson correlation: {corr:.6f}")

# Now compute preference using BOTH methods
# My method
hist_len = T - WINDOW
my_hist = my_attn_soft[:, :, :, :hist_len]
my_entropy = -torch.sum(my_hist * torch.log(my_hist + 1e-10))
my_var = torch.var(my_hist, dim=-2).sum()
my_pref = (my_entropy ** (1/1.6) * my_var ** (1/0.4)).item()

# Model method
model_hist = model_reshaped[:, :, :, :hist_len]
model_entropy = -torch.sum(model_hist * torch.log(model_hist + 1e-10))
model_var = torch.var(model_hist, dim=-2).sum()
model_pref = (model_entropy ** (1/1.6) * model_var ** (1/0.4)).item()

print(f"\nMy preference: {my_pref:.6f}")
print(f"Model preference: {model_pref:.6f}")

# Compare all layers
print(f"\n--- All layers preference comparison ---")
my_prefs = []
model_prefs = []
for i in range(num_layers):
    # My computation
    q_i = qk_outputs[f'layer_{i}']['q'].squeeze(0)
    k_i = qk_outputs[f'layer_{i}']['k'].squeeze(0)
    
    q_r = q_i.reshape(T, num_kv_heads, num_q_per_kv, head_dim)
    k_r = k_i.reshape(T, num_kv_heads, head_dim)
    q_o = q_r[T - WINDOW:].permute(1, 2, 0, 3)
    k_t = k_r.permute(1, 2, 0)
    a = torch.matmul(q_o, k_t.unsqueeze(1)) / scale
    a = a.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
    a_s = F.softmax(a, dim=-1, dtype=torch.float32)
    
    h = a_s[:, :, :, :hist_len]
    e = -torch.sum(h * torch.log(h + 1e-10))
    v = torch.var(h, dim=-2).sum()
    my_prefs.append((e ** (1/1.6) * v ** (1/0.4)).item())
    
    # Model computation
    ma = outputs.attentions[i][0]  # [num_heads, T, T]
    ma_r = ma.reshape(num_kv_heads, num_q_per_kv, T, T)
    mh = ma_r[:, :, -WINDOW:, :hist_len]
    me = -torch.sum(mh * torch.log(mh + 1e-10))
    mv = torch.var(mh, dim=-2).sum()
    model_prefs.append((me ** (1/1.6) * mv ** (1/0.4)).item())

from scipy.stats import spearmanr
sr, sp = spearmanr(my_prefs, model_prefs)
print(f"Spearman ρ = {sr:.4f} (p = {sp:.2e})")
print(f"My prefs range: [{min(my_prefs):.4f}, {max(my_prefs):.4f}]")
print(f"Model prefs range: [{min(model_prefs):.4f}, {max(model_prefs):.4f}]")

# Save for analysis
results = {
    "my_prefs": my_prefs,
    "model_prefs": model_prefs,
    "spearman_r": float(sr),
    "max_diff": float((my_attn_soft - model_reshaped).abs().max().item()),
    "mean_diff": float((my_attn_soft - model_reshaped).abs().mean().item()),
    "pearson_corr": float(corr),
}
with open(os.path.join(OUTPUT_DIR, "debug_attention.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n[DONE]")