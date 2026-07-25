# Day 11 代码变更日志

## 新增文件
- `scripts/phase1_paper_equivalence.py` — Paper-equivalence 对照脚本
- `scripts/phase2_chunked_prefill.py` — Chunked prefill 模拟验证脚本
- `scripts/debug_attention.py` — 注意力值对比调试脚本
- `scripts/test_chunked_integration.py` — 端到端 chunked prefill 集成测试

## 修改文件

### 1. `vllm/config/cache.py`
- 新增 `compression_cake_tau1: float = 1.0` — CAKE tau1 配置项
- 新增 `compression_cake_tau2: float = 1.0` — CAKE tau2 配置项
- 添加到 `compute_hash` 的 `ignored_factors` 列表

### 2. `vllm/v1/attention/compression/cake.py`
- 修复：添加 causal mask 到 attention 计算
- 原因：CAKE 官方参考使用 causal attention（每个 query 只能 attend 到其位置之前的 key），
  CakeScorer 之前未使用 causal mask，导致 softmax 归一化范围错误
- 影响：修复后 preference Spearman 从 0.73 提升到 0.76（未用 RoPE 前），
  最终结合 RoPE 修复达到 0.999

### 3. `vllm/v1/attention/compression/compressor.py`
- `set_qk_scorers()` 新增 `cake_tau1`、`cake_tau2` 参数
- 传递给 `build_qk_scorer()`

### 4. `vllm/v1/attention/compression/scorer.py`
- `build_qk_scorer()` 新增 `cake_tau1`、`cake_tau2` 参数
- 传递给 `CakeScorer()` 构造函数

### 5. `vllm/v1/worker/compression_model_runner_mixin.py`
- `set_qk_scorers()` 调用新增 `cake_tau1`、`cake_tau2` 参数传递

### 6. `vllm/engine/arg_utils.py`
- `EngineArgs` 新增 `compression_cake_tau1`、`compression_cake_tau2` 字段
- 缓存配置创建时传递 tau 参数
- CLI 新增 `--compression-cake-tau1`、`--compression-cake-tau2` 参数

## 关键发现
1. **RoPE 位置编码**：CakeScorer 接收的是 post-RoPE Q/K，但 standalone 脚本中直接从 
   q_proj/k_proj 捕获的是 pre-RoPE Q/K。需要在捕获后手动应用 RoPE 才能匹配模型注意力。
2. **Causal Mask**：CAKE 官方参考使用 causal attention，CakeScorer 需要同样的 causal mask。
3. **Chunked Prefill 收敛**：CAKE-Chunk 在 chunk 较小时 Spearman ≈ 0.6，这是预期行为——
   每个 chunk 的 observation window 不在 prompt 末尾，导致注意力模式不同。