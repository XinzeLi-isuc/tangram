# Day 5: CakeScorer 实现与接入 Tangram

**日期**: 2026-07-24  
**工作时段**: 22:50 ~ 23:15  
**状态**: ✅ 完成

---

## 今日完成

### 1. 新增 `cake.py` — CakeScorer

**文件**: `vllm/v1/attention/compression/cake.py`

**设计**:
- 继承 `QKScorer`，遵循 SnapKV 相同的接口模式
- `consumes = "qk"`, `name = "cake"`
- 使用 CAKE 评分公式: `S = Mean_q(A) + gamma * Var_q(A)`
- 平滑: `avg_pool1d(kernel_size=cake_kernel_size)`
- 短 chunk (<1000 tokens) 自适应缩小 window 到 min(16, chunk_len)
- 暂不返回 layer preference（Day 7 添加）

**默认参数**: window_size=32, kernel_size=5, gamma=1.0

### 2. 修改 `scorer.py`

| 修改 | 说明 |
|------|------|
| 导入 CakeScorer | 新增 import |
| 注册到 `_QK_SCORERS` | 加入 registry |
| 添加 "cake" 分支到 `build_qk_scorer` | 构建方法 |
| 添加 `cake_window_size`, `cake_kernel_size`, `cake_gamma` 参数 | 支持 CAKE 超参数 |

### 3. 修改 `compressor.py`

- `set_qk_scorers()` 方法添加 `cake_window_size`, `cake_kernel_size`, `cake_gamma` 参数
- 传递到 `build_qk_scorer()` 调用

### 4. 测试验证

**4 个测试全部通过**:

| 测试 | ratio | prompt | 结果 |
|------|-------|--------|------|
| CakeScorer FullKV | 1.0 | 简短 | ✅ 输出合理 |
| CakeScorer 0.5 | 0.5 | 简短 | ✅ 输出合理 |
| CakeScorer 0.5 | 0.5 | 重复 prompt | ✅ 无 NaN/无 OOM |
| CakeScorer 0.25 | 0.25 | 简短 | ✅ 输出合理 |

**模型加载**: 14.99 GiB weights, 26.4 GiB KV cache, CUDA graphs PIECEWISE 模式
**Scorer 注册**: `('snapkv', 'cake', 'keydiff', 'streamingllm', 'tova', 'expected_attention')`

---

## 代码变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `vllm/v1/attention/compression/cake.py` | 新增 | CakeScorer 实现 (4.5 KB) |
| `vllm/v1/attention/compression/scorer.py` | 修改 | 注册 + 构建 + 参数 |
| `vllm/v1/attention/compression/compressor.py` | 修改 | set_qk_scorers 传递 CAKE 参数 |

---

## 明天计划（第6天）

验证 Cake token scorer:
1. 对比 SnapKV uniform vs CAKE uniform
2. 检查 score shape, NaN, top-k overlap
3. scorer 耗时分析

---

## 问题与风险

| 问题 | 状态 |
|------|------|
| transformers 降级问题 | ✅ 已修复（重新安装 4.57.6） |
| CakeScorer 注册成功 | ✅ |
| 所有测试通过 | ✅ |