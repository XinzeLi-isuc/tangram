# Day 4: 纯 PyTorch CAKE 算法单元

**日期**: 2026-07-24  
**工作时段**: 22:15 ~ 22:50  
**状态**: ✅ 完成

---

## 今日完成

### 1. 独立函数 `cake_algorithm.py`

**文件**: `scripts/cake_algorithm.py`

**`compute_cake_scores(query, key, ...)`** — CAKE token scorer
- 输入: post-RoPE Q/K, GQA 参数, CAKE 超参数
- 输出: `token_scores [num_kv_heads, T]` + `layer_pref scalar`
- 实现:
  1. Reshape GQA Q/K → [T, num_kv_heads, num_q_per_kv, head_dim]
  2. 取最后 window_size 个 query 计算 attention
  3. **层偏好**: softmax → entropy^(1/τ1) * var^(1/τ2)
  4. **Token 分数**: mean + γ * var → avg_pool1d(kernel=5) → mean over GQA groups

**`allocate_cake_budgets(pref_scores, total_budget, eval_len, ...)`** — CAKE budget allocator
- 实现: proportional → floor → remainder → cap → redistribute → final adjust
- 可选 block_size 对齐（为 Tangram 集成准备）

### 2. 单元测试 `test_cake_algorithm.py`

**文件**: `tests/cake_serve/test_cake_algorithm.py`

**11 个测试全部通过**:

| 测试 | 验证内容 |
|------|---------|
| test_imports | 函数正确导入 |
| test_allocate_budgets_sum_matches | 不同层数和预算下总和精确匹配 |
| test_allocate_budgets_within_eval_len | 每层预算不超过 eval_len |
| test_allocate_budgets_monotonic | 高 ratio → 每层预算 ≥ 低 ratio |
| test_allocate_budgets_block_alignment | block_size 对齐正确 |
| test_allocate_budgets_uniform_fallback | 全零偏好时均匀分配 |
| test_compute_cake_scores_no_nan | 极短输入、零值、大值无 NaN/Inf |
| test_compute_cake_scores_pref_positive | 偏好分数非负 |
| test_budget_against_reference | 与 CAKE 官方 adjust_budgets Spearman=0.83 |
| test_token_scores_against_reference | token score shape 匹配 [8, 1032] |
| test_against_reference | 模型 forward + score 计算通路验证 |

### 3. 与 CAKE 官方对比

| 指标 | 值 |
|------|------|
| 预算 Spearman ρ | **0.8347** (p=0.0000) |
| 我们的预算范围 | [909, 1032] |
| CAKE 官方预算范围 | [951, 1045] |
| 差异原因 | 官方函数不严格 cap 于 eval_len |

---

## 代码变更

| 文件 | 说明 |
|------|------|
| `scripts/cake_algorithm.py` | 新增: 纯 PyTorch CAKE 算法实现 |
| `tests/cake_serve/test_cake_algorithm.py` | 新增: 11 个单元测试 |

---

## 明天计划（第5天）

实现 CakeScorer 并接入 Tangram:
1. 新建 `vllm/v1/attention/compression/cake.py`
2. 注册 `compression_scorer="cake"`
3. 临时只返回 token score，不连接 layer preference
4. 验证 batch=1 下能正常生成

---

## 问题与风险

| 问题 | 状态 |
|------|------|
| 与 CAKE 官方预算分配 Spearman 0.83 | ✅ 可接受（差异来自 cap 策略不同） |
| 所有边界条件测试通过 | ✅ |
| 算法单元可在 Tangram 中复用 | ✅ |