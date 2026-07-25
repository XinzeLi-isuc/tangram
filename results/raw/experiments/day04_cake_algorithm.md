# 实验报告：Day 4 — 纯 PyTorch CAKE 算法单元

## 实验目的
编写独立于 vLLM 的纯 PyTorch CAKE 算法函数，与 Day 3 参考数据做数值对照，为 Tangram 集成做准备。

## 环境
- 同 Day 1

## 复现步骤

### 1. 运行单元测试
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 \
/home/lixinze/miniconda3/envs/cake-serve/bin/python tests/cake_serve/test_cake_algorithm.py
```

## 结果

### 算法函数

**`compute_cake_scores(query, key, ...)`**
```
输入:  query [T, num_kv_heads * num_q_per_kv * head_size]
       key   [T, num_kv_heads * head_size]
输出:  token_scores [num_kv_heads, T], layer_pref scalar
算法:
  1. Reshape GQA → [T, num_kv, num_q, d]
  2. 取最后 window_size 个 query 计算 attention
  3. 层偏好: softmax → entropy^(1/τ1) * var^(1/τ2)
  4. Token 分数: mean + γ * var → avg_pool1d(kernel=5) → mean over GQA groups
```

**`allocate_cake_budgets(pref_scores, total_budget, eval_len, ...)`**
```
输入:  pref_scores [num_layers], total_budget, eval_len
输出:  budgets [num_layers] int64
算法:  proportional → floor → remainder → cap → redistribute → final adjust
```

### 11 个单元测试全部通过

| 测试 | 验证内容 |
|------|---------|
| `test_imports` | 函数正确导入 |
| `test_allocate_budgets_sum_matches` | 不同层数/预算下总和精确匹配 |
| `test_allocate_budgets_within_eval_len` | 每层预算 ≤ eval_len |
| `test_allocate_budgets_monotonic` | 高 ratio → 每层预算 ≥ 低 ratio |
| `test_allocate_budgets_block_alignment` | block_size 对齐正确 |
| `test_allocate_budgets_uniform_fallback` | 全零偏好时均匀分配 |
| `test_compute_cake_scores_no_nan` | 极短/零值/大值输入无 NaN |
| `test_compute_cake_scores_pref_positive` | 偏好分数非负 |
| `test_budget_against_reference` | 与 CAKE 官方 adjust_budgets 对比 |
| `test_token_scores_against_reference` | token score shape 匹配 [8, 1032] |
| `test_against_reference` | 模型 forward + score 计算通路验证 |

### 与 CAKE 官方对比

| 指标 | 值 |
|------|------|
| 预算 Spearman ρ | **0.8347** (p=0.0000) |
| 我们的预算范围 | [909, 1032] |
| CAKE 官方预算范围 | [951, 1045] |
| 差异原因 | 官方函数不严格 cap 于 eval_len |

## 产出文件
- 算法实现: `scripts/cake_algorithm.py`
- 单元测试: `tests/cake_serve/test_cake_algorithm.py`
- 日报: `results/processed/day04_report.md`