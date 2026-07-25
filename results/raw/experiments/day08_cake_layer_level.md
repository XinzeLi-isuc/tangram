# 实验报告：Day 8 — CakeLayerLevel 实现

## 实验目的
实现 CAKE 层预算分配器 `CakeLayerLevel`，使 `compression_level="cake_layer"` 可用。完成 CAKE-Serve 全管道（scorer + level）的 MVP 闭环。

## 修改的文件

### 1. `vllm/v1/attention/compression/selection_level.py`
- 新增 `SelectionContext` dataclass（携带 `layer_preferences: [num_layers]`）
- 新增 `CakeLayerLevel` 类，实现 CAKE 预算分配
- 注册 `compression_level="cake_layer"`
- 所有 `compute_counts` 方法增加 `context` 参数

### 2. `vllm/v1/attention/compression/compressor.py`
- 导入 `SelectionContext`
- `prepare_keep_decision` 收集 preferences 并传递给 `compute_counts`

## CakeLayerLevel 预算分配算法

```
1. Proportional: budget_per_layer = pref / sum(pref) * total_budget
2. Floor + remainder distribution
3. Cap at eval_len
4. Redistribute excess from capped layers
5. Final adjustment to match total_budget exactly
6. Spread per-layer budget across groups
```

## 复现步骤

### 单元测试
```bash
cd ~/cake-serve
python tests/cake_serve/test_cake_layer_level.py
```

### 端到端验证
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='...', compression_ratio=0.5,
          compression_scorer='cake', compression_level='cake_layer')
out = llm.generate(['What is KV cache?'], SamplingParams(temperature=0, max_tokens=64))
print(out[0].outputs[0].text)
"
```

## 结果

### 7 个单元测试全部通过

| 测试 | 验证内容 |
|------|---------|
| `test_registered` | `cake_layer` 在 registry 中且可构造 |
| `test_uniform_fallback` | 无 preferences 时回退到均匀分配 |
| `test_preferences_affect_budget` | 高偏好层获得更多预算 |
| `test_budget_sum_matches` | 不同 eval_len/ratio 下总和匹配 |
| `test_budget_within_eval_len` | 每层预算 ≤ eval_len |
| `test_monotonic` | 高 ratio 每层预算 ≥ 低 ratio |
| `test_extreme_preferences` | 极端偏好不崩溃 |

### 端到端验证
- `compression_scorer="cake"` + `compression_level="cake_layer"`
- ratio=1.0/0.5/0.25 均正常生成

## 产出文件
- 修改: `vllm/v1/attention/compression/selection_level.py`
- 修改: `vllm/v1/attention/compression/compressor.py`
- 测试: `tests/cake_serve/test_cake_layer_level.py`
- 日报: `results/processed/day08_report.md`