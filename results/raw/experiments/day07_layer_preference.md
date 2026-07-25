# 实验报告：Day 7 — Layer Preference 传递与请求隔离

## 实验目的
将 CAKE 层偏好分数从 scorer 传递到 compressor 的请求状态中，实现 per-request 隔离存储，为 Day 8 的 CakeLayerLevel 预算分配做准备。

## 修改内容

### 1. `cake.py` — CakeScoreOutput 和偏好计算

```python
@dataclass
class CakeScoreOutput:
    token_scores: torch.Tensor       # [num_kv_heads, T]
    layer_preference: torch.Tensor   # scalar fp32
```

- `CakeScorer.forward()` 现在返回 `CakeScoreOutput` 而非 `torch.Tensor`
- 新增层偏好计算: `P_l = Entropy(A_hist)^(1/τ1) * TemporalVar(A_hist)^(1/τ2)`
- 新增参数: `cake_tau1=1.0`, `cake_tau2=1.0`, `cake_eps=1e-10`

### 2. `compressor.py` — 偏好存储和传递

| 修改 | 说明 |
|------|------|
| `_LayerCompressState.pending_preference` | 新增字段，存储 per-layer 偏好 |
| `receive_score(layer_preference=...)` | 新增可选参数，接收并存储偏好 |
| `_make_qk_scorer()` | 检测 `CakeScoreOutput`，提取 token_scores + layer_preference |
| `_collect_layer_tensors()` | 消费偏好（重置为 None） |
| `_collect_preferences()` | 新增方法，收集所有层的偏好为 `[num_layers]` tensor |

### 请求隔离设计
- 偏好存储在 `req_state[req_id].layer_states[layer_idx].pending_preference`
- 每个请求独立，多请求共享 scorer 实例但不会串数据
- chunked prefill 下使用 token 数加权平均聚合偏好

## 复现步骤

### 1. 单元测试
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python tests/cake_serve/test_layer_preference.py
```

### 2. 端到端验证
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='...', compression_ratio=0.5, compression_scorer='cake', compression_level='uniform')
out = llm.generate(['What is KV cache?'], SamplingParams(temperature=0, max_tokens=64))
print(out[0].outputs[0].text)
"
```

## 结果

### 单元测试 (5/5 通过)

| 测试 | 验证内容 |
|------|---------|
| `test_scorer_returns_cake_output` | `forward()` 返回 `CakeScoreOutput` 类型 |
| `test_preference_deterministic` | 相同输入产生相同偏好 |
| `test_different_inputs_different_preferences` | 不同输入产生不同偏好 |
| `test_short_input_no_preference` | 短输入 (< window) 偏好=0 |
| `test_preference_positive` | 偏好始终非负 |

### 偏好值示例
- 100 token 随机输入: 偏好 ~741.89
- 200 token 随机输入: 偏好 ~526.41
- 偏好范围: 随输入内容和长度变化

### 端到端验证
- compression_scorer="cake" 在 ratio=1.0/0.5 下均正常生成
- 无错误，无 NaN

## 产出文件
- 修改: `vllm/v1/attention/compression/cake.py`
- 修改: `vllm/v1/attention/compression/compressor.py`
- 测试: `tests/cake_serve/test_layer_preference.py`
- 日报: `results/processed/day07_report.md`