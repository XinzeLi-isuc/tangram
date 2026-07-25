# 实验报告：Day 5 — CakeScorer 实现与接入 Tangram

## 实验目的
将 CAKE 评分算法实现为 Tangram QKScorer 并注册到 scorer 工厂，使 `compression_scorer="cake"` 可在 Tangram 中正常使用。

## 环境
- 同 Day 1

## 复现步骤

### 1. 代码修改

**新增文件**: `vllm/v1/attention/compression/cake.py`
```python
class CakeScorer(QKScorer):
    name = "cake"
    consumes = "qk"
    # 参数: cake_window_size=32, cake_kernel_size=5, cake_gamma=1.0
    # 评分: S = Mean_q(A) + gamma * Var_q(A) → avg_pool1d
```

**修改文件**:
- `vllm/v1/attention/compression/scorer.py`: 导入 + 注册 + 构建方法
- `vllm/v1/attention/compression/compressor.py`: set_qk_scorers 传递 CAKE 参数

### 2. 验证注册
```bash
cd ~/cake-serve
python -c "
from vllm.v1.attention.compression.scorer import QK_SCORERS
print(QK_SCORERS)
assert 'cake' in QK_SCORERS
"
# 预期输出: ('snapkv', 'cake', 'keydiff', 'streamingllm', 'tova', 'expected_attention')
```

### 3. 运行生成测试
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 python scripts/test_cake_scorer.py
```

## 结果

### 生成测试 (所有比例通过)

| 配置 | ratio | prompt | 输出 | 状态 |
|------|-------|--------|------|------|
| CakeScorer FullKV | 1.0 | 简短 | 合理文本 | ✅ |
| CakeScorer compression | 0.5 | 简短 | 合理文本 | ✅ |
| CakeScorer compression | 0.5 | 重复 1K | 生成正常 | ✅ |
| CakeScorer compression | 0.25 | 简短 | 合理文本 | ✅ |

### 详细验证
```text
ratio=1.00: "A brief explanation of KV cache compression in LLMs is that it's a technique..."
ratio=0.50: "A brief explanation of KV cache compression in LLMs is that it's a technique..."
ratio=0.25: "A brief explanation of KV cache compression in LLMs is that it's a technique..."
```

### 模型加载信息
- 权重: 14.99 GiB
- KV Cache: 26.44 GiB (216,624 tokens)
- 最大并发 (8K): 26.44x
- CUDA Graph: PIECEWISE 模式, 51 graphs, ~5s

## 代码变更汇总

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `vllm/v1/attention/compression/cake.py` | 新增 | CakeScorer 实现 (4.5 KB) |
| `vllm/v1/attention/compression/scorer.py` | 修改 | 导入 + 注册 + 参数 + 构建 |
| `vllm/v1/attention/compression/compressor.py` | 修改 | set_qk_scorers 传递 CAKE 参数 |

## 产出文件
- CakeScorer: `vllm/v1/attention/compression/cake.py`
- 测试脚本: `scripts/test_cake_scorer.py`
- 代码日志: `results/raw/day05_logs/code_changes.md`
- 日报: `results/processed/day05_report.md`