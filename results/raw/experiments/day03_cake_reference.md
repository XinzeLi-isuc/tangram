# 实验报告：Day 3 — CAKE 官方参考实现复现与数据导出

## 实验目的
分析 CAKE 官方代码结构，在 Llama-3.1-8B 上运行并导出层偏好、预算、token score、top-k index，作为后续 Tangram 集成的参考对照。

## 环境
- 同 Day 1
- 额外安装: `pip install -e ~/third_party/cakekv/` (CAKE 官方包，仅用于 utils 函数)

## 复现步骤

### 1. 分析 CAKE 代码结构
```bash
# CAKE 仓库位置
ls ~/third_party/cakekv/cake/
# 核心文件:
#   model/modify_llama.py  — 替换 attention forward，实现 CAKE 评分
#   cake_cache.py          — CakeCache + CakeprefillKVCache + CakeDecodingKVCache
#   utils.py               — calculate_entropy + adjust_budgets
#   monkeypatch.py          — 替换 transformers forward
```

### 2. 运行导出脚本
```bash
cd ~/cake-serve
CUDA_VISIBLE_DEVICES=0 \
/home/lixinze/miniconda3/envs/cake-serve/bin/python scripts/export_cake_reference.py 2>&1
```

### 3. 查看结果
```bash
ls -la results/raw/day03_cake_reference/
```

## 结果

### CAKE 算法公式

**层偏好**: `P_l = Entropy(A_l)^(1/τ1) * Var(A_l)^(1/τ2)`
- `Entropy(A) = -sum(p * log(p))` — attention 分布越分散，熵越高
- `Var(A) = var over query positions` — attention 随时间变化越明显，层更需要保留历史

**Token 分数**: `S = Mean_q(A) + γ * Var_q(A)` → avg_pool1d(kernel=5) → mean over GQA groups

**预算分配**: `B_l = P_l / ΣP_j * total_budget` → cap → redistribute

### CAKE 超参数 (Llama-3.1-8B, cache_size=1024)
| 参数 | 值 |
|------|------|
| tau1 | 1.6 |
| tau2 | 0.4 |
| gamma | 200.0 |
| window_size | 32 |
| cache_size | 1024 |

### 导出数据

| 文件 | 大小 | 内容 |
|------|------|------|
| `pref_scores.json` | 286 B | 32 层偏好分数 (范围 2.1~380.0) |
| `layer_budgets.json` | 245 B | 32 层预算 (总和 31744) |
| `token_scores.pt` | 538 KB | 每层 [8, 1032] token 分数 |
| `topk_indices.json` | 1.6 MB | 每层 top-k 索引 |
| `generation.txt` | 671 B | 模型输出文本 |
| `summary.json` | 880 B | 汇总信息 |

### 层偏好分布
- 最低: Layer 3 (2.1) — 注意力最集中
- 最高: Layer 8 (380.0) — 注意力最分散/不稳定
- 高偏好层: 8, 15, 10 (需要更多预算)
- 低偏好层: 3, 1, 2 (可更激进压缩)

## 关键发现
- 因 flash-attn CUDA 13.0 不兼容，改为 eager 模式
- 短 prompt (<32 tokens) 无法计算偏好 (hist_len=0)
- 使用 1064 token 长 prompt 成功导出

## 产出文件
- 导出数据: `results/raw/day03_cake_reference/`
- 导出脚本: `scripts/export_cake_reference.py`
- 代码日志: `results/raw/day03_logs/code_changes.md`
- 日报: `results/processed/day03_report.md`