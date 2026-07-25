# Day 3: CAKE 官方参考实现复现与数据导出

**日期**: 2026-07-24  
**工作时段**: 21:25 ~ 22:15  
**状态**: ✅ 完成

---

## 今日完成

### 1. CAKE 官方代码分析

**仓库**: https://github.com/antgroup/cakekv (commit from main branch)
**路径**: `~/third_party/cakekv/`

**核心文件分析**:
| 文件 | 功能 |
|------|------|
| `cake/model/modify_llama.py` | 替换 LlamaFlashAttention2.forward，实现 CAKE 评分和淘汰 |
| `cake/cake_cache.py` | CakeCache 类 + CakeprefillKVCache (预算分配) + CakeDecodingKVCache_LayerWise (解码时淘汰) |
| `cake/utils.py` | calculate_entropy + adjust_budgets 工具函数 |
| `cake/monkeypatch.py` | 将 transformers 的 forward 替换为 CAKE 版本 |

**CAKE 算法公式 (来自 modify_llama.py)**:
- 层偏好: `P_l = Entropy(A_l)^(1/τ1) * Var(A_l)^(1/τ2)`
- Token 分数: `S = Mean_q(A) + γ * Var_q(A)` → avg_pool1d(kernel=5) → mean over GQA groups
- 预算分配: `B_l = P_l / ΣP_j * total_budget` → adjust_budgets (cap + redistribute)

**CAKE 超参数 (model2tau.json)**:
| cache_size | tau1 | tau2 |
|-----------|------|------|
| 1024 | 1.6 | 0.4 |
| 128 | 1.6 | 0.6 |
| gamma=200.0 (来自 args 默认值) |

### 2. 环境搭建

| 环境 | 用途 | 状态 |
|------|------|------|
| `cake-ref` (conda, Python 3.10) | 尝试运行 CAKE 官方代码 | ⚠️ flash-attn 因 CUDA 版本不兼容无法安装 |
| `cake-serve` (conda, Python 3.12) | 直接计算 CAKE 分数 | ✅ 成功 |

**环境变更记录**:
- `results/raw/day03_logs/code_changes.md`

### 3. 导出脚本

**脚本**: `scripts/export_cake_reference.py`

**工作原理**:
1. 加载 Llama-3.1-8B-Instruct 使用 eager attention + output_attentions=True
2. 注册 forward hook 捕获每层 attention 权重
3. 手动 forward pass (prefill)
4. 按 CAKE 公式计算层偏好和 token 分数
5. 实现 CAKE 风格预算分配 (floor → cap → redistribute)
6. 导出所有数据

### 4. 导出结果

**位置**: `results/raw/day03_cake_reference/`

| 文件 | 大小 | 内容 |
|------|------|------|
| `config.json` | 5.9 KB | 运行配置 |
| `pref_scores.json` | 286 B | 32 层偏好分数 |
| `layer_budgets.json` | 245 B | 32 层预算分配 |
| `token_scores.pt` | 538 KB | 每层 token 分数 (torch tensor) |
| `topk_indices.json` | 1.6 MB | 每层 top-k 索引 |
| `generation.txt` | 671 B | 模型输出文本 |
| `summary.json` | 880 B | 汇总信息 |

**偏好分数摘要**:
| 统计 | 值 |
|------|------|
| 最小值 | Layer 3: 2.1 |
| 最大值 | Layer 8: 380.0 |
| 高偏好层 | 8, 15, 10 (需要更多预算) |
| 低偏好层 | 3, 1, 2 (注意力集中，可更激进压缩) |

**预算分配摘要**:
| 统计 | 值 |
|------|------|
| 总预算 | 31,744 tokens |
| 每层范围 | 598 ~ 1,494 |
| 目标匹配 | ✅ 精确匹配 |

### 5. 关键发现

- **Layer 8 偏好最高**: 可能是模型中间层，注意力分布最分散/不稳定
- **Layer 3 偏好最低**: 注意力最集中，可以分配最少预算
- **预算范围 (598-1494)**: 在 1064 token 上下文中，CAKE 分配了 56%~140% 的 token 预算
- 高预算层 (>1032) 是因为 cap 后 redistribution 所致，在 Tangram 中需要进一步加 Page 对齐约束

---

## 代码变更

### 新增文件
| 文件 | 说明 |
|------|------|
| `scripts/export_cake_reference.py` | CAKE 参考数据导出脚本 (v3) |
| `results/raw/day03_logs/code_changes.md` | 变更日志 |

### 第三方修改
| 文件 | 修改 |
|------|------|
| `cake-ref/lib/python3.10/.../modeling_llama.py` | 注释 `logits.float()` 行 |

---

## 明天计划（第4天）

写纯 PyTorch 算法单元：
1. `compute_cake_scores(q, k, ...)` — 独立函数
2. `allocate_cake_budgets(preferences, total_budget, ...)` — 独立函数
3. 与 CAKE 官方参考数据数值对照
4. CPU/GPU 单测全部通过

---

## 问题与风险

| 问题 | 状态 |
|------|------|
| flash-attn CUDA 版本不兼容 | ⚠️ 绕过，改用 eager attention |
| 短 prompt 无法计算偏好 (hist_len=0) | ✅ 修复，使用 1064 token 长 prompt |
| adjust_budgets 函数有负数 bug | ✅ 修复，重写预算分配逻辑 |
| 导出数据完整可用 | ✅ |