# Day 8: CakeLayerLevel 实现 — CAKE-Serve MVP 闭环

**日期**: 2026-07-24  
**状态**: ✅ 完成

---

## 今日完成

### 代码修改

| 文件 | 修改 |
|------|------|
| `selection_level.py` | 新增 `SelectionContext`、`CakeLayerLevel`、注册 `cake_layer` |
| `compressor.py` | 导入 `SelectionContext`，`prepare_keep_decision` 传递 preferences |

### 测试结果

| 测试 | 结果 |
|------|------|
| 7 个单元测试 | 全部通过 ✅ |
| 端到端 (scorer=cake + level=cake_layer, r=1.0/0.5/0.25) | 正常生成 ✅ |

### 数据流

```
CakeScorer.forward() → CakeScoreOutput
  → receive_score(..., layer_preference=pref)
  → _collect_preferences() → [num_layers] tensor
  → SelectionContext(layer_preferences=prefs)
  → CakeLayerLevel.compute_counts(..., context=ctx)
  → per-(layer, group) kept COUNT
```

### 面试准备

**面试官可能问：**
1. CakeLayerLevel 和已有的 crosslayer_cluster 有什么区别？
2. 为什么预算分配需要 SelectionContext 而不是直接读取 compressor 状态？
3. 32 层的偏好如何分配到 num_groups 个 group？

**回答要点：**
- crosslayer_cluster 用全局 threshold 选 token，所有层共享一个阈值。CakeLayerLevel 用 CAKE 偏好给每层分配固定预算，高偏好层保留更多 token。
- SelectionContext 是显式依赖注入，比隐式读取 compressor 状态更可测试、更清晰。只需传入 preferences 即可。
- 每层总预算按 preference 分配，然后在层内均匀分摊到各个 group。更精细的策略（如 per-group 偏好）可以后续扩展。

**自建项目可能遇到的难题：**
1. `compute_counts` 签名修改会影响所有 selection level。需要加 `context` 参数并给默认值 None。
2. 预算从 layer 级别分摊到 group 级别时，整除问题会导致微小误差。用 floor + remainder 处理。
3. `_collect_preferences` 返回 None 时（非 CAKE scorer），需要优雅降级到 uniform。

---

## 明天计划（第9天）

完整 CAKE-Serve MVP：
1. 与 CAKE 官方对照验收
2. 验收标准：preference Spearman ≥0.85, token top-k overlap ≥0.80
3. 运行 RULER 8K 测试