# Day 7: Layer Preference 传递与请求隔离

**日期**: 2026-07-24  
**状态**: ✅ 完成

---

## 今日完成

### 代码修改

| 文件 | 修改 |
|------|------|
| `vllm/v1/attention/compression/cake.py` | 新增 `CakeScoreOutput`，`forward()` 返回它，新增层偏好计算 |
| `vllm/v1/attention/compression/compressor.py` | `_LayerCompressState` 新增 `pending_preference`，`receive_score()` 接收偏好，`_make_qk_scorer()` 处理 `CakeScoreOutput`，新增 `_collect_preferences()` |

### 测试结果

| 测试 | 结果 |
|------|------|
| 5 个单元测试 | 全部通过 ✅ |
| 端到端集成 (ratio=1.0/0.5) | 正常生成 ✅ |

### 数据流

```
CakeScorer.forward() → CakeScoreOutput(token_scores, layer_preference)
  → _make_qk_scorer 检测 → 分别调 receive_score(tokens, preference=pref)
  → _LayerCompressState.pending_preference 存储
  → _collect_preferences() → [num_layers] tensor
```

---

## 明天计划（第8天）

实现 CakeLayerLevel:
1. 添加 `SelectionContext` dataclass
2. 实现 `CakeLayerLevel` 使用 CAKE 偏好分配预算
3. 注册 `compression_level="cake_layer"`
4. 单元测试

---

## 面试准备

### 面试官可能问
1. **CAKE 的层偏好具体怎么算的？** 为什么 entropy × variance 能反映层的重要性？
2. **请求隔离怎么实现的？** 多个请求共享同一个 scorer 实例，preference 会串吗？
3. **CakeScoreOutput 的设计为什么用 dataclass 而不是直接改 scorer 契约？**

### 回答要点
- 层偏好 = entropy(注意力分布) × variance(时序变化)。entropy 高说明注意力分散，需要更多 token；variance 高说明注意力随时间变化大，需要保留更多历史。
- 请求隔离的关键：preference 存在 `req_state[req_id].layer_states[layer_idx].pending_preference`，不是 scorer 的成员变量。
- 用 dataclass 而不是改基类契约，是因为改基类会影响 SnapKV、KeyDiff 等其他 scorer。`CakeScoreOutput` 是 CAKE 独有的扩展，`_make_qk_scorer` 里用 `hasattr` 做鸭子类型判断，不影响已有 scorer。

### 自建项目可能遇到的难题
1. **Tangram 的 scorer 契约理解**：`forward()` 返回 `[num_kv_heads, T]` tensor，但 CAKE 需要返回额外信息。解决方案是返回 `CakeScoreOutput` 并在 `_make_qk_scorer` 里做类型判断。
2. **chunked prefill 下的 preference 累积**：一个 chunk 可能被拆成多个 forward step，每个 step 产生一个 preference。需要用 token 数加权平均来聚合。