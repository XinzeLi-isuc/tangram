# CAKE-Serve 面试准备（累积）

## 项目整体

### 面试官可能问
1. **这个项目解决了什么问题？**
2. **CAKE 和 Tangram 各自做了什么？为什么需要把两者结合起来？**
3. **你的真实贡献是什么？不是"提出了 CAKE"吧？**

### 回答要点
- 问题：长上下文 LLM 服务中，KV Cache 占用成为并发瓶颈。CAKE 在 Transformers 上验证了层偏好 + 时序淘汰有效，但不能直接用于 vLLM 在线服务。Tangram 能把非均匀压缩转化为物理显存收益，但缺少 CAKE 的层偏好模型。
- 真实贡献：复现并分析 CAKE 算法 → 将 CAKE 从 Transformers tensor cache 移植到 Tangram/vLLM Ragged Paging → 扩展 Tangram scorer 接口携带 layer preference → 实现面向 Page Group 的 CAKE 层预算分配器 → 处理 chunked prefill 下的偏好聚合。
- 不能说自己"提出了 CAKE"。

### 自建项目可能遇到的难题
1. **环境依赖冲突**：CAKE 官方要求 transformers==4.43.3，但 Tangram 要求 transformers>=4.56.0。两个不能共存于一个环境。需要分开两个 conda env。
2. **flash-attn 编译失败**：CAKE 官方需要 flash_attention_2，但 CUDA 13.0 与 flash-attn 不兼容。解决方案：改用 eager 模式 + output_attentions=True 手动捕获 attention。

---

## Day 1-2: 环境搭建 + RULER Baseline

### 面试官可能问
1. **为什么选择 SnapKV + crosslayer_cluster 作为 baseline？**
2. **RULER 是什么？为什么用它评估？**
3. **A6000 48GB 上 8K 上下文能同时跑多少个请求？**

### 回答要点
- SnapKV 是 Tangram 官方验证的 scorer，crosslayer_cluster 是 Tangram 最强的 budget level（跨层非均匀分配）。用它们做 baseline 可以隔离"CAKE 的 token scorer"和"CAKE 的 layer budget"两个维度的贡献。
- RULER 是 13 个合成长上下文任务的套件，测试 retrieval、tracking、QA 等能力。比 LongBench 更可控（精确的上下文长度、gold answer）。
- A6000 48GB 上，Llama-3.1-8B 的 KV Cache 每 token 约 128KB。8K 上下文约 1GB，26.4GB KV cache 可用 → 约 26 个并发请求。

### 自建项目可能遇到的难题
1. **Tangram 安装编译**：vLLM 从源码编译需要 8-10 分钟，容易超时。用 `VLLM_USE_PRECOMPILED=1` 可以加速。CUDA graph 捕获需要 5-6 秒。
2. **RULER 数据集依赖**：需要 `datasets`、`pyarrow`、`cloudpickle` 等库。安装 `datasets` 会升级 numpy 到 2.5，导致 numba 报错。需要固定 numpy<2.3。
3. **model2tau.json 路径**：CAKE 的配置文件在 `experiments/LongBench/config/` 下，不是代码里 hardcode 的。需要自己找到它。

---

## Day 3: CAKE 官方复现

### 面试官可能问
1. **CAKE 的层偏好公式为什么有效？**
2. **CAKE 和 SnapKV 的 token 评分有什么本质区别？**
3. **CAKE 的 adjust_budgets 函数有什么边界问题？**

### 回答要点
- 层偏好 = entropy^(1/τ1) × variance^(1/τ2)。entropy 衡量注意力分布的分散程度——注意力越分散的层需要保留更多 token。variance 衡量注意力随时间的变化程度——变化越大的层，历史 token 的重要性越不稳定，需要更多预算。
- SnapKV 用 amax（取 GQA 组最大值），CAKE 用 mean + gamma * variance。SnapKV 认为"只要有一个 head 关注就重要"，CAKE 考虑"所有 head 的均值 + 时序波动"。
- adjust_budgets 在预算极小时会产生负数（dtype=int 的截断问题）。需要自己实现更健壮的预算分配器。

### 自建项目可能遇到的难题
1. **flash-attn 与 CUDA 13.0 不兼容**：CAKE 官方依赖 flash_attention_2，但 torch 2.13.0+cu130 没有预编译的 flash-attn wheel。需要手动编译，但编译又需要 nvcc。解决方案：改用 eager 模式 + output_attentions=True 手动捕获 attention 分数。
2. **CAKE 官方代码的 install.sh 不能用**：它用固定路径 `/opt/conda/lib/python3.10/` 的 sed 修改 transformers，不适用于 conda 环境。需要手动 patch。
3. **短 prompt 无法计算偏好**：CAKE 需要 `seq_len > window_size` 才能计算偏好。短 prompt（<32 tokens）下 hist_len=0，偏好为 0。

---

## Day 4-5: CAKE 算法单元 + CakeScorer 实现

### 面试官可能问
1. **CakeScorer 为什么是 stateless 的？所有层共享一个实例？**
2. **CAKE 的 gamma 参数在论文里是 200.0，为什么第一版用 1.0？**
3. **怎么保证 CakeScorer 和 Tangram 的其他 scorer 兼容？**

### 回答要点
- Stateless 设计是为了性能——所有层共享一个 scorer 实例，不需要为每层创建独立对象。状态存在 compressor 的 req_state 里。
- gamma=200.0 是 CAKE 论文在 LongBench 上调参得到的。第一版用 gamma=1.0 是先确认通路正确，第 15 天会做参数消融。面试时可以说"这是 staged approach——先验证功能，再优化参数"。
- CakeScorer 继承 QKScorer 基类，实现 forward() 契约。注册到 _QK_SCORERS 字典，scorer 工厂自动发现。不需要修改其他 scorer 的代码。

### 自建项目可能遇到的难题
1. **独立函数与 Tangram 的接口差异**：独立函数 `compute_cake_scores()` 返回 `(scores, pref)` tuple，但 Tangram 的 QKScorer.forward() 只返回一个 tensor。解决方案：用 `CakeScoreOutput` dataclass 包装，在 `_make_qk_scorer` 里做类型判断。
2. **预算分配与 CAKE 官方的差异**：`allocate_cake_budgets()` 的 cap + redistribute 逻辑与 CAKE 官方的 `adjust_budgets()` 不完全一致。需要仔细对比 Spearman 相关系数确保排名一致。
3. **测试需要 GPU**：`compute_cake_scores` 的数值验证需要 GPU 运行模型 forward 获取 attention。纯 CPU 测试只能验证预算分配逻辑。

---

## Day 6: SnapKV vs CAKE 对比

### 面试官可能问
1. **CAKE 比 SnapKV 慢 1.69x，这个开销可以接受吗？**
2. **Top-k 重叠只有 67-70%，为什么差异这么大？**
3. **如果 CAKE 不比 SnapKV 好，这个项目还有意义吗？**

### 回答要点
- 0.43ms 的 scorer 开销在 prefill 阶段占比很小（prefill 通常几百毫秒到几秒）。而且 CAKE 的额外开销换来了层偏好信息，这是 SnapKV 完全不具备的能力。
- 差异来自评分策略不同：SnapKV 用 amax（GQA 组最大值），CAKE 用 mean + variance。这会选出不同的 top-k token。这不是 bug，是 CAKE 的设计意图——它保留了那些"重要性随时间波动"的 token。
- 项目的核心价值不是"CAKE 比 SnapKV 好"，而是"CAKE 的层偏好 + Tangram 的物理 Page 回收 = 可量化的系统收益"。即使 token scorer 不比 SnapKV 好，层预算分配也可能带来并发提升。

### 自建项目可能遇到的难题
1. **合成数据 vs 真实数据**：对比测试用合成 Q/K 数据，无法完全反映真实 prompt 下的评分差异。需要配合 RULER/LongBench 做质量评估。
2. **单层 vs 多层偏好**：对比测试只计算单层偏好，但 CAKE 的预算分配需要跨 32 层的偏好比较。单层偏好的绝对值在不同输入间变化很大，但排名应该一致。
3. **Timing 测量噪声**：GPU 上的 timing 测量受 CUDA kernel launch、warmup 等影响。需要多次测量取平均。

---

## Day 7: Layer Preference 传递

### 面试官可能问
1. **请求隔离怎么做的？为什么不能把 preference 存在 scorer 里？**
2. **CakeScoreOutput 的设计为什么用 dataclass 而不是改基类？**
3. **chunked prefill 下 preference 怎么累积？**

### 回答要点
- Scorer 是 stateless 的，所有请求共享。如果把 preference 存在 scorer 里，请求 A 的 preference 会被请求 B 覆盖。解决方案：preference 存在 compressor 的 `req_state[req_id].layer_states[layer_idx].pending_preference`。
- 改基类 QKScorer 的 forward() 返回值会影响 SnapKV、KeyDiff 等所有 scorer。用 CakeScoreOutput dataclass + hasattr 鸭子类型判断，只影响 CAKE。
- 用 token 数加权平均：`P_avg = (P_prev * n_prev + P_curr * n_curr) / (n_prev + n_curr)`。

### 自建项目可能遇到的难题
1. **compressor.py 的代码复杂度**：`compressor.py` 近 900 行，涉及 `_make_qk_scorer`、`receive_score`、`_collect_layer_tensors`、`prepare_keep_decision` 等多个方法。需要仔细理解每个方法的调用链和数据流。
2. **CakeScoreOutput 与已有 scorer 的兼容性**：`_make_qk_scorer` 里用 `hasattr` 判断 CakeScoreOutput，但如果其他 scorer 也返回了同名属性，会误判。需要确保 `token_scores` 和 `layer_preference` 属性名不会与其他 scorer 冲突。
3. **preference 的清理时机**：`_collect_layer_tensors` 消费 pending_score 后需要同时清理 pending_preference，否则会累积到下一个 chunk 造成错误。