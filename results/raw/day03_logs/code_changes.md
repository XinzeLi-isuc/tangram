# Day 3 代码变更日志

## 1. transformers patches (CAKE 官方需要)
### 文件: `/home/lixinze/miniconda3/envs/cake-ref/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py`
- 行 1161: `logits = logits.float()` → `# logits = logits.float()`
- 原因: CAKE 使用 flash_attention_2 时，`logits.float()` 会产生类型冲突。CAKE 官方 install.sh 通过 sed 固定路径修改，此处手动 patch。

## 2. 新增脚本
### 文件: `~/cake-serve/scripts/export_cake_reference.py`
- 功能: 加载 Llama-3.1-8B-Instruct 并应用 CAKE 压缩，导出层偏好、预算、token score、top-k index
- CAKE 参数: cache_size=1024, window_size=32, tau1=1.6, tau2=0.4, gamma=200.0 (来自 model2tau.json)
- 输出: results/raw/day03_cake_reference/