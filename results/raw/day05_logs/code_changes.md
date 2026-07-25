# Day 5 代码变更日志

## 新增文件
| 文件 | 说明 |
|------|------|
| `vllm/v1/attention/compression/cake.py` | CakeScorer(QKScorer) 实现 |

## 修改文件
| 文件 | 修改内容 |
|------|---------|
| `vllm/v1/attention/compression/scorer.py` | 导入 CakeScorer，注册到 _QK_SCORERS，添加 "cake" 分支到 build_qk_scorer，添加 cake_window_size/cake_kernel_size/cake_gamma 参数 |
| `vllm/v1/attention/compression/compressor.py` | set_qk_scorers 添加 cake 参数并传递到 build_qk_scorer |

## CakeScorer 设计
- 遵循 SnapKV 模式（QKScorer 子类，consumes="qk"）
- 使用 CAKE 评分公式: Mean_q(A) + gamma * Var_q(A)
- 平滑: avg_pool1d(kernel_size=cake_kernel_size)
- 默认参数: window_size=32, kernel_size=5, gamma=1.0
- 短 chunk (<1000 tokens) 自适应缩小 window 到 min(16, chunk_len)
- 暂不返回 layer preference（Day 7 添加）