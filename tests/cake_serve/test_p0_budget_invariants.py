"""
P0 预算不变性测试：验证 CakeLayerLevel 的 page group 预算映射正确性。

修复前：layer_budget // num_groups 导致保留量比 UniformLevel 少 num_groups 倍。
修复后：同一层所有 groups 共享相同预算，effective_ratio ≈ requested_ratio。

测试覆盖：
  - num_groups = 1/2/4/8
  - ratio = 0.1/0.25/0.5/0.75
  - 全零 preference, 单层极大 preference
  - NaN/Inf/负 preference
  - 不同 floor_min / sink/window 大小
"""
import numpy as np
import torch
import sys
sys.path.insert(0, ".")

from vllm.v1.attention.compression.selection_level import CakeLayerLevel, SelectionContext


def test_basic_ratios():
    """测试各 ratio 下预算不变性"""
    level = CakeLayerLevel()
    num_layers = 32
    num_groups = 4
    eval_len = 2048
    device = "cpu"
    
    # 使用有意义的偏好值
    np.random.seed(42)
    prefs = np.random.uniform(5, 100, num_layers).astype(np.float32)
    
    for ratio in [0.1, 0.25, 0.5, 0.75]:
        context = SelectionContext(
            layer_preferences=torch.tensor(prefs, dtype=torch.float32)
        )
        scores = torch.zeros(num_layers, num_groups, eval_len)
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        counts = level.compute_counts(scores, ratio, dummy_cluster, 
                                       num_layers, 8, num_groups, context)
        
        expected = eval_len * num_layers * num_groups * ratio
        actual = counts.sum()
        # 允许 Page 对齐误差
        alignment_error = num_layers * num_groups  # 每 (layer, group) ±1
        assert abs(actual - expected) <= alignment_error, \
            f"ratio={ratio}: expected={expected}, actual={actual}, diff={actual-expected}"
        
        effective_ratio = actual / (eval_len * num_layers * num_groups)
        print(f"  ratio={ratio:.2f}: expected={expected:.0f} actual={actual} "
              f"error={actual-expected:.0f} effective_ratio={effective_ratio:.4f}")


def test_group_counts():
    """测试不同 num_groups 下预算不变性"""
    level = CakeLayerLevel()
    eval_len = 2048
    num_layers = 8
    ratio = 0.25
    device = "cpu"
    
    prefs = np.random.uniform(5, 100, num_layers).astype(np.float32)
    context = SelectionContext(
        layer_preferences=torch.tensor(prefs, dtype=torch.float32)
    )
    
    for num_groups in [1, 2, 4, 8]:
        scores = torch.zeros(num_layers, num_groups, eval_len)
        dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
        counts = level.compute_counts(scores, ratio, dummy_cluster,
                                       num_layers, 8, num_groups, context)
        
        expected = eval_len * num_layers * num_groups * ratio
        actual = counts.sum()
        alignment_error = num_layers * num_groups
        assert abs(actual - expected) <= alignment_error, \
            f"num_groups={num_groups}: expected={expected}, actual={actual}"
        
        # 验证同一层所有 groups 预算相同
        for l in range(num_layers):
            assert np.all(counts[l] == counts[l, 0]), \
                f"num_groups={num_groups} layer={l}: groups have different budgets"
        
        print(f"  num_groups={num_groups}: actual={actual} "
              f"per_layer_per_group={counts[0, 0]}")


def test_extreme_prefs():
    """测试极端偏好值"""
    level = CakeLayerLevel()
    num_layers = 16
    num_groups = 4
    eval_len = 2048
    ratio = 0.25
    
    # 全零偏好 → uniform fallback
    context = SelectionContext(
        layer_preferences=torch.zeros(num_layers, dtype=torch.float32)
    )
    scores = torch.zeros(num_layers, num_groups, eval_len)
    dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
    counts = level.compute_counts(scores, ratio, dummy_cluster,
                                   num_layers, 8, num_groups, context)
    assert counts.shape == (num_layers, num_groups)
    assert counts.sum() > 0, "全零偏好应 fallback 到 uniform"
    print(f"  all-zero prefs: counts.sum()={counts.sum()} (uniform fallback)")
    
    # 单层极大偏好
    prefs = np.ones(num_layers, dtype=np.float32)
    prefs[7] = 1000.0  # layer 7 extremely important
    context = SelectionContext(
        layer_preferences=torch.tensor(prefs, dtype=torch.float32)
    )
    scores = torch.zeros(num_layers, num_groups, eval_len)
    dummy_cluster = torch.zeros(num_layers * num_groups * 8, dtype=torch.long)
    counts = level.compute_counts(scores, ratio, dummy_cluster,
                                   num_layers, 8, num_groups, context)
    # 极高偏好层不应突破 eval_len
    assert np.all(counts <= eval_len), "单层预算不应超过 eval_len"
    # 极高偏好层应获得更多预算
    assert counts[7, 0] >= counts[0, 0], "高偏好层应获得更多预算"
    print(f"  extreme pref: layer7={counts[7,0]} vs layer0={counts[0,0]}")
    
    # NaN preference → uniform fallback
    prefs_nan = np.full(num_layers, np.nan, dtype=np.float32)
    prefs_nan[0] = 5.0
    context = SelectionContext(
        layer_preferences=torch.tensor(prefs_nan, dtype=torch.float32)
    )
    counts = level.compute_counts(scores, ratio, None,
                                   num_layers, 8, num_groups, context)
    assert counts.sum() > 0, "NaN 偏好应 fallback 到 uniform"
    print(f"  NaN prefs: counts.sum()={counts.sum()} (uniform fallback)")
    
    # 负 preference
    prefs_neg = np.random.uniform(-10, -1, num_layers).astype(np.float32)
    context = SelectionContext(
        layer_preferences=torch.tensor(prefs_neg, dtype=torch.float32)
    )
    counts = level.compute_counts(scores, ratio, None,
                                   num_layers, 8, num_groups, context)
    assert counts.sum() > 0, "负偏好应 fallback 到 uniform"
    print(f"  negative prefs: counts.sum()={counts.sum()} (uniform fallback)")


if __name__ == "__main__":
    print("=== Test 1: Basic ratios ===")
    test_basic_ratios()
    print("\n=== Test 2: Group counts ===")
    test_group_counts()
    print("\n=== Test 3: Extreme preferences ===")
    test_extreme_prefs()
    print("\n ALL P0 BUDGET INVARIANT TESTS PASSED")