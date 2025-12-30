"""
C-GSVR 测试脚本

用于验证 C-GSVR 实现的正确性
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

# 添加项目路径
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_marginal_utility_allocator():
    """测试边际效用秩分配器"""
    print("\n" + "="*60)
    print("Test 1: Marginal Utility Rank Allocator")
    print("="*60)
    
    from component.c_gsvr import MarginalUtilityRankAllocator
    
    # 创建模拟数据
    sigma_dict = {
        'layer_0.self_attn.q_proj': torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5]),
        'layer_0.self_attn.k_proj': torch.tensor([8.0, 4.0, 2.0, 1.0]),
        'layer_0.mlp.gate_proj': torch.tensor([15.0, 8.0, 4.0, 2.0, 1.0, 0.5]),
    }
    
    shapes_dict = {
        'layer_0.self_attn.q_proj': (4096, 4096),
        'layer_0.self_attn.k_proj': (4096, 4096),
        'layer_0.mlp.gate_proj': (11008, 4096),
    }
    
    layer_module_map = {
        'layer_0.self_attn.q_proj': (0, 'self_attn.q_proj'),
        'layer_0.self_attn.k_proj': (0, 'self_attn.k_proj'),
        'layer_0.mlp.gate_proj': (0, 'mlp.gate_proj'),
    }
    
    # 计算原始参数量
    total_original = sum(s[0] * s[1] for s in shapes_dict.values())
    print(f"Total original params: {total_original:,}")
    
    # 创建分配器
    allocator = MarginalUtilityRankAllocator()
    allocator.build_global_utility_table(sigma_dict, shapes_dict, layer_module_map)
    
    # 测试不同压缩率
    for ratio in [0.8, 0.5, 0.3]:
        print(f"\n--- Target ratio: {ratio:.0%} ---")
        rank_allocation = allocator.allocate_ranks(total_original, ratio)
        
        # 计算实际压缩后参数量
        actual_params = 0
        for name, rank in rank_allocation.items():
            shape = shapes_dict[name]
            actual_params += rank * (shape[0] + shape[1])
        
        actual_ratio = actual_params / total_original
        print(f"Allocated ranks: {rank_allocation}")
        print(f"Actual ratio: {actual_ratio:.2%}")
        print(f"Target ratio: {ratio:.2%}")
        print(f"Error: {abs(actual_ratio - ratio) / ratio * 100:.2f}%")
    
    print("\n✓ Marginal utility allocator test passed!")
    return True


def test_fisher_hessian_computer():
    """测试 Fisher-Hessian 计算器"""
    print("\n" + "="*60)
    print("Test 2: Fisher-Hessian Computer")
    print("="*60)
    
    from component.c_gsvr import FisherHessianComputer
    
    computer = FisherHessianComputer(damp=0.01)
    
    # 测试矩阵平方根
    print("\nTesting matrix square root...")
    M = torch.randn(64, 64)
    M = M @ M.T  # 确保正定
    
    M_sqrt = computer.compute_sqrt_matrix(M)
    M_reconstructed = M_sqrt @ M_sqrt
    
    error = torch.norm(M_reconstructed - M) / torch.norm(M)
    print(f"Reconstruction error: {error:.6f}")
    assert error < 0.1, f"Matrix sqrt error too large: {error}"
    
    # 测试逆平方根
    print("\nTesting inverse square root...")
    M_inv_sqrt = computer.compute_inv_sqrt_matrix(M)
    identity_approx = M_sqrt @ M_inv_sqrt
    
    identity_error = torch.norm(identity_approx - torch.eye(64)) / 64
    print(f"Identity approximation error: {identity_error:.6f}")
    assert identity_error < 0.2, f"Inverse sqrt error too large: {identity_error}"
    
    print("\n✓ Fisher-Hessian computer test passed!")
    return True


def test_weighted_svd():
    """测试加权 SVD"""
    print("\n" + "="*60)
    print("Test 3: Weighted SVD Computer")
    print("="*60)
    
    from component.c_gsvr import WeightedSVDComputer
    
    computer = WeightedSVDComputer(damp=0.01, use_fisher=False)
    
    # 创建测试数据
    W = torch.randn(128, 64)
    H = torch.randn(64, 64)
    H = H @ H.T  # Hessian 应该是正定的
    
    print(f"Weight shape: {W.shape}")
    print(f"Hessian shape: {H.shape}")
    
    # 执行加权 SVD
    U, S, Vt, H_sqrt, F_sqrt = computer.weighted_svd(W, H, None, 'cpu')
    
    print(f"U shape: {U.shape}")
    print(f"S shape: {S.shape}")
    print(f"Vt shape: {Vt.shape}")
    print(f"Top-5 singular values: {S[:5].tolist()}")
    
    # 验证 SVD 分解
    W_reconstructed = U @ torch.diag(S) @ Vt
    W_whitened = W @ H_sqrt
    recon_error = torch.norm(W_reconstructed - W_whitened) / torch.norm(W_whitened)
    print(f"SVD reconstruction error: {recon_error:.6f}")
    
    assert recon_error < 0.01, f"SVD reconstruction error too large: {recon_error}"
    
    print("\n✓ Weighted SVD test passed!")
    return True


def test_compression_ratio():
    """测试压缩率控制精度"""
    print("\n" + "="*60)
    print("Test 4: Compression Ratio Control")
    print("="*60)
    
    from component.c_gsvr import MarginalUtilityRankAllocator
    
    # 模拟一个小型模型（32层，每层7个模块）
    n_layers = 32
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    sigma_dict = {}
    shapes_dict = {}
    layer_module_map = {}
    
    np.random.seed(42)
    
    for i in range(n_layers):
        for m in modules:
            global_name = f"layer_{i}.{m}"
            
            # 模拟奇异值（衰减）
            if 'proj' in m and m.startswith(('gate', 'up', 'down')):
                out_dim, in_dim = 11008, 4096
            else:
                out_dim, in_dim = 4096, 4096
            
            rank = min(out_dim, in_dim)
            # 模拟衰减的奇异值
            sigma = torch.tensor([100.0 * np.exp(-k/50) + np.random.rand() for k in range(rank)])
            
            sigma_dict[global_name] = sigma
            shapes_dict[global_name] = (out_dim, in_dim)
            layer_module_map[global_name] = (i, m)
    
    total_original = sum(s[0] * s[1] for s in shapes_dict.values())
    print(f"Simulated model: {n_layers} layers, {len(sigma_dict)} modules")
    print(f"Total original params: {total_original:,}")
    
    allocator = MarginalUtilityRankAllocator()
    allocator.build_global_utility_table(sigma_dict, shapes_dict, layer_module_map)
    
    # 测试不同压缩率
    errors = []
    for ratio in [0.7, 0.5, 0.3, 0.2]:
        rank_allocation = allocator.allocate_ranks(total_original, ratio)
        
        actual_params = sum(
            rank * (shapes_dict[name][0] + shapes_dict[name][1])
            for name, rank in rank_allocation.items()
        )
        actual_ratio = actual_params / total_original
        error = abs(actual_ratio - ratio) / ratio * 100
        errors.append(error)
        
        print(f"Ratio {ratio:.0%}: actual={actual_ratio:.4%}, error={error:.2f}%")
    
    avg_error = np.mean(errors)
    print(f"\nAverage ratio error: {avg_error:.2f}%")
    
    # 压缩率误差应该很小（因为贪心选择）
    assert avg_error < 5.0, f"Average ratio error too large: {avg_error}%"
    
    print("\n✓ Compression ratio control test passed!")
    return True


def test_layer_wise_distribution():
    """测试层间秩分配的差异化"""
    print("\n" + "="*60)
    print("Test 5: Layer-wise Rank Distribution")
    print("="*60)
    
    from component.c_gsvr import MarginalUtilityRankAllocator
    
    # 模拟场景：前面层"好压"，后面层"难压"
    n_layers = 8
    
    sigma_dict = {}
    shapes_dict = {}
    layer_module_map = {}
    
    for i in range(n_layers):
        global_name = f"layer_{i}.proj"
        
        # 前面层的奇异值衰减快（好压）
        # 后面层的奇异值衰减慢（难压）
        decay_rate = 0.1 if i < n_layers // 2 else 0.02
        sigma = torch.tensor([10.0 * np.exp(-k * decay_rate) for k in range(100)])
        
        sigma_dict[global_name] = sigma
        shapes_dict[global_name] = (1024, 1024)
        layer_module_map[global_name] = (i, 'proj')
    
    total_original = sum(s[0] * s[1] for s in shapes_dict.values())
    
    allocator = MarginalUtilityRankAllocator()
    allocator.build_global_utility_table(sigma_dict, shapes_dict, layer_module_map)
    
    rank_allocation = allocator.allocate_ranks(total_original, 0.5)
    
    print("\nRank distribution (expecting harder layers to get more ranks):")
    early_layers_rank = 0
    late_layers_rank = 0
    
    for name, rank in sorted(rank_allocation.items()):
        layer_idx = int(name.split('_')[1].split('.')[0])
        print(f"  {name}: rank={rank}")
        if layer_idx < n_layers // 2:
            early_layers_rank += rank
        else:
            late_layers_rank += rank
    
    print(f"\nEarly layers (easy) total rank: {early_layers_rank}")
    print(f"Late layers (hard) total rank: {late_layers_rank}")
    
    # 验证：难压的层应该获得更多秩
    assert late_layers_rank > early_layers_rank, \
        "Harder layers should receive more ranks!"
    
    print("\n✓ Layer-wise distribution test passed!")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("C-GSVR Unit Tests")
    print("="*70)
    
    tests = [
        ("Marginal Utility Allocator", test_marginal_utility_allocator),
        ("Fisher-Hessian Computer", test_fisher_hessian_computer),
        ("Weighted SVD", test_weighted_svd),
        ("Compression Ratio Control", test_compression_ratio),
        ("Layer-wise Distribution", test_layer_wise_distribution),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} failed!")
        except Exception as e:
            failed += 1
            print(f"✗ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"Test Results: {passed}/{len(tests)} passed")
    print("="*70)
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} tests failed!")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='C-GSVR Test Suite')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'allocator', 'fh', 'svd', 'ratio', 'distribution'],
                        help='Which test to run')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
        exit(0 if success else 1)
    elif args.test == 'allocator':
        test_marginal_utility_allocator()
    elif args.test == 'fh':
        test_fisher_hessian_computer()
    elif args.test == 'svd':
        test_weighted_svd()
    elif args.test == 'ratio':
        test_compression_ratio()
    elif args.test == 'distribution':
        test_layer_wise_distribution()
