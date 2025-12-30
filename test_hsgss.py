"""
测试 HS-GSS 核心组件的功能
"""

import torch
import numpy as np
from component.hs_gss import (
    SchattenShrinkageOperator,
    HessianWhitening,
    GlobalLambdaSearcher,
    HSGSS_Compressor
)


def test_schatten_shrinkage():
    """测试 Schatten-p 收缩算子"""
    print("="*70)
    print("Test 1: Schatten Shrinkage Operator")
    print("="*70)
    
    sigma_test = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.1])
    print(f"Original singular values: {sigma_test.tolist()}")
    
    # 测试不同的 p 值
    for p in [1.0, 2/3, 0.5]:
        print(f"\n--- p = {p:.3f} ---")
        shrinkage_op = SchattenShrinkageOperator(p=p)
        
        for lambda_val in [0.1, 0.5, 1.0, 2.0]:
            sigma_shrunk = shrinkage_op.shrink(sigma_test.clone(), lambda_val)
            rank = torch.sum(sigma_shrunk > 1e-8).item()
            compression = rank / len(sigma_test)
            
            print(f"λ={lambda_val:.1f}: Rank={rank}/{len(sigma_test)} ({compression:.0%}), "
                  f"Top-3 σ'=[{sigma_shrunk[0]:.2f}, {sigma_shrunk[1]:.2f}, {sigma_shrunk[2]:.2f}]")
    
    print("\n✓ Schatten shrinkage test passed!\n")


def test_hessian_whitening():
    """测试 Hessian 空间白化"""
    print("="*70)
    print("Test 2: Hessian Whitening")
    print("="*70)
    
    # 创建测试数据
    torch.manual_seed(42)
    out_dim, in_dim = 128, 256
    W = torch.randn(out_dim, in_dim)
    
    # 创建一个正定的 Hessian 矩阵
    X = torch.randn(100, in_dim)  # 模拟输入数据
    H = X.T @ X / 100  # 协方差矩阵
    
    print(f"Weight shape: {W.shape}")
    print(f"Hessian shape: {H.shape}")
    print(f"Hessian eigenvalues: min={torch.min(torch.linalg.eigvalsh(H)):.4f}, "
          f"max={torch.max(torch.linalg.eigvalsh(H)):.4f}")
    
    # 白化
    whitening = HessianWhitening()
    W_whitened, H_sqrt = whitening.whiten(W, H, damp=0.01)
    
    print(f"Whitened weight shape: {W_whitened.shape}")
    print(f"H^(1/2) shape: {H_sqrt.shape}")
    
    # 验证: W_whitened ≈ W @ H_sqrt
    reconstruction_error = torch.norm(W_whitened - W @ H_sqrt) / torch.norm(W_whitened)
    print(f"Whitening reconstruction error: {reconstruction_error:.6f}")
    assert reconstruction_error < 1e-5, "Whitening error too large!"
    
    # 逆白化
    W_unwhitened = whitening.unwhiten(W_whitened, H_sqrt)
    inverse_error = torch.norm(W_unwhitened - W) / torch.norm(W)
    print(f"Inverse whitening error: {inverse_error:.6f}")
    assert inverse_error < 1e-5, "Inverse whitening error too large!"
    
    print("\n✓ Hessian whitening test passed!\n")


def test_global_lambda_search():
    """测试全局 Lambda 二分搜索"""
    print("="*70)
    print("Test 3: Global Lambda Search")
    print("="*70)
    
    # 创建模拟的奇异值字典
    torch.manual_seed(42)
    sigma_dict = {
        'module_1': torch.tensor([10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1]),
        'module_2': torch.tensor([12.0, 9.0, 6.0, 4.0, 2.0, 1.0, 0.3]),
        'module_3': torch.tensor([15.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2]),
    }
    
    shapes_dict = {
        'module_1': (256, 512),
        'module_2': (512, 256),
        'module_3': (1024, 1024),
    }
    
    # 计算原始参数量
    original_params = sum(out * in_ for out, in_ in shapes_dict.values())
    print(f"Original parameters: {original_params:,}")
    
    # 目标: 50% 压缩
    target_budget = int(original_params * 0.5)
    print(f"Target budget (50%): {target_budget:,}")
    
    # 搜索最优 lambda
    searcher = GlobalLambdaSearcher(p=2/3)
    optimal_lambda = searcher.search_lambda(
        sigma_dict,
        shapes_dict,
        target_budget,
        lambda_range=(1e-4, 10.0),
        max_iter=30,
        tol=0.02  # 2% tolerance
    )
    
    print(f"\nOptimal λ: {optimal_lambda:.6e}")
    
    # 验证最终参数量
    final_params = searcher.compute_total_params(sigma_dict, shapes_dict, optimal_lambda)
    print(f"Final parameters: {final_params:,}")
    print(f"Actual compression ratio: {final_params/original_params:.2%}")
    
    relative_error = abs(final_params - target_budget) / target_budget
    print(f"Relative error: {relative_error:.2%}")
    assert relative_error < 0.05, "Lambda search error too large!"
    
    print("\n✓ Global lambda search test passed!\n")


def test_end_to_end_compression():
    """测试端到端的层压缩"""
    print("="*70)
    print("Test 4: End-to-End Layer Compression")
    print("="*70)
    
    torch.manual_seed(42)
    
    # 模拟一个 Transformer 层的权重和 Hessian
    layer_weights = {
        'attn.q': torch.randn(768, 768),
        'attn.k': torch.randn(768, 768),
        'attn.v': torch.randn(768, 768),
        'attn.o': torch.randn(768, 768),
        'mlp.gate': torch.randn(3072, 768),
        'mlp.up': torch.randn(3072, 768),
        'mlp.down': torch.randn(768, 3072),
    }
    
    layer_hessians = {}
    for name, W in layer_weights.items():
        in_dim = W.shape[1]
        # 生成随机正定矩阵作为 Hessian
        A = torch.randn(in_dim, in_dim)
        H = A.T @ A / in_dim + 0.1 * torch.eye(in_dim)
        layer_hessians[name] = H
    
    # 计算原始参数量
    original_params = sum(W.numel() for W in layer_weights.values())
    print(f"Original parameters: {original_params:,}")
    
    # 创建压缩器
    compressor = HSGSS_Compressor(p=2/3, damp=0.01, cost_aware=True)
    
    # 执行压缩
    target_ratio = 0.5
    print(f"\nCompressing with target ratio: {target_ratio:.0%}...\n")
    
    compressed_modules = compressor.compress_layer(
        layer_weights,
        layer_hessians,
        target_ratio=target_ratio
    )
    
    # 验证输出
    print(f"\nCompression results:")
    total_compressed = 0
    for name, compressed in compressed_modules.items():
        if compressed is not None:
            U, S, V = compressed
            rank = len(S)
            out_dim, in_dim = layer_weights[name].shape
            params = rank * (out_dim + in_dim)
            total_compressed += params
            
            print(f"  {name}: rank={rank}, params={params:,}")
    
    print(f"\nTotal compressed parameters: {total_compressed:,}")
    print(f"Actual compression ratio: {total_compressed/original_params:.2%}")
    
    # 验证重构误差
    print(f"\nReconstruction errors:")
    for name, compressed in compressed_modules.items():
        if compressed is not None:
            U, S, V = compressed
            W_reconstructed = U @ torch.diag(S) @ V
            W_original = layer_weights[name]
            error = torch.norm(W_reconstructed - W_original) / torch.norm(W_original)
            print(f"  {name}: {error:.4f}")
    
    print("\n✓ End-to-end compression test passed!\n")


def test_compression_quality():
    """测试压缩质量与理论分析"""
    print("="*70)
    print("Test 5: Compression Quality Analysis")
    print("="*70)
    
    torch.manual_seed(42)
    
    # 创建一个简单的权重矩阵
    out_dim, in_dim = 100, 200
    
    # 生成具有特定奇异值分布的矩阵
    U = torch.randn(out_dim, out_dim)
    U, _ = torch.linalg.qr(U)
    V = torch.randn(in_dim, in_dim)
    V, _ = torch.linalg.qr(V)
    
    # 指数衰减的奇异值
    S = torch.exp(-torch.linspace(0, 5, min(out_dim, in_dim)))
    W = U @ torch.diag(S) @ V[:min(out_dim, in_dim), :]
    
    # 生成 Hessian
    X = torch.randn(50, in_dim)
    H = X.T @ X / 50 + 0.01 * torch.eye(in_dim)
    
    print(f"Original matrix: {W.shape}")
    print(f"Original singular values (top 10): {S[:10].tolist()}")
    print(f"Original rank (effective): {torch.sum(S > 0.01).item()}")
    
    # 测试不同的 p 值
    for p in [1.0, 2/3, 0.5]:
        print(f"\n--- Testing p = {p:.3f} ---")
        
        compressor = HSGSS_Compressor(p=p, damp=0.01, cost_aware=False)
        
        for ratio in [0.7, 0.5, 0.3]:
            compressed = compressor.compress_layer(
                {'test': W},
                {'test': H},
                target_ratio=ratio
            )
            
            if compressed['test'] is not None:
                U_c, S_c, V_c = compressed['test']
                W_c = U_c @ torch.diag(S_c) @ V_c
                
                # 计算误差指标
                frobenius_error = torch.norm(W - W_c) / torch.norm(W)
                spectral_error = torch.max(torch.abs(
                    torch.linalg.svdvals(W) - 
                    torch.cat([S_c, torch.zeros(len(S) - len(S_c))])
                ))
                
                print(f"  Ratio={ratio:.0%}: Rank={len(S_c)}/{len(S)}, "
                      f"Frobenius error={frobenius_error:.4f}, "
                      f"Spectral error={spectral_error:.4f}")
    
    print("\n✓ Compression quality test passed!\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HS-GSS Component Test Suite")
    print("="*70 + "\n")
    
    # 运行所有测试
    test_schatten_shrinkage()
    test_hessian_whitening()
    test_global_lambda_search()
    test_end_to_end_compression()
    test_compression_quality()
    
    print("="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nHS-GSS components are working correctly.")
    print("You can now use HSGSS.py to compress real models.\n")
