"""
可视化 HS-GSS 的收缩效果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from component.hs_gss import SchattenShrinkageOperator


def visualize_shrinkage_effect():
    """可视化不同 p 值和 λ 值下的收缩效果"""
    
    # 创建指数衰减的奇异值
    sigma_original = torch.exp(-torch.linspace(0, 5, 50))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HS-GSS: Schatten-p Shrinkage Effect', fontsize=16, fontweight='bold')
    
    # 测试不同的 p 值
    p_values = [1.0, 2/3, 0.5]
    lambda_values = [0.1, 0.3, 0.5, 1.0]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(lambda_values)))
    
    # 子图 1: 不同 λ 对 p=2/3 的影响
    ax = axes[0, 0]
    ax.plot(sigma_original.numpy(), 'k--', linewidth=2, label='Original', alpha=0.7)
    
    shrinkage_op = SchattenShrinkageOperator(p=2/3)
    for i, lambda_val in enumerate(lambda_values):
        sigma_shrunk = shrinkage_op.shrink(sigma_original.clone(), lambda_val)
        ax.plot(sigma_shrunk.numpy(), color=colors[i], linewidth=2, 
                label=f'λ={lambda_val}')
    
    ax.set_title('Effect of λ (p = 2/3)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子图 2: 不同 p 值对 λ=0.3 的影响
    ax = axes[0, 1]
    ax.plot(sigma_original.numpy(), 'k--', linewidth=2, label='Original', alpha=0.7)
    
    for i, p in enumerate(p_values):
        shrinkage_op = SchattenShrinkageOperator(p=p)
        sigma_shrunk = shrinkage_op.shrink(sigma_original.clone(), 0.3)
        ax.plot(sigma_shrunk.numpy(), linewidth=2, label=f'p={p:.2f}')
    
    ax.set_title('Effect of p (λ = 0.3)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子图 3: 收缩率分析
    ax = axes[1, 0]
    
    lambda_val = 0.5
    for i, p in enumerate(p_values):
        shrinkage_op = SchattenShrinkageOperator(p=p)
        sigma_shrunk = shrinkage_op.shrink(sigma_original.clone(), lambda_val)
        shrinkage_ratio = sigma_shrunk / (sigma_original + 1e-8)
        ax.plot(shrinkage_ratio.numpy(), linewidth=2, label=f'p={p:.2f}')
    
    ax.set_title(f'Shrinkage Ratio (λ = {lambda_val})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('σ\' / σ')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
    
    # 子图 4: 有效秩分析
    ax = axes[1, 1]
    
    effective_ranks = {}
    for p in p_values:
        shrinkage_op = SchattenShrinkageOperator(p=p)
        ranks = []
        lambda_range = np.linspace(0.05, 1.5, 30)
        
        for lambda_val in lambda_range:
            sigma_shrunk = shrinkage_op.shrink(sigma_original.clone(), lambda_val)
            rank = torch.sum(sigma_shrunk > 1e-6).item()
            ranks.append(rank)
        
        ax.plot(lambda_range, ranks, linewidth=2, marker='o', 
                markersize=4, label=f'p={p:.2f}')
    
    ax.set_title('Effective Rank vs λ', fontsize=12, fontweight='bold')
    ax.set_xlabel('λ (Lagrangian Multiplier)')
    ax.set_ylabel('Effective Rank')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hsgss_shrinkage_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: hsgss_shrinkage_visualization.png")
    plt.show()


def compare_hard_vs_soft_thresholding():
    """对比硬截断 (CPSVD) 和软收缩 (HS-GSS)"""
    
    # 创建奇异值
    sigma = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.1])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('CPSVD (Hard Thresholding) vs HS-GSS (Soft Shrinkage)', 
                 fontsize=14, fontweight='bold')
    
    # 子图 1: 原始奇异值
    ax = axes[0]
    x = np.arange(len(sigma))
    ax.bar(x, sigma.numpy(), color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title('Original Singular Values', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 子图 2: 硬截断 (CPSVD, k=5)
    ax = axes[1]
    k = 5
    sigma_hard = sigma.clone()
    sigma_hard[k:] = 0
    
    colors_hard = ['green' if i < k else 'red' for i in range(len(sigma))]
    ax.bar(x, sigma_hard.numpy(), color=colors_hard, alpha=0.7, edgecolor='black')
    ax.set_title(f'CPSVD: Hard Truncation (k={k})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.5, 0.95, f'Rank: {k}/{len(sigma)}', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 子图 3: 软收缩 (HS-GSS, λ=0.5, p=2/3)
    ax = axes[2]
    shrinkage_op = SchattenShrinkageOperator(p=2/3)
    sigma_soft = shrinkage_op.shrink(sigma.clone(), lambda_val=0.5)
    rank_soft = torch.sum(sigma_soft > 1e-6).item()
    
    colors_soft = ['green' if s > 1e-6 else 'red' for s in sigma_soft]
    ax.bar(x, sigma_soft.numpy(), color=colors_soft, alpha=0.7, edgecolor='black')
    ax.set_title('HS-GSS: Soft Shrinkage (λ=0.5, p=2/3)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3, axis='y')
    ax.text(0.5, 0.95, f'Rank: {rank_soft}/{len(sigma)}', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 添加注释
    for i, (s_orig, s_soft) in enumerate(zip(sigma[:3], sigma_soft[:3])):
        if s_soft > 1e-6:
            shrinkage = (s_orig - s_soft) / s_orig * 100
            axes[2].text(i, s_soft + 0.3, f'-{shrinkage:.1f}%', 
                        ha='center', fontsize=9, color='darkred')
    
    plt.tight_layout()
    plt.savefig('hard_vs_soft_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison saved to: hard_vs_soft_comparison.png")
    plt.show()


def visualize_global_allocation():
    """可视化全局 λ 如何自动分配压缩率"""
    
    # 模拟三个模块的奇异值和 Hessian
    modules = {
        'Attention (Important)': {
            'sigma': torch.tensor([8.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2]),
            'hessian_scale': 2.0,  # 大 Hessian → 等效奇异值更大
            'color': 'royalblue'
        },
        'FFN Gate (Medium)': {
            'sigma': torch.tensor([7.0, 5.0, 3.0, 1.5, 0.8, 0.3, 0.1]),
            'hessian_scale': 1.0,
            'color': 'orange'
        },
        'FFN Down (Less Important)': {
            'sigma': torch.tensor([6.0, 4.0, 2.0, 1.0, 0.4, 0.15, 0.05]),
            'hessian_scale': 0.5,  # 小 Hessian → 等效奇异值更小
            'color': 'crimson'
        }
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('HS-GSS: Global λ Automatic Allocation', 
                 fontsize=14, fontweight='bold')
    
    # 全局 λ
    lambda_global = 0.8
    shrinkage_op = SchattenShrinkageOperator(p=2/3)
    
    for ax_idx, (name, data) in enumerate(modules.items()):
        ax = axes[ax_idx]
        
        sigma_orig = data['sigma']
        hessian_scale = data['hessian_scale']
        
        # 等效奇异值 (经过 Hessian 加权)
        sigma_effective = sigma_orig * np.sqrt(hessian_scale)
        
        # 收缩
        sigma_shrunk_effective = shrinkage_op.shrink(sigma_effective.clone(), lambda_global)
        
        # 逆变换
        sigma_shrunk = sigma_shrunk_effective / np.sqrt(hessian_scale)
        
        rank = torch.sum(sigma_shrunk > 1e-6).item()
        retention = rank / len(sigma_orig)
        
        # 绘制
        x = np.arange(len(sigma_orig))
        width = 0.35
        
        ax.bar(x - width/2, sigma_orig.numpy(), width, 
               label='Original', color=data['color'], alpha=0.5, edgecolor='black')
        ax.bar(x + width/2, sigma_shrunk.numpy(), width,
               label='After HS-GSS', color=data['color'], alpha=0.9, edgecolor='black')
        
        ax.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加信息框
        info_text = f'Hessian: {hessian_scale:.1f}x\n'
        info_text += f'Rank: {rank}/{len(sigma_orig)}\n'
        info_text += f'Retention: {retention:.0%}'
        
        ax.text(0.98, 0.97, info_text, transform=ax.transAxes,
               ha='right', va='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 添加全局说明
    fig.text(0.5, 0.02, 
            f'Unified λ = {lambda_global} automatically allocates: '
            f'Important modules (large Hessian) → More retention | '
            f'Less important modules → More compression',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig('global_allocation_visualization.png', dpi=300, bbox_inches='tight')
    print("Global allocation visualization saved to: global_allocation_visualization.png")
    plt.show()


if __name__ == "__main__":
    print("Generating HS-GSS visualizations...\n")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 无需显示窗口
        
        print("1. Visualizing shrinkage effect...")
        visualize_shrinkage_effect()
        
        print("\n2. Comparing hard vs soft thresholding...")
        compare_hard_vs_soft_thresholding()
        
        print("\n3. Visualizing global allocation...")
        visualize_global_allocation()
        
        print("\n" + "="*70)
        print("All visualizations generated successfully!")
        print("="*70)
        print("\nGenerated files:")
        print("  - hsgss_shrinkage_visualization.png")
        print("  - hard_vs_soft_comparison.png")
        print("  - global_allocation_visualization.png")
        
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        print("Install it with: pip install matplotlib")
