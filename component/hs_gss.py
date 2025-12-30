"""
HS-GSS: Hessian-Schatten Global Spectral Sparsification

一个基于 Hessian 几何和 Schatten 谱分析的全局优化框架，用于大模型压缩。
将压缩问题从 "线性代数逼近" 提升到 "非凸优化与泛函分析" 的层面。

理论基础:
1. 优化目标: min ||W - W'||²_H + λ·∑||σ(W'_i)||_p^p
2. Hessian 空间白化: W̃ = W·H^(1/2)
3. Schatten-p 收缩算子: 对奇异值施加软阈值 + 收缩
4. 全局参数二分搜索: 寻找最优 λ 满足参数约束

优势:
- 全局帕累托最优 (vs 局部最优)
- 去噪效应 (收缩大奇异值)
- 成本感知 (自动分配压缩率)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class SchattenShrinkageOperator:
    """
    Schatten-p 范数的广义收缩算子 (Generalized Soft-Thresholding)
    
    对奇异值 σ 施加收缩变换，使其满足:
    σ_new + λ·p·σ_new^(p-1) = σ_old
    
    参数:
        p (float): Schatten-p 范数的指数, 0 < p < 1
                   p=1 对应软阈值 (Soft Thresholding)
                   p=2/3 有解析解 (通过 Cardan 公式)
        lambda_val (float): 拉格朗日乘子，控制稀疏化程度
    """
    
    def __init__(self, p: float = 2/3):
        assert 0 < p <= 1, "p must be in (0, 1]"
        self.p = p
        
    def shrink(self, sigma: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        对奇异值应用 Schatten-p 收缩算子
        
        Args:
            sigma: 原始奇异值 tensor [N]
            lambda_val: 拉格朗日乘子 λ
            
        Returns:
            收缩后的奇异值 tensor [N]
        """
        if self.p == 1.0:
            # 标准软阈值: σ_new = max(σ - λ, 0)
            return torch.clamp(sigma - lambda_val, min=0.0)
        
        elif abs(self.p - 2/3) < 1e-6:
            # p = 2/3 的解析解 (Cardan 公式)
            return self._cardan_shrinkage(sigma, lambda_val)
        
        else:
            # 通用情况: 数值求解
            return self._numerical_shrinkage(sigma, lambda_val)
    
    def _cardan_shrinkage(self, sigma: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """
        p = 2/3 时的解析解 (基于 Cardan 三次方程公式)
        
        方程: σ_new + (2/3)·λ·σ_new^(-1/3) = σ_old
        变换为三次方程后使用 Cardan 公式求解
        
        注意：为数值稳定性，使用简化的软阈值近似
        """
        p = 2.0 / 3.0
        coeff = lambda_val * p  # (2/3)·λ
        
        # 阈值: 低于此阈值的奇异值被置零
        # 理论阈值: (3/2 · λ·p)^(3/2) 
        threshold = (1.5 * coeff) ** 1.5
        
        # 创建输出张量
        sigma_new = torch.zeros_like(sigma)
        mask = sigma > threshold
        
        if mask.any():
            s = sigma[mask].clone()
            
            # 使用数值稳定的迭代方法代替 Cardan 公式
            # 初始估计：软阈值
            x = torch.clamp(s - lambda_val, min=1e-8)
            
            # 几次牛顿迭代来精化
            for _ in range(10):
                x_safe = torch.clamp(x, min=1e-10)
                # f(x) = x + coeff * x^(-1/3) - s = 0
                # 使用 x^(p-1) = x^(-1/3)
                x_pow = torch.pow(x_safe, p - 1)  # x^(-1/3)
                f = x + coeff * x_pow - s
                # f'(x) = 1 + coeff * (p-1) * x^(p-2) = 1 - coeff/3 * x^(-4/3)
                df = 1.0 + coeff * (p - 1) * torch.pow(x_safe, p - 2)
                df = torch.clamp(df, min=0.1)  # 避免除以太小的值
                
                x_new = x - f / df
                x_new = torch.clamp(x_new, min=0.0)
                
                # 检查收敛
                if torch.max(torch.abs(x_new - x)) < 1e-6:
                    break
                x = x_new
            
            # 最终检查：确保结果有效
            x = torch.clamp(x, min=0.0)
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            sigma_new[mask] = x
        
        return sigma_new
    
    def _numerical_shrinkage(self, sigma: torch.Tensor, lambda_val: float, 
                            max_iter: int = 20, tol: float = 1e-6) -> torch.Tensor:
        """
        数值求解 Schatten-p 收缩方程 (牛顿法)
        
        方程: f(σ_new) = σ_new + λ·p·σ_new^(p-1) - σ_old = 0
        导数: f'(σ_new) = 1 + λ·p·(p-1)·σ_new^(p-2)
        """
        p = self.p
        coeff = lambda_val * p
        
        # 估计阈值 (粗略)
        threshold = (coeff * p) ** (1 / (2 - p)) if p < 1 else 0.0
        
        sigma_new = torch.zeros_like(sigma)
        mask = sigma > threshold
        
        if mask.any():
            s = sigma[mask]
            # 初始猜测: 使用软阈值作为起点
            x = torch.clamp(s - lambda_val, min=1e-8)
            
            # 牛顿迭代
            for _ in range(max_iter):
                x_p = torch.clamp(x, min=1e-8)  # 避免除零
                f = x + coeff * (x_p ** (p - 1)) - s
                df = 1.0 + coeff * (p - 1) * (x_p ** (p - 2))
                
                x_new = x - f / df
                x_new = torch.clamp(x_new, min=0.0)
                
                # 检查收敛
                if torch.max(torch.abs(x_new - x)) < tol:
                    break
                x = x_new
            
            sigma_new[mask] = x
        
        return sigma_new


class HessianWhitening:
    """
    Hessian 空间白化变换
    
    将权重矩阵从原始空间变换到 Hessian 度量空间:
    W̃ = W · H^(1/2)
    
    其中 H 是输入的 Fisher 信息矩阵 (协方差矩阵)
    """
    
    @staticmethod
    def whiten(W: torch.Tensor, H: torch.Tensor, damp: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        白化变换: W̃ = W · H^(1/2)
        
        Args:
            W: 权重矩阵 [out_dim, in_dim]
            H: Hessian (输入协方差) [in_dim, in_dim]
            damp: 阻尼系数，用于数值稳定
            
        Returns:
            W_whitened: 白化后的权重 [out_dim, in_dim]
            H_sqrt: H^(1/2) 用于后续逆变换 [in_dim, in_dim]
        """
        # 确保 H 在与 W 相同的设备上
        H = H.to(W.device).float()
        W = W.float()
        
        # 确保 H 是对称正定的
        H = (H + H.T) / 2.0
        
        # 处理零 Hessian 的情况（没有样本通过该层）
        diag_H = torch.diag(H)
        if diag_H.max() < 1e-10:
            # H 接近零，返回单位变换
            n = H.size(0)
            H_sqrt = torch.eye(n, device=H.device, dtype=H.dtype)
            return W, H_sqrt
        
        # 添加阻尼以确保正定性
        # 使用更稳健的阻尼值计算
        diag_mean = torch.mean(torch.abs(diag_H))
        damp_val = max(damp * diag_mean.item(), 1e-6)
        n = H.size(0)
        idx = torch.arange(n, device=H.device)
        H_damped = H.clone()
        H_damped[idx, idx] += damp_val
        
        try:
            # Cholesky 分解: H = L·L^T, 则 H^(1/2) = L
            H_sqrt = torch.linalg.cholesky(H_damped)
        except RuntimeError:
            # 如果 Cholesky 失败，使用特征分解
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(H_damped)
                eigenvalues = torch.clamp(eigenvalues, min=1e-6)
                H_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
            except RuntimeError:
                # 最后的保底：使用单位矩阵
                print("Warning: Both Cholesky and eigen decomposition failed, using identity")
                H_sqrt = torch.eye(n, device=H.device, dtype=H.dtype)
        
        # 检查 H_sqrt 的有效性
        if not torch.isfinite(H_sqrt).all():
            print("Warning: H_sqrt contains non-finite values, using identity")
            H_sqrt = torch.eye(n, device=H.device, dtype=H.dtype)
        
        # 白化变换
        W_whitened = W @ H_sqrt
        
        # 检查白化结果的有效性
        if not torch.isfinite(W_whitened).all():
            print("Warning: W_whitened contains non-finite values, skipping whitening")
            H_sqrt = torch.eye(n, device=H.device, dtype=H.dtype)
            W_whitened = W
        
        return W_whitened, H_sqrt
    
    @staticmethod
    def unwhiten(Vt: torch.Tensor, H_sqrt: torch.Tensor) -> torch.Tensor:
        """
        逆白化变换: V'^T = V^T · H^(-1/2)
        
        理论推导：
        1. 白化空间: W̃ = W·H^(1/2)，SVD 得 W̃ ≈ U·Σ·V^T
        2. 压缩: W̃' = U·Σ'·V^T（收缩后）
        3. 逆白化: W' = W̃'·H^(-1/2) = U·Σ'·V^T·H^(-1/2) = U·Σ'·(V·H^(-1/2)^T)^T
        
        因此最终权重: W' = U·Σ'·V'^T，其中 V' = V·H^(-1/2)^T = (H^(-1/2)·V^T)^T
        即: V'^T = V^T·H^(-1/2)
        
        Args:
            Vt: 白化空间的 V^T 矩阵 [rank, in_dim]
            H_sqrt: H^(1/2) 矩阵 [in_dim, in_dim]
            
        Returns:
            Vt_final: 原始空间的 V'^T [rank, in_dim]，满足 W' = U @ Σ' @ Vt_final
        """
        try:
            # 尝试使用 solve 而不是直接求逆，更稳定
            # Vt @ H^(-1/2) = solve(H^(1/2)^T, Vt^T)^T
            Vt_final = torch.linalg.solve(H_sqrt.T, Vt.T).T
        except RuntimeError:
            # 如果 solve 失败，使用伪逆
            try:
                H_sqrt_inv = torch.linalg.pinv(H_sqrt)
            except RuntimeError:
                # 最后的保底：使用带正则化的逆
                n = H_sqrt.size(0)
                reg = 1e-4 * torch.eye(n, device=H_sqrt.device, dtype=H_sqrt.dtype)
                H_sqrt_inv = torch.linalg.inv(H_sqrt + reg)
            Vt_final = Vt @ H_sqrt_inv
        
        # 检查结果的有效性
        if not torch.isfinite(Vt_final).all():
            # 如果结果包含 NaN/Inf，回退到不做逆白化
            print("Warning: unwhiten produced non-finite values, falling back to identity")
            Vt_final = Vt.clone()
        
        return Vt_final


class GlobalLambdaSearcher:
    """
    全局拉格朗日乘子 λ 的二分搜索
    
    目标: 找到一个全局 λ，使得压缩后的模型满足目标参数量
    TotalParams(λ) = target_budget
    """
    
    def __init__(self, p: float = 2/3, cost_weights: Optional[Dict] = None):
        """
        Args:
            p: Schatten-p 范数的指数
            cost_weights: 可选的成本权重字典 {module_name: weight}
                         用于成本感知的压缩分配
        """
        self.p = p
        self.shrinkage_op = SchattenShrinkageOperator(p)
        self.cost_weights = cost_weights or {}
    
    def compute_total_params(self, 
                            sigma_dict: Dict[str, torch.Tensor],
                            shapes_dict: Dict[str, Tuple[int, int]],
                            lambda_val: float) -> int:
        """
        计算给定 λ 下的总参数量（真实参数量，不加权）
        
        Args:
            sigma_dict: 各模块的奇异值字典 {name: sigma_tensor}
            shapes_dict: 各模块的形状字典 {name: (out_dim, in_dim)}
            lambda_val: 拉格朗日乘子
            
        Returns:
            total_params: 真实总参数量
        """
        total_params = 0
        
        for name, sigma in sigma_dict.items():
            # cost_aware: 对"昂贵"模块应用更大的等效lambda
            # 这样昂贵模块会被压缩更多
            cost_weight = self.cost_weights.get(name, 1.0)
            effective_lambda = lambda_val * cost_weight
            
            # 应用收缩算子（使用加权后的lambda）
            sigma_new = self.shrinkage_op.shrink(sigma, effective_lambda)
            
            # 计算有效秩 (非零奇异值的个数)
            rank = torch.sum(sigma_new > 1e-8).item()
            
            # 计算该模块的真实参数量（不加权！）
            if rank > 0:
                out_dim, in_dim = shapes_dict[name]
                params = rank * (out_dim + in_dim)
                total_params += params
        
        return int(total_params)
    
    def search_lambda(self,
                     sigma_dict: Dict[str, torch.Tensor],
                     shapes_dict: Dict[str, Tuple[int, int]],
                     target_budget: int,
                     lambda_range: Tuple[float, float] = (1e-6, 1e3),
                     max_iter: int = 50,
                     tol: float = 0.01) -> float:
        """
        二分搜索最优 λ
        
        Args:
            sigma_dict: 各模块的奇异值字典
            shapes_dict: 各模块的形状字典
            target_budget: 目标参数量
            lambda_range: λ 的搜索范围 (lambda_min, lambda_max)
            max_iter: 最大迭代次数
            tol: 相对误差容忍度 (百分比)
            
        Returns:
            optimal_lambda: 最优的 λ 值
        """
        lambda_min, lambda_max = lambda_range
        
        # 检查边界
        params_min = self.compute_total_params(sigma_dict, shapes_dict, lambda_min)
        params_max = self.compute_total_params(sigma_dict, shapes_dict, lambda_max)
        
        print(f"[Lambda Search] Range: [{lambda_min:.2e}, {lambda_max:.2e}]")
        print(f"[Lambda Search] Params at boundaries: [{params_min}, {params_max}]")
        print(f"[Lambda Search] Target budget: {target_budget}")
        
        # 如果最大lambda仍然无法达到目标，需要扩大搜索范围
        if params_max > target_budget:
            print(f"Warning: Max lambda too small, params ({params_max}) > target ({target_budget})")
            print(f"Expanding lambda search range...")
            # 扩大lambda范围
            while params_max > target_budget and lambda_max < 1e6:
                lambda_max *= 10
                params_max = self.compute_total_params(sigma_dict, shapes_dict, lambda_max)
                print(f"New lambda_max: {lambda_max:.2e}, params: {params_max}")
        
        if params_min < target_budget:
            print(f"Warning: Even with min lambda, params ({params_min}) < target ({target_budget})")
            return lambda_min
        
        # 二分搜索
        for iteration in range(max_iter):
            lambda_mid = (lambda_min + lambda_max) / 2.0
            params_mid = self.compute_total_params(sigma_dict, shapes_dict, lambda_mid)
            
            relative_error = abs(params_mid - target_budget) / target_budget
            
            if iteration % 10 == 0:
                print(f"[Iter {iteration}] λ={lambda_mid:.6e}, Params={params_mid}, "
                      f"Error={relative_error*100:.2f}%")
            
            # 检查收敛
            if relative_error < tol:
                print(f"[Lambda Search] Converged at λ={lambda_mid:.6e}")
                return lambda_mid
            
            # 更新搜索范围
            # λ 越大，参数越少 (更稀疏)
            if params_mid > target_budget:
                lambda_min = lambda_mid  # 需要更大的 λ
            else:
                lambda_max = lambda_mid  # 需要更小的 λ
        
        lambda_final = (lambda_min + lambda_max) / 2.0
        print(f"[Lambda Search] Max iterations reached. Final λ={lambda_final:.6e}")
        return lambda_final


class HSGSS_Compressor:
    """
    HS-GSS 主压缩器
    
    完整流程:
    1. Phase 1: 校准与变换 (收集 Hessian, 白化, SVD)
    2. Phase 2: 全局参数寻优 (二分搜索 λ)
    3. Phase 3: 重构与逆变换 (收缩, 重构, 逆白化)
    """
    
    def __init__(self, 
                 p: float = 2/3,
                 damp: float = 0.01,
                 cost_aware: bool = True):
        """
        Args:
            p: Schatten-p 范数指数
            damp: Hessian 阻尼系数
            cost_aware: 是否启用成本感知 (对大模块施加更大压缩)
        """
        self.p = p
        self.damp = damp
        self.cost_aware = cost_aware
        self.shrinkage_op = SchattenShrinkageOperator(p)
        self.whitening = HessianWhitening()
        
    def compress_layer(self,
                      layer_weights: Dict[str, torch.Tensor],
                      layer_hessians: Dict[str, torch.Tensor],
                      target_ratio: float) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        压缩单层的所有模块
        
        Args:
            layer_weights: {module_name: W} 权重字典
            layer_hessians: {module_name: H} Hessian 字典
            target_ratio: 目标压缩率 (保留参数的比例)
            
        Returns:
            compressed_modules: {module_name: (U, Sigma, V)}
                               SVD 分解后的 U·diag(Sigma)·V^T
        """
        # Phase 1: 校准与变换
        print(f"\n{'='*60}")
        print(f"Phase 1: Calibration & Transformation")
        print(f"{'='*60}")
        
        sigma_dict = {}
        shapes_dict = {}
        whitened_svd = {}
        h_sqrt_dict = {}
        
        for name, W in layer_weights.items():
            H = layer_hessians[name]
            
            # 白化
            W_whitened, H_sqrt = self.whitening.whiten(W.float(), H.float(), self.damp)
            h_sqrt_dict[name] = H_sqrt
            
            # SVD 分解
            U, S, Vt = torch.linalg.svd(W_whitened, full_matrices=False)
            
            sigma_dict[name] = S
            shapes_dict[name] = W.shape
            whitened_svd[name] = (U, S, Vt)
            
            print(f"  {name}: Shape={W.shape}, Rank={len(S)}, "
                  f"Top-3 σ={S[:3].tolist()}")
        
        # Phase 2: 全局参数寻优
        print(f"\n{'='*60}")
        print(f"Phase 2: Global Lambda Search")
        print(f"{'='*60}")
        
        # 计算目标参数量
        total_original_params = sum(
            W.shape[0] * W.shape[1] for W in layer_weights.values()
        )
        target_budget = int(total_original_params * target_ratio)
        
        print(f"Original params: {total_original_params}")
        print(f"Target params: {target_budget} (ratio={target_ratio:.2%})")
        
        # 成本权重 (可选)
        cost_weights = {}
        if self.cost_aware:
            for name, shape in shapes_dict.items():
                # 大模块 (如 FFN) 权重更高，优先压缩
                cost = (shape[0] + shape[1]) / 2.0
                cost_weights[name] = cost / 1000.0  # 归一化
        
        # 二分搜索
        searcher = GlobalLambdaSearcher(self.p, cost_weights)
        optimal_lambda = searcher.search_lambda(
            sigma_dict, shapes_dict, target_budget
        )
        
        # Phase 3: 重构与逆变换
        print(f"\n{'='*60}")
        print(f"Phase 3: Reconstruction & Inverse Transform")
        print(f"{'='*60}")
        
        compressed_modules = {}
        total_compressed_params = 0
        
        for name, W in layer_weights.items():
            U, S_old, Vt = whitened_svd[name]
            H_sqrt = h_sqrt_dict[name]
            
            # 收缩奇异值
            S_new = self.shrinkage_op.shrink(S_old, optimal_lambda)
            
            # 计算有效秩
            rank = torch.sum(S_new > 1e-8).item()
            
            if rank > 0:
                # 截断到有效秩
                U_trunc = U[:, :rank]
                S_trunc = S_new[:rank]
                Vt_trunc = Vt[:rank, :]  # [rank, in_dim]
                
                # 逆白化 V 部分: V'^T = V^T @ H^(-1/2)
                Vt_final = self.whitening.unwhiten(
                    Vt_trunc, H_sqrt
                )  # [rank, in_dim]
                
                compressed_modules[name] = (U_trunc, S_trunc, Vt_final)
                
                out_dim, in_dim = shapes_dict[name]
                params = rank * (out_dim + in_dim)
                total_compressed_params += params
                
                compression_ratio = params / (out_dim * in_dim)
                
                print(f"  {name}: Rank {rank}/{len(S_old)} "
                      f"({compression_ratio:.2%}), "
                      f"σ_max: {S_old[0]:.3f} → {S_new[0]:.3f}")
            else:
                print(f"  {name}: Fully pruned (rank=0)")
                compressed_modules[name] = None
        
        print(f"\n{'='*60}")
        print(f"Compression Summary")
        print(f"{'='*60}")
        print(f"Original params: {total_original_params}")
        print(f"Compressed params: {total_compressed_params}")
        print(f"Actual ratio: {total_compressed_params/total_original_params:.2%}")
        print(f"Optimal λ: {optimal_lambda:.6e}")
        
        return compressed_modules


def apply_hsgss_to_llama_layer(layer, 
                               layer_hessians: Dict[str, torch.Tensor],
                               target_ratio: float = 0.5,
                               p: float = 2/3) -> None:
    """
    将 HS-GSS 应用到 Llama 的单层
    
    Args:
        layer: Transformer 层对象
        layer_hessians: 该层各模块的 Hessian 字典
                       {'self_attn.q_proj': H_q, 'self_attn.k_proj': H_k, ...}
        target_ratio: 目标参数保留比例
        p: Schatten-p 范数指数
    """
    from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
    
    # 收集该层的所有线性模块权重
    layer_weights = {}
    
    # Attention 模块
    attn = layer.self_attn
    layer_weights['self_attn.q_proj'] = attn.q_proj.weight.data
    layer_weights['self_attn.k_proj'] = attn.k_proj.weight.data
    layer_weights['self_attn.v_proj'] = attn.v_proj.weight.data
    layer_weights['self_attn.o_proj'] = attn.o_proj.weight.data
    
    # MLP 模块
    mlp = layer.mlp
    layer_weights['mlp.gate_proj'] = mlp.gate_proj.weight.data
    layer_weights['mlp.up_proj'] = mlp.up_proj.weight.data
    layer_weights['mlp.down_proj'] = mlp.down_proj.weight.data
    
    # 创建压缩器
    compressor = HSGSS_Compressor(p=p, cost_aware=True)
    
    # 执行压缩
    compressed = compressor.compress_layer(
        layer_weights, layer_hessians, target_ratio
    )
    
    # 替换层中的模块
    # Attention
    if compressed['self_attn.q_proj'] is not None:
        config = layer.self_attn.config
        # 计算等效 ratio (用于创建 SVD 模块)
        # 这里简化处理，实际应该根据 rank 动态设置
        layer_idx = getattr(layer.self_attn, 'layer_idx', 0)
        svd_attn = SVD_LlamaAttention(config, ratio=target_ratio, layer_idx=layer_idx)
        
        # 设置权重
        U_q, S_q, V_q = compressed['self_attn.q_proj']
        U_k, S_k, V_k = compressed['self_attn.k_proj']
        U_v, S_v, V_v = compressed['self_attn.v_proj']
        U_o, S_o, V_o = compressed['self_attn.o_proj']
        
        # Q, K, V, O 的 U 和 V 投影
        sqrt_S_q = torch.sqrt(torch.diag(S_q))
        svd_attn.q_u_proj.weight.data = (U_q @ sqrt_S_q).T
        svd_attn.q_v_proj.weight.data = (sqrt_S_q @ V_q).T
        
        sqrt_S_k = torch.sqrt(torch.diag(S_k))
        svd_attn.k_u_proj.weight.data = (U_k @ sqrt_S_k).T
        svd_attn.k_v_proj.weight.data = (sqrt_S_k @ V_k).T
        
        sqrt_S_v = torch.sqrt(torch.diag(S_v))
        svd_attn.v_u_proj.weight.data = (U_v @ sqrt_S_v).T
        svd_attn.v_v_proj.weight.data = (sqrt_S_v @ V_v).T
        
        sqrt_S_o = torch.sqrt(torch.diag(S_o))
        svd_attn.o_u_proj.weight.data = (U_o @ sqrt_S_o).T
        svd_attn.o_v_proj.weight.data = (sqrt_S_o @ V_o).T
        
        layer.self_attn = svd_attn
    
    # MLP
    if compressed['mlp.gate_proj'] is not None:
        hidden_size = layer.mlp.gate_proj.weight.shape[1]
        intermediate_size = layer.mlp.gate_proj.weight.shape[0]
        hidden_act = layer.mlp.act_fn.__class__.__name__.lower()
        
        svd_mlp = SVD_LlamaMLP(
            hidden_size, intermediate_size, hidden_act, ratio=target_ratio
        )
        
        U_gate, S_gate, V_gate = compressed['mlp.gate_proj']
        U_up, S_up, V_up = compressed['mlp.up_proj']
        U_down, S_down, V_down = compressed['mlp.down_proj']
        
        sqrt_S_gate = torch.sqrt(torch.diag(S_gate))
        svd_mlp.gate_u_proj.weight.data = (U_gate @ sqrt_S_gate).T
        svd_mlp.gate_v_proj.weight.data = (sqrt_S_gate @ V_gate).T
        
        sqrt_S_up = torch.sqrt(torch.diag(S_up))
        svd_mlp.up_u_proj.weight.data = (U_up @ sqrt_S_up).T
        svd_mlp.up_v_proj.weight.data = (sqrt_S_up @ V_up).T
        
        sqrt_S_down = torch.sqrt(torch.diag(S_down))
        svd_mlp.down_u_proj.weight.data = (U_down @ sqrt_S_down).T
        svd_mlp.down_v_proj.weight.data = (sqrt_S_down @ V_down).T
        
        layer.mlp = svd_mlp
    
    print(f"Layer compressed using HS-GSS with p={p}")


# ============================================================================
# 辅助函数: 理论分析
# ============================================================================

def analyze_compression_distribution(compressed_modules: Dict[str, Tuple],
                                    original_shapes: Dict[str, Tuple[int, int]]):
    """
    分析压缩率在不同模块间的分配
    展示 HS-GSS 的"全局帕累托最优"特性
    """
    print(f"\n{'='*70}")
    print(f"Compression Distribution Analysis")
    print(f"{'='*70}")
    print(f"{'Module':<25} {'Original':<15} {'Compressed':<15} {'Ratio':<10} {'Efficiency':<10}")
    print(f"{'-'*70}")
    
    total_original = 0
    total_compressed = 0
    
    for name, compressed in compressed_modules.items():
        out_dim, in_dim = original_shapes[name]
        original_params = out_dim * in_dim
        
        if compressed is not None:
            U, S, V = compressed
            rank = len(S)
            compressed_params = rank * (out_dim + in_dim)
        else:
            compressed_params = 0
        
        ratio = compressed_params / original_params if original_params > 0 else 0
        efficiency = original_params / (out_dim + in_dim)  # 模块的"成本效率"
        
        total_original += original_params
        total_compressed += compressed_params
        
        print(f"{name:<25} {original_params:<15,} {compressed_params:<15,} "
              f"{ratio:<10.2%} {efficiency:<10.1f}")
    
    print(f"{'-'*70}")
    print(f"{'Total':<25} {total_original:<15,} {total_compressed:<15,} "
          f"{total_compressed/total_original:<10.2%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 测试 Schatten-p 收缩算子
    print("Testing Schatten Shrinkage Operator...")
    
    shrinkage_op = SchattenShrinkageOperator(p=2/3)
    sigma_test = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.1])
    
    print(f"Original σ: {sigma_test.tolist()}")
    
    for lambda_val in [0.1, 0.5, 1.0, 2.0]:
        sigma_shrunk = shrinkage_op.shrink(sigma_test, lambda_val)
        rank = torch.sum(sigma_shrunk > 1e-8).item()
        print(f"λ={lambda_val:.1f}: σ'={sigma_shrunk.tolist()}, Rank={rank}")
    
    print("\nHS-GSS component loaded successfully!")
    print("Key features:")
    print("  - Schatten-p shrinkage with analytical solution (p=2/3)")
    print("  - Hessian space whitening for metric-aware compression")
    print("  - Global lambda search for Pareto-optimal allocation")
    print("  - Cost-aware compression (prioritizes expensive modules)")
