"""
C-GSVR: Compensated Global Semantic Variable-Rank (补偿性全局语义变秩法)

核心思想：
1. 边际效用等价原理 - 每层增加一个秩的效用/成本比相等时全局最优
2. SVD-LLM 白化 - 使用 Cholesky 分解进行输入协方差白化
3. 递归误差吸收 - 利用下一层 Hessian 逆吸收当前层截断误差

主要优势：
- 严格保证全局压缩率（通过边际收益排序+硬截断）
- 层间自适应（好压的层多出预算，难压的层获得更多秩）
- 误差补偿（顺序压缩时让下一层消化前一层损失）
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import heapq
from utils.model_utils import find_layers


@dataclass
class MarginalUtilityEntry:
    """边际效用条目：记录每个奇异值的边际收益信息"""
    layer_idx: int           # 层索引
    module_name: str         # 模块名称
    rank_idx: int           # 秩索引 (第几个奇异值)
    marginal_utility: float  # 边际效用 (误差下降量 / 参数成本)
    weighted_sigma_sq: float # 加权奇异值平方 (重要性得分)
    param_cost: int         # 增加该秩的参数成本
    
    def __lt__(self, other):
        """用于堆排序：按边际效用降序"""
        return self.marginal_utility > other.marginal_utility


class WhiteningComputer:
    """
    SVD-LLM 风格的白化计算器
    
    使用 Cholesky 分解对输入协方差矩阵进行白化：
    1. 收集输入协方差: H = X^T @ X
    2. Cholesky 分解: L = cholesky(H)
    3. 白化后 SVD: W @ L = U @ S @ V^T
    4. 逆变换: V' = V @ L^(-1)
    
    这是 SVD-LLM 论文中的核心方法，比 Fisher-Hessian 方法更稳定高效
    """
    
    def __init__(self, damp: float = 1e-6):
        """
        Args:
            damp: 阻尼系数，用于数值稳定
        """
        self.damp = damp
    
    @torch.no_grad()
    def collect_scaling_matrices(self, 
                                  model, 
                                  calib_loader, 
                                  device: str,
                                  model_name: str) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        收集白化矩阵 (SVD-LLM 方式)
        
        步骤：
        1. 注册 hook 收集每个 Linear 层的输入协方差 X^T @ X
        2. 对协方差矩阵进行 Cholesky 分解得到白化矩阵 L
        
        Returns:
            scaling_mat: {layer_idx: {module_name: L_matrix}}
        """
        if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
            layers = model.model.layers
        elif "opt" in model_name:
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(device)
        print(f"\n{'='*70}")
        print(f"[C-GSVR] Collecting Whitening Matrices (SVD-LLM Style)...")
        print(f"{'='*70}")
        
        # Hook 函数：累积输入协方差 X^T @ X
        def make_hook(module):
            def hook(m, inp, out):
                x = inp[0].detach().float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                # X^T @ X: [in_dim, in_dim]
                adds = torch.matmul(x.transpose(1, 2), x)
                adds_sum = torch.sum(adds, dim=0)
                m.raw_scaling_diag_matrix += adds_sum
                del x, adds, adds_sum
                torch.cuda.empty_cache()
            return hook
        
        # 注册 hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.raw_scaling_diag_matrix = 0
                h = module.register_forward_hook(make_hook(module))
                hooks.append(h)
        
        # 前向传播收集统计信息
        for batch in tqdm(calib_loader, desc="Collecting input covariance"):
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
                model(**batch)
            else:
                if isinstance(batch, (tuple, list)):
                    input_ids = batch[0].to(device)
                else:
                    input_ids = batch.to(device)
                attention_mask = torch.ones_like(input_ids)
                model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 清理 hooks
        for h in hooks:
            h.remove()
        
        torch.cuda.empty_cache()
        model = model.cpu()
        
        # 将原始协方差矩阵移到 CPU
        for i in range(len(layers)):
            subset = find_layers(layers[i])
            for name in subset:
                if hasattr(subset[name], 'raw_scaling_diag_matrix'):
                    subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
        
        # Cholesky 分解
        scaling_mat = {}
        print("Start Cholesky Decomposition...")
        for i in tqdm(range(len(layers)), desc="Cholesky decomposition"):
            layer_scaling = {}
            subset = find_layers(layers[i])
            for name in subset:
                if not hasattr(subset[name], 'raw_scaling_diag_matrix'):
                    continue
                    
                raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(device)
                
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                except Exception as e:
                    print(f"Warning: Cholesky failed for layer {i}.{name}, adding regularization")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    raw_scaling_diag_matrix += (-eigenvalues[0] + self.damp) * torch.eye(
                        raw_scaling_diag_matrix.shape[0], device=device
                    )
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                
                layer_scaling[name] = scaling_diag_matrix.cpu()
                
                # 清理
                del subset[name].raw_scaling_diag_matrix
                del raw_scaling_diag_matrix, scaling_diag_matrix
                torch.cuda.empty_cache()
            
            scaling_mat[i] = layer_scaling
        
        torch.cuda.empty_cache()
        print(f"Whitening matrix collection completed for {len(scaling_mat)} layers.\n")
        return scaling_mat


class WhitenedSVDComputer:
    """
    白化 SVD 计算器 (SVD-LLM 风格)
    
    对 W @ L 进行 SVD 分解，其中 L 是 Cholesky 分解得到的白化矩阵
    """
    
    def __init__(self, damp: float = 1e-6):
        self.damp = damp
    
    def whitened_svd(self,
                     W: torch.Tensor,
                     scaling_matrix: torch.Tensor,
                     device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行白化后的 SVD
        
        W_scaled = W @ L
        W_scaled = U @ S @ V^T
        V_final = V @ L^(-1)  (逆变换到原始空间)
        
        Args:
            W: 原始权重矩阵 [out_dim, in_dim]
            scaling_matrix: Cholesky 分解得到的白化矩阵 L [in_dim, in_dim]
            device: 计算设备
            
        Returns:
            U, S, Vt: SVD 分解结果（Vt 已经逆变换到原始空间）
            scaling_matrix: 白化矩阵（用于后续参考）
        """
        W = W.float().to(device)
        scaling_matrix = scaling_matrix.float().to(device)
        dtype = W.dtype
        
        # 计算白化矩阵的逆
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_matrix)
        except Exception as e:
            print("Warning: scaling_matrix is not full rank, adding regularization")
            scaling_matrix += self.damp * torch.eye(scaling_matrix.shape[0], device=device)
            scaling_matrix_inv = torch.linalg.inv(scaling_matrix)
        
        # 白化: W @ L
        W_scaled = torch.matmul(W, scaling_matrix)
        
        # SVD 分解
        try:
            U, S, Vt = torch.linalg.svd(W_scaled, full_matrices=False)
        except RuntimeError as e:
            print(f"Warning: SVD failed: {e}, using unweighted SVD")
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            return U, S, Vt, torch.eye(W.size(1), device=device, dtype=dtype)
        
        # 逆变换: V' = V @ L^(-1)
        # 注意: Vt 是 V 的转置，所以 Vt_final = Vt @ L^(-1)
        Vt_final = torch.matmul(Vt, scaling_matrix_inv)
        
        return U, S, Vt_final, scaling_matrix


class MarginalUtilityRankAllocator:
    """
    全局边际效用秩分配器
    
    基于边际效用等价原理，将参数预算最优分配到各层
    """
    
    def __init__(self):
        self.global_utility_pool: List[MarginalUtilityEntry] = []
    
    def build_global_utility_table(self,
                                   sigma_dict: Dict[str, torch.Tensor],
                                   shapes_dict: Dict[str, Tuple[int, int]],
                                   layer_module_map: Dict[str, Tuple[int, str]]) -> None:
        """
        构建全局边际效用表
        
        对每一层每个模块的每个奇异值，计算其边际效用:
        MU(l, k) = σ_k² / (out_dim + in_dim)
        
        Args:
            sigma_dict: {global_name: sigma_tensor} 奇异值字典
            shapes_dict: {global_name: (out_dim, in_dim)} 形状字典
            layer_module_map: {global_name: (layer_idx, module_name)} 映射
        """
        self.global_utility_pool = []
        
        for global_name, sigma in sigma_dict.items():
            out_dim, in_dim = shapes_dict[global_name]
            layer_idx, module_name = layer_module_map[global_name]
            param_cost = out_dim + in_dim  # 增加一个秩的参数成本
            
            for k, s in enumerate(sigma):
                s_val = s.item() if isinstance(s, torch.Tensor) else s
                weighted_sigma_sq = s_val ** 2
                
                # 边际效用 = 重建误差下降量 / 参数成本
                marginal_utility = weighted_sigma_sq / param_cost
                
                entry = MarginalUtilityEntry(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    rank_idx=k,
                    marginal_utility=marginal_utility,
                    weighted_sigma_sq=weighted_sigma_sq,
                    param_cost=param_cost
                )
                self.global_utility_pool.append(entry)
        
        # 按边际效用降序排列
        self.global_utility_pool.sort(key=lambda x: -x.marginal_utility)
        
        print(f"[MarginalUtility] Built utility table with {len(self.global_utility_pool)} entries")
    
    def allocate_ranks(self,
                       total_original_params: int,
                       target_ratio: float) -> Dict[str, int]:
        """
        严格压缩率约束下的秩分配
        
        贪心选择边际效用最高的奇异值，直到达到参数预算
        
        Args:
            total_original_params: 原始总参数量
            target_ratio: 目标参数保留率
            
        Returns:
            rank_allocation: {global_name: allocated_rank}
        """
        target_budget = int(total_original_params * target_ratio)
        
        print(f"\n{'='*70}")
        print(f"[C-GSVR] Global Rank Allocation (Greedy Selection)")
        print(f"{'='*70}")
        print(f"Original params: {total_original_params:,}")
        print(f"Target budget: {target_budget:,} ({target_ratio:.2%})")
        
        # 初始化：每个模块秩为0
        rank_allocation = {}  # {global_name: rank}
        current_budget = 0
        
        # 贪心选择
        for entry in self.global_utility_pool:
            global_name = f"layer_{entry.layer_idx}.{entry.module_name}"
            
            if global_name not in rank_allocation:
                rank_allocation[global_name] = 0
            
            # 检查是否可以增加这个秩
            if current_budget + entry.param_cost <= target_budget:
                # 确保秩是连续的（不能跳过）
                if entry.rank_idx == rank_allocation[global_name]:
                    rank_allocation[global_name] += 1
                    current_budget += entry.param_cost
        
        # 统计
        print(f"\nAllocated budget: {current_budget:,} ({current_budget/total_original_params:.2%})")
        print(f"Budget error: {abs(current_budget - target_budget)/target_budget*100:.2f}%")
        
        # 打印每层分配
        layer_ranks = {}
        for name, rank in rank_allocation.items():
            parts = name.split('.')
            layer_idx = int(parts[0].replace('layer_', ''))
            if layer_idx not in layer_ranks:
                layer_ranks[layer_idx] = {}
            module_name = '.'.join(parts[1:])
            layer_ranks[layer_idx][module_name] = rank
        
        print(f"\nRank allocation per layer:")
        for layer_idx in sorted(layer_ranks.keys())[:5]:
            print(f"  Layer {layer_idx}: {layer_ranks[layer_idx]}")
        if len(layer_ranks) > 5:
            print(f"  ... ({len(layer_ranks) - 5} more layers)")
        
        return rank_allocation

    def allocate_fixed_ranks(self,
                              total_original_params: int,
                              target_ratio: float,
                              shapes_dict: Dict[str, Tuple[int, int]],
                              fixed_rank_value: int = None) -> Dict[str, int]:
        """
        固定秩分配：所有层使用相同的秩
        
        如果未指定 fixed_rank_value，则根据目标压缩率计算一个合适的固定秩值。
        
        Args:
            total_original_params: 原始总参数量
            target_ratio: 目标参数保留率
            shapes_dict: {global_name: (out_dim, in_dim)} 形状字典
            fixed_rank_value: 固定秩值（可选，如不指定则自动计算）
            
        Returns:
            rank_allocation: {global_name: allocated_rank}
        """
        target_budget = int(total_original_params * target_ratio)
        
        print(f"\n{'='*70}")
        print(f"[C-GSVR] Fixed Rank Allocation")
        print(f"{'='*70}")
        print(f"Original params: {total_original_params:,}")
        print(f"Target budget: {target_budget:,} ({target_ratio:.2%})")
        
        if fixed_rank_value is not None:
            # 使用用户指定的固定秩值
            fixed_rank = fixed_rank_value
            print(f"Using user-specified fixed rank: {fixed_rank}")
        else:
            # 根据目标压缩率自动计算固定秩
            # 计算方法：假设所有层使用相同的秩 r，则
            # total_compressed = sum_{all layers} r * (out_dim + in_dim)
            # 我们需要找到满足 total_compressed <= target_budget 的最大 r
            
            total_cost_per_rank = sum(out_dim + in_dim for out_dim, in_dim in shapes_dict.values())
            fixed_rank = max(1, target_budget // total_cost_per_rank)
            print(f"Auto-computed fixed rank: {fixed_rank}")
            print(f"  (based on {len(shapes_dict)} modules, cost per rank: {total_cost_per_rank:,})")
        
        # 为所有模块分配相同的秩，但需要确保不超过模块的最小维度
        rank_allocation = {}
        current_budget = 0
        
        for global_name, (out_dim, in_dim) in shapes_dict.items():
            # 秩不能超过矩阵的最小维度
            max_rank = min(out_dim, in_dim)
            actual_rank = min(fixed_rank, max_rank)
            rank_allocation[global_name] = actual_rank
            current_budget += actual_rank * (out_dim + in_dim)
        
        # 统计
        print(f"\nFixed rank applied: {fixed_rank}")
        print(f"Allocated budget: {current_budget:,} ({current_budget/total_original_params:.2%})")
        print(f"Budget error: {abs(current_budget - target_budget)/target_budget*100:.2f}%")
        
        # 打印每层分配示例
        layer_ranks = {}
        for name, rank in rank_allocation.items():
            parts = name.split('.')
            layer_idx = int(parts[0].replace('layer_', ''))
            if layer_idx not in layer_ranks:
                layer_ranks[layer_idx] = {}
            module_name = '.'.join(parts[1:])
            layer_ranks[layer_idx][module_name] = rank
        
        print(f"\nRank allocation per layer (fixed):")
        for layer_idx in sorted(layer_ranks.keys())[:3]:
            print(f"  Layer {layer_idx}: {layer_ranks[layer_idx]}")
        if len(layer_ranks) > 3:
            print(f"  ... ({len(layer_ranks) - 3} more layers with same fixed rank)")
        
        return rank_allocation


class RecursiveErrorCompensator:
    """
    递归误差吸收器
    
    利用白化矩阵信息，将当前层的截断误差压入下一层权重
    
    理论：
    Δ_l = W_l @ X - W'_l @ X  (当前层误差)
    W_{l+1}' = W_{l+1} + H_{l+1}^(-1) @ Δ_l^T @ ...
    """
    
    def __init__(self, damp: float = 0.01):
        self.damp = damp
    
    def compute_layer_error(self,
                            W_original: torch.Tensor,
                            U_trunc: torch.Tensor,
                            S_trunc: torch.Tensor,
                            Vt_trunc: torch.Tensor,
                            X_calib: torch.Tensor) -> torch.Tensor:
        """
        计算层压缩误差
        
        Δ = (W - W') @ X = W @ X - U @ Σ' @ V^T @ X
        
        Args:
            W_original: 原始权重 [out_dim, in_dim]
            U_trunc, S_trunc, Vt_trunc: 截断后的 SVD
            X_calib: 校准输入 [batch, seq, in_dim]
            
        Returns:
            delta: 误差矩阵 [batch, seq, out_dim]
        """
        # 原始输出
        Y_original = X_calib @ W_original.T  # [batch, seq, out_dim]
        
        # 压缩后输出
        sqrt_S = torch.sqrt(S_trunc)
        W_compressed = (U_trunc @ torch.diag(S_trunc)) @ Vt_trunc
        Y_compressed = X_calib @ W_compressed.T
        
        delta = Y_original - Y_compressed
        return delta
    
    def compensate_next_layer(self,
                              delta: torch.Tensor,
                              W_next: torch.Tensor,
                              H_next: torch.Tensor,
                              X_next: torch.Tensor) -> torch.Tensor:
        """
        补偿下一层权重以吸收误差
        
        W'_{l+1} = W_{l+1} - H_{l+1}^(-1) @ X^T @ Δ / N
        
        Args:
            delta: 当前层误差 [batch, seq, out_dim] = [batch, seq, in_dim_next]
            W_next: 下一层原始权重 [out_dim_next, in_dim_next]
            H_next: 下一层 Hessian [in_dim_next, in_dim_next]
            X_next: 下一层输入（通常就是当前层输出）
            
        Returns:
            W_compensated: 补偿后的权重
        """
        device = W_next.device
        dtype = W_next.dtype
        
        # 计算 Hessian 逆（使用简单的阻尼逆）
        H_next = H_next.to(device).float()
        n = H_next.size(0)
        damp_val = max(self.damp * H_next.diag().abs().mean().item(), 1e-6)
        H_damped = H_next + damp_val * torch.eye(n, device=device)
        try:
            H_inv = torch.linalg.inv(H_damped)
        except:
            H_inv = torch.linalg.pinv(H_damped)
        
        # 计算补偿量
        # Δ: [batch, seq, in_dim_next]
        # X_next: [batch, seq, in_dim_next]
        # 补偿: ΔW = H^(-1) @ (X^T @ Δ) / N
        
        delta_flat = delta.view(-1, delta.shape[-1])  # [N, in_dim_next]
        X_flat = X_next.view(-1, X_next.shape[-1])    # [N, in_dim_next]
        
        # 这里使用简化的补偿：对权重施加小的修正
        # 完整的 OBS 风格补偿需要更复杂的推导
        N = delta_flat.shape[0]
        grad_approx = (delta_flat.T @ X_flat) / N  # [in_dim_next, in_dim_next]
        
        # 投影到权重空间
        # 简化：直接对 W 的行进行调整
        correction = H_inv @ grad_approx.T @ W_next.float()  # 近似
        
        W_compensated = W_next.float() - 0.1 * correction.T  # 使用小步长避免过度补偿
        
        return W_compensated.to(dtype)


class CGSVRCompressor:
    """
    C-GSVR 主压缩器
    
    完整流程:
    1. 收集白化矩阵 (SVD-LLM 风格，使用 Cholesky 分解)
    2. 白化后 SVD 分解
    3. 严格压缩率约束下的秩分配 (贪心选择)
    4. 顺序压缩与跨层误差补偿
    5. 模型重构
    """
    
    def __init__(self,
                 damp: float = 0.01,
                 use_fisher: bool = True,
                 use_compensation: bool = False,
                 compensation_strength: float = 0.1,
                 fixed_rank: bool = False,
                 fixed_rank_value: int = None):
        """
        Args:
            damp: 阻尼系数
            use_fisher: 是否使用 Fisher 信息（保留参数，但现在使用 SVD-LLM 白化）
            use_compensation: 是否启用误差补偿
            compensation_strength: 误差补偿强度
            fixed_rank: 是否使用固定秩（所有层使用相同的秩）
            fixed_rank_value: 固定秩的值（仅当 fixed_rank=True 时使用）
        """
        self.damp = damp
        self.use_fisher = use_fisher
        self.use_compensation = use_compensation
        self.compensation_strength = compensation_strength
        self.fixed_rank = fixed_rank
        self.fixed_rank_value = fixed_rank_value
        
        # 使用 SVD-LLM 风格的白化
        self.whitening_computer = WhiteningComputer(damp)
        self.svd_computer = WhitenedSVDComputer(damp)
        self.rank_allocator = MarginalUtilityRankAllocator()
        self.error_compensator = RecursiveErrorCompensator(damp)
    
    def compress(self,
                 model,
                 calib_loader,
                 target_ratio: float,
                 device: str,
                 model_name: str,
                 checkpoint_path: str = None) -> nn.Module:
        """
        执行 C-GSVR 压缩
        
        Args:
            model: 待压缩的模型
            calib_loader: 校准数据加载器
            target_ratio: 目标参数保留率
            device: 计算设备
            model_name: 模型名称
            checkpoint_path: 中间结果保存路径 (Optional)
            
        Returns:
            compressed_model: 压缩后的模型
        """
        print(f"\n{'='*70}")
        print(f"C-GSVR: Compensated Global Semantic Variable-Rank Compression")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Target ratio: {target_ratio:.2%}")
        print(f"Whitening Method: SVD-LLM (Cholesky)")
        print(f"Use Compensation: {self.use_compensation}")
        print(f"Rank Mode: {'Fixed' if self.fixed_rank else 'Variable (Marginal Utility)'}")
        if self.fixed_rank and self.fixed_rank_value:
            print(f"Fixed Rank Value: {self.fixed_rank_value}")
        if checkpoint_path:
            print(f"Checkpoint path: {checkpoint_path}")
            os.makedirs(checkpoint_path, exist_ok=True)
        print(f"{'='*70}\n")
        
        # ========== Step 1: 收集白化矩阵 (SVD-LLM Style) ==========
        print(f"\n[Step 1/5] Collecting Whitening Matrices (SVD-LLM Style)...")
        
        scaling_mat = None
        stats_loaded = False
        
        if checkpoint_path:
            stats_ckpt = os.path.join(checkpoint_path, "whitening_checkpoint.pt")
            if os.path.exists(stats_ckpt):
                print(f"Loading whitening matrices from {stats_ckpt}...")
                try:
                    stats_data = torch.load(stats_ckpt, map_location='cpu')
                    scaling_mat = stats_data['scaling_mat']
                    stats_loaded = True
                    print("Whitening matrices loaded successfully!")
                except Exception as e:
                    print(f"Failed to load whitening matrices: {e}")
        
        if not stats_loaded:
            scaling_mat = self.whitening_computer.collect_scaling_matrices(
                model, calib_loader, device, model_name
            )
            
            if checkpoint_path:
                stats_ckpt = os.path.join(checkpoint_path, "whitening_checkpoint.pt")
                print(f"Saving whitening matrices to {stats_ckpt}...")
                torch.save({
                    'scaling_mat': scaling_mat
                }, stats_ckpt)
        
        # ========== Step 2: 白化后 SVD ==========
        print(f"\n[Step 2/5] Computing Whitened SVD...")
        if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
            layers = model.model.layers
        elif "opt" in model_name:
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        svd_loaded = False
        sigma_dict = {}
        shapes_dict = {}
        svd_cache = {}  # {global_name: (U, S, Vt, scaling_matrix)}
        layer_module_map = {}
        total_original_params = 0
        
        if checkpoint_path:
            svd_ckpt = os.path.join(checkpoint_path, "svd_whitened_checkpoint.pt")
            if os.path.exists(svd_ckpt):
                print(f"Loading SVD results from {svd_ckpt}...")
                try:
                    svd_data = torch.load(svd_ckpt, map_location='cpu')
                    sigma_dict = svd_data['sigma_dict']
                    shapes_dict = svd_data['shapes_dict']
                    svd_cache = svd_data['svd_cache']
                    layer_module_map = svd_data['layer_module_map']
                    total_original_params = svd_data['total_original_params']
                    svd_loaded = True
                    print("SVD results loaded successfully!")
                except Exception as e:
                    print(f"Failed to load SVD results: {e}")
        
        if not svd_loaded:
            for i in tqdm(range(len(layers)), desc="SVD computation"):
                layer = layers[i]
                layer_scaling = scaling_mat[i]
                
                for name, module in layer.named_modules():
                    if isinstance(module, nn.Linear) and name in layer_scaling:
                        W = module.weight.data
                        scaling_matrix = layer_scaling[name]
                        
                        global_name = f"layer_{i}.{name}"
                        
                        # 执行白化后 SVD
                        U, S, Vt, _ = self.svd_computer.whitened_svd(
                            W, scaling_matrix, device
                        )
                        
                        # 保存结果
                        sigma_dict[global_name] = S.cpu()
                        shapes_dict[global_name] = W.shape
                        svd_cache[global_name] = (U.cpu(), S.cpu(), Vt.cpu())
                        layer_module_map[global_name] = (i, name)
                        total_original_params += W.numel()
                
                torch.cuda.empty_cache()
            
            if checkpoint_path:
                svd_ckpt = os.path.join(checkpoint_path, "svd_whitened_checkpoint.pt")
                print(f"Saving SVD results to {svd_ckpt}...")
                torch.save({
                    'sigma_dict': sigma_dict,
                    'shapes_dict': shapes_dict,
                    'svd_cache': svd_cache,
                    'layer_module_map': layer_module_map,
                    'total_original_params': total_original_params
                }, svd_ckpt)
        
        # ========== Step 3: 全局秩分配 ==========
        print(f"\n[Step 3/5] Global Rank Allocation...")
        
        if self.fixed_rank:
            # 固定秩模式：所有层使用相同的秩
            rank_allocation = self.rank_allocator.allocate_fixed_ranks(
                total_original_params, target_ratio, shapes_dict, self.fixed_rank_value
            )
        else:
            # 可变秩模式：基于边际效用的自适应秩分配
            self.rank_allocator.build_global_utility_table(
                sigma_dict, shapes_dict, layer_module_map
            )
            rank_allocation = self.rank_allocator.allocate_ranks(
                total_original_params, target_ratio
            )
        
        # ========== Step 4: 顺序压缩与跨层误差补偿 ==========
        print(f"\n[Step 4/5] Sequential Compression & Cross-layer Compensation...")
        compressed_modules = {}
        
        modules_loaded = False
        if checkpoint_path:
            comp_str = "comp" if self.use_compensation else "nocomp"
            modules_ckpt = os.path.join(checkpoint_path, f"compressed_modules_{target_ratio}_{comp_str}.pt")
            if os.path.exists(modules_ckpt):
                print(f"Loading compressed modules from {modules_ckpt}...")
                try:
                    compressed_modules = torch.load(modules_ckpt, map_location='cpu')
                    modules_loaded = True
                    print("Compressed modules loaded successfully!")
                except Exception as e:
                    print(f"Failed to load compressed modules: {e}")
        
        if not modules_loaded:
            # 收集校准输入用于误差补偿（如果启用）
            layer_inputs = {}
            if self.use_compensation:
                layer_inputs = self._collect_layer_inputs(model, layers, calib_loader, device)
            
            # 存储每层的累积误差，用于传递给下一层
            accumulated_error = None
            
            for i in tqdm(range(len(layers)), desc="Compressing with compensation"):
                layer = layers[i]
                layer_error = {}  # 本层各模块的误差
                
                for name, module in layer.named_modules():
                    if isinstance(module, nn.Linear):
                        global_name = f"layer_{i}.{name}"
                        
                        if global_name not in svd_cache:
                            continue
                        
                        rank = rank_allocation.get(global_name, 0)
                        
                        if rank > 0:
                            # SVD-LLM 风格：SVD 已经在白化空间完成并逆变换回原始空间
                            U, S, Vt = svd_cache[global_name]
                            W_original = module.weight.data.clone()
                            
                            # 截断到分配的秩
                            U_trunc = U[:, :rank].to(device)
                            S_trunc = S[:rank].to(device)
                            Vt_trunc = Vt[:rank, :].to(device)
                            
                            # 计算本层压缩误差（用于传递给下一层）
                            if self.use_compensation and i < len(layers) - 1:
                                W_compressed = (U_trunc @ torch.diag(S_trunc)) @ Vt_trunc
                                delta_W = W_original.to(device).float() - W_compressed
                                layer_error[name] = delta_W.cpu()
                            
                            compressed_modules[global_name] = (
                                U_trunc.cpu(), S_trunc.cpu(), Vt_trunc.cpu()
                            )
                        else:
                            compressed_modules[global_name] = None
                        
                        # 释放 SVD 缓存
                        if global_name in svd_cache:
                            del svd_cache[global_name]
                
                # ===== 跨层误差补偿：将本层误差传递给下一层 =====
                if self.use_compensation and i < len(layers) - 1 and layer_error:
                    next_layer = layers[i + 1]
                    next_layer_scaling = scaling_mat.get(i + 1, {})
                    
                    # 对下一层的权重进行补偿
                    self._apply_error_compensation_whitening(
                        next_layer, layer_error, next_layer_scaling, 
                        layer_inputs.get(i + 1, None), device
                    )
                
                torch.cuda.empty_cache()
            
            if checkpoint_path:
                comp_str = "comp" if self.use_compensation else "nocomp"
                modules_ckpt = os.path.join(checkpoint_path, f"compressed_modules_{target_ratio}_{comp_str}.pt")
                print(f"Saving compressed modules to {modules_ckpt}...")
                torch.save(compressed_modules, modules_ckpt)
        
        # ========== Step 5: 模型重构 ==========
        print(f"\n[Step 5/5] Model Reconstruction...")
        model = self._replace_layers(model, layers, compressed_modules, model_name)
        
        # 统计压缩结果
        total_compressed_params = 0
        for name, result in compressed_modules.items():
            if result is not None:
                U, S, Vt = result
                out_dim = U.shape[0]
                in_dim = Vt.shape[1]
                rank = len(S)
                total_compressed_params += rank * (out_dim + in_dim)
        
        print(f"\n{'='*70}")
        print(f"C-GSVR Compression Summary")
        print(f"{'='*70}")
        print(f"Original parameters: {total_original_params:,}")
        print(f"Compressed parameters: {total_compressed_params:,}")
        print(f"Actual compression ratio: {total_compressed_params/total_original_params:.2%}")
        print(f"Target ratio: {target_ratio:.2%}")
        print(f"Ratio error: {abs(total_compressed_params/total_original_params - target_ratio)/target_ratio*100:.2f}%")
        print(f"{'='*70}\n")
        
        return model
    
    @torch.no_grad()
    def _collect_layer_inputs(self, model, layers, calib_loader, device):
        """
        收集每层的输入用于误差补偿
        
        Returns:
            layer_inputs: {layer_idx: input_tensor}
        """
        layer_inputs = {}
        
        # 为每层注册 hook 收集输入
        def make_input_hook(layer_idx):
            def hook(module, inp, out):
                if layer_idx not in layer_inputs:
                    x = inp[0].detach()
                    # 只保存第一个 batch 的输入（节省内存）
                    layer_inputs[layer_idx] = x.cpu()
            return hook
        
        hooks = []
        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(make_input_hook(i))
            hooks.append(h)
        
        # 只用一个 batch 收集
        for batch in calib_loader:
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(device)
            elif isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            else:
                input_ids = batch.to(device)
            
            attention_mask = torch.ones_like(input_ids)
            model(input_ids=input_ids, attention_mask=attention_mask)
            break  # 只用一个 batch
        
        # 清理 hooks
        for h in hooks:
            h.remove()
        
        return layer_inputs
    
    def _apply_error_compensation(self, next_layer, layer_error, next_layer_h, 
                                   next_layer_input, device):
        """
        将当前层的误差补偿到下一层的权重
        
        基于 OBS (Optimal Brain Surgeon) 思想：
        W_{l+1}' = W_{l+1} - α * H_{l+1}^(-1) @ ∂E/∂W_{l+1}
        
        其中误差传播近似为：∂E/∂W_{l+1} ≈ Δ_l^T @ X_{l+1} / N
        
        Args:
            next_layer: 下一层模块
            layer_error: 当前层各模块的权重误差 {name: delta_W}
            next_layer_h: 下一层的 Hessian 字典
            next_layer_input: 下一层的输入
            device: 计算设备
        """
        if next_layer_input is None:
            return
        
        # 计算当前层对输出的总体影响（简化：使用 o_proj 和 down_proj 的误差）
        # 因为这两个是各自子模块的输出层，对下一层影响最大
        key_modules = ['self_attn.o_proj', 'mlp.down_proj']
        
        total_error_effect = None
        for key in key_modules:
            # 找到匹配的误差
            for name, delta_W in layer_error.items():
                if key.split('.')[-1] in name:
                    if total_error_effect is None:
                        total_error_effect = delta_W.to(device)
                    else:
                        # 累加误差影响（这是一个近似）
                        if total_error_effect.shape == delta_W.shape:
                            total_error_effect = total_error_effect + delta_W.to(device)
        
        if total_error_effect is None:
            return
        
        # 对下一层的输入投影层进行补偿
        # 下一层的输入层：self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, mlp.gate_proj, mlp.up_proj
        input_layer_names = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
        
        for name, module in next_layer.named_modules():
            if isinstance(module, nn.Linear) and any(n in name for n in input_layer_names):
                H = next_layer_h.get(name, None)
                if H is None:
                    continue
                
                H = H.to(device).float()
                W = module.weight.data.to(device).float()
                
                # 计算 Hessian 逆（带阻尼）
                n = H.size(0)
                damp_val = max(0.01 * H.diag().abs().mean().item(), 1e-6)
                H_damped = H + damp_val * torch.eye(n, device=device)
                
                try:
                    H_inv = torch.linalg.inv(H_damped)
                except:
                    # 使用伪逆
                    H_inv = torch.linalg.pinv(H_damped)
                
                # 计算补偿量
                # 简化的补偿：基于误差矩阵的 Frobenius 范数缩放
                error_scale = total_error_effect.norm() / (W.norm() + 1e-8)
                
                # 补偿方向：H^(-1) 作用于权重的行
                # ΔW ≈ α * error_scale * H^(-1) @ W^T @ W / ||W||
                compensation = self.compensation_strength * error_scale * (H_inv @ W.T).T
                
                # 应用补偿（限制补偿幅度）
                max_change = 0.1 * W.abs().mean()
                compensation = torch.clamp(compensation, -max_change, max_change)
                
                # 更新权重
                module.weight.data = (W - compensation).to(module.weight.dtype)
    
    def _apply_error_compensation_whitening(self, next_layer, layer_error, next_layer_scaling, 
                                             next_layer_input, device):
        """
        将当前层的误差补偿到下一层的权重 (SVD-LLM 白化版本)
        
        Args:
            next_layer: 下一层模块
            layer_error: 当前层各模块的权重误差 {name: delta_W}
            next_layer_scaling: 下一层的白化矩阵字典
            next_layer_input: 下一层的输入
            device: 计算设备
        """
        if next_layer_input is None:
            return
        
        # 计算当前层对输出的总体影响
        key_modules = ['self_attn.o_proj', 'mlp.down_proj']
        
        total_error_effect = None
        for key in key_modules:
            for name, delta_W in layer_error.items():
                if key.split('.')[-1] in name:
                    if total_error_effect is None:
                        total_error_effect = delta_W.to(device)
                    else:
                        if total_error_effect.shape == delta_W.shape:
                            total_error_effect = total_error_effect + delta_W.to(device)
        
        if total_error_effect is None:
            return
        
        # 对下一层的输入投影层进行补偿
        input_layer_names = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']
        
        for name, module in next_layer.named_modules():
            if isinstance(module, nn.Linear) and any(n in name for n in input_layer_names):
                scaling = next_layer_scaling.get(name, None)
                if scaling is None:
                    continue
                
                scaling = scaling.to(device).float()
                W = module.weight.data.to(device).float()
                
                # 使用白化矩阵计算逆
                try:
                    scaling_inv = torch.linalg.inv(scaling)
                    H_approx_inv = scaling_inv @ scaling_inv.T
                except:
                    H_approx_inv = torch.eye(scaling.shape[0], device=device)
                
                # 计算补偿量
                error_scale = total_error_effect.norm() / (W.norm() + 1e-8)
                compensation = self.compensation_strength * error_scale * (H_approx_inv @ W.T).T
                
                # 限制补偿幅度
                max_change = 0.1 * W.abs().mean()
                compensation = torch.clamp(compensation, -max_change, max_change)
                
                # 更新权重
                module.weight.data = (W - compensation).to(module.weight.dtype)
    
    def _replace_layers(self, model, layers, compressed_modules, model_name):
        """替换模型中的压缩层"""
        from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
        
        for i in tqdm(range(len(layers)), desc="Replacing layers"):
            layer = layers[i]
            
            # 替换 Attention
            attn_names = ['self_attn.q_proj', 'self_attn.k_proj', 
                          'self_attn.v_proj', 'self_attn.o_proj']
            attn_compressed = {name: compressed_modules.get(f"layer_{i}.{name}") 
                               for name in attn_names}
            
            if all(v is not None for v in attn_compressed.values()):
                self._replace_attention(layer, attn_compressed, i)
            
            # 替换 MLP
            mlp_names = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
            mlp_compressed = {name: compressed_modules.get(f"layer_{i}.{name}")
                              for name in mlp_names}
            
            if all(v is not None for v in mlp_compressed.values()):
                self._replace_mlp(layer, mlp_compressed)
        
        return model
    
    def _replace_attention(self, layer, compressed, layer_idx):
        """替换 Attention 模块"""
        from component.svd_llama import SVD_LlamaAttention
        
        config = layer.self_attn.config
        original_dtype = layer.self_attn.q_proj.weight.dtype
        original_device = layer.self_attn.q_proj.weight.device
        
        svd_attn = SVD_LlamaAttention(config, ratio=1.0, layer_idx=layer_idx)
        
        for proj in ['q', 'k', 'v', 'o']:
            module_name = f'self_attn.{proj}_proj'
            U, S, Vt = compressed[module_name]
            
            rank = len(S)
            out_dim = U.shape[0]
            in_dim = Vt.shape[1]
            
            v_proj = nn.Linear(in_dim, rank, bias=False)
            u_proj = nn.Linear(rank, out_dim, bias=False)
            
            S_safe = torch.clamp(S, min=1e-8)
            sqrt_S = torch.sqrt(S_safe)
            sqrt_S_diag = torch.diag(sqrt_S)
            
            v_weight = sqrt_S_diag @ Vt
            u_weight = U @ sqrt_S_diag
            
            v_proj.weight.data = v_weight.to(dtype=original_dtype, device=original_device)
            u_proj.weight.data = u_weight.to(dtype=original_dtype, device=original_device)
            
            setattr(svd_attn, f'{proj}_v_proj', v_proj)
            setattr(svd_attn, f'{proj}_u_proj', u_proj)
        
        svd_attn = svd_attn.to(dtype=original_dtype, device=original_device)
        layer.self_attn = svd_attn
    
    def _replace_mlp(self, layer, compressed):
        """替换 MLP 模块"""
        from component.svd_llama import SVD_LlamaMLP
        
        hidden_size = layer.mlp.gate_proj.weight.shape[1]
        intermediate_size = layer.mlp.gate_proj.weight.shape[0]
        hidden_act = getattr(layer.mlp.config, 'hidden_act', 'silu')
        original_dtype = layer.mlp.gate_proj.weight.dtype
        original_device = layer.mlp.gate_proj.weight.device
        
        svd_mlp = SVD_LlamaMLP(hidden_size, intermediate_size, hidden_act, ratio=1.0)
        
        for proj in ['gate', 'up', 'down']:
            module_name = f'mlp.{proj}_proj'
            U, S, Vt = compressed[module_name]
            
            rank = len(S)
            out_dim = U.shape[0]
            in_dim = Vt.shape[1]
            
            v_proj = nn.Linear(in_dim, rank, bias=False)
            u_proj = nn.Linear(rank, out_dim, bias=False)
            
            S_safe = torch.clamp(S, min=1e-8)
            sqrt_S = torch.sqrt(S_safe)
            sqrt_S_diag = torch.diag(sqrt_S)
            
            v_weight = sqrt_S_diag @ Vt
            u_weight = U @ sqrt_S_diag
            
            v_proj.weight.data = v_weight.to(dtype=original_dtype, device=original_device)
            u_proj.weight.data = u_weight.to(dtype=original_dtype, device=original_device)
            
            setattr(svd_mlp, f'{proj}_v_proj', v_proj)
            setattr(svd_mlp, f'{proj}_u_proj', u_proj)
        
        svd_mlp = svd_mlp.to(dtype=original_dtype, device=original_device)
        layer.mlp = svd_mlp
