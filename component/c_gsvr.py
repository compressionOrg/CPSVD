"""
C-GSVR: Compensated Global Semantic Variable-Rank (补偿性全局语义变秩法)

核心思想：
1. 边际效用等价原理 - 每层增加一个秩的效用/成本比相等时全局最优
2. Fisher-Hessian 联合空间 - 双重加权的奇异值重要性评分
3. 递归误差吸收 - 利用下一层 Hessian 逆吸收当前层截断误差

主要优势：
- 严格保证全局压缩率（通过边际收益排序+硬截断）
- 层间自适应（好压的层多出预算，难压的层获得更多秩）
- 误差补偿（顺序压缩时让下一层消化前一层损失）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import heapq


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


class FisherHessianComputer:
    """
    Fisher-Hessian 联合信息计算器
    
    - Hessian (H): 输入协方差矩阵，反映输入能量分布
    - Fisher (F): 输出敏感度矩阵，反映对损失的影响
    
    联合空间变换: W̃ = F^(1/2) @ W @ H^(1/2)
    """
    
    def __init__(self, damp: float = 0.01, fisher_weight: float = 1.0):
        """
        Args:
            damp: 阻尼系数，用于数值稳定
            fisher_weight: Fisher 信息权重 (相对于 Hessian)
        """
        self.damp = damp
        self.fisher_weight = fisher_weight
    
    @torch.no_grad()
    def collect_hessian(self, 
                        model, 
                        calib_loader, 
                        device: str,
                        model_name: str) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        收集 Hessian 矩阵 (输入协方差)
        
        H_l = (1/N) * Σ X_l^T @ X_l
        
        Returns:
            h_mat: {layer_idx: {module_name: H_matrix}}
        """
        if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
            layers = model.model.layers
        elif "opt" in model_name:
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model = model.to(device)
        print(f"\n{'='*70}")
        print(f"[C-GSVR] Collecting Hessian (Input Covariance) Matrices...")
        print(f"{'='*70}")
        
        # 存储样本数
        sample_count = [0]
        
        def make_hook(module):
            def hook(m, inp, out):
                x = inp[0].detach().float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                # X^T @ X: [in_dim, in_dim]
                batch_size, seq_len = x.shape[0], x.shape[1]
                x_flat = x.view(-1, x.shape[-1])  # [batch*seq, in_dim]
                h_add = (x_flat.T @ x_flat).cpu()  # 立即移到 CPU 避免 GPU OOM
                
                if not hasattr(m, '_hessian_acc'):
                    m._hessian_acc = torch.zeros_like(h_add)
                    m._hessian_count = 0
                
                m._hessian_acc += h_add
                m._hessian_count += x_flat.shape[0]
                
                del x, x_flat, h_add
            return hook
        
        # 注册 hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_forward_hook(make_hook(module))
                hooks.append(h)
        
        # 前向传播
        for batch in tqdm(calib_loader, desc="Hessian forward pass"):
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(device)
            elif isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
            else:
                input_ids = batch.to(device)
            
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            sample_count[0] += 1
            
            # 定期清理内存
            if sample_count[0] % 16 == 0:
                torch.cuda.empty_cache()
        
        # 清理 hooks
        for h in hooks:
            h.remove()
        
        # 收集并归一化
        h_mat = {}
        for i, layer in enumerate(layers):
            layer_h = {}
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, '_hessian_acc'):
                    # 归一化（已经在 CPU 上）
                    H = module._hessian_acc / max(module._hessian_count, 1)
                    layer_h[name] = H
                    del module._hessian_acc
                    del module._hessian_count
            h_mat[i] = layer_h
        
        torch.cuda.empty_cache()
        print(f"Hessian collection completed for {len(h_mat)} layers.\n")
        return h_mat
    
    def collect_fisher(self,
                       model,
                       calib_loader,
                       device: str,
                       model_name: str,
                       use_diagonal: bool = True) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        收集 Fisher 信息矩阵 (输出敏感度)
        
        F_l = (1/N) * Σ (∂L/∂Y_l)^T @ (∂L/∂Y_l)
        
        为避免 GPU 内存溢出，使用对角近似：只保留对角线元素
        F_diag = (1/N) * Σ (∂L/∂Y_l)² 
        
        其中 Y_l = W_l @ X_l 是层 l 的输出
        
        Args:
            use_diagonal: 是否使用对角近似（推荐 True 以节省内存）
        
        Returns:
            f_mat: {layer_idx: {module_name: F_matrix}}
        """
        if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
            layers = model.model.layers
        elif "opt" in model_name:
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        print(f"\n{'='*70}")
        print(f"[C-GSVR] Collecting Fisher (Output Sensitivity) Matrices...")
        print(f"[C-GSVR] Using diagonal approximation: {use_diagonal}")
        print(f"{'='*70}")
        
        def make_grad_hook(module, use_diag):
            def hook(m, grad_input, grad_output):
                grad = grad_output[0]
                if grad is None:
                    return
                
                grad = grad.detach().float()
                if grad.dim() == 2:
                    grad = grad.unsqueeze(0)
                
                grad_flat = grad.view(-1, grad.shape[-1])  # [batch*seq, out_dim]
                
                if use_diag:
                    # 对角近似：只计算每个输出维度的梯度平方和
                    # f_diag = sum(grad^2, dim=0)  shape: [out_dim]
                    f_add = (grad_flat ** 2).sum(dim=0).cpu()  # 立即移到 CPU
                    
                    if not hasattr(m, '_fisher_diag'):
                        m._fisher_diag = torch.zeros_like(f_add)
                        m._fisher_count = 0
                    
                    m._fisher_diag += f_add
                    m._fisher_count += grad_flat.shape[0]
                else:
                    # 完整 Fisher（可能导致 OOM）
                    f_add = (grad_flat.T @ grad_flat).cpu()  # 立即移到 CPU
                    
                    if not hasattr(m, '_fisher_acc'):
                        m._fisher_acc = torch.zeros_like(f_add)
                        m._fisher_count = 0
                    
                    m._fisher_acc += f_add
                    m._fisher_count += grad_flat.shape[0]
                
                # 及时清理 GPU 内存
                del grad, grad_flat, f_add
            return hook
        
        # 注册反向 hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_full_backward_hook(make_grad_hook(module, use_diagonal))
                hooks.append(h)
        
        model.train()  # 需要启用梯度
        
        # 限制处理的样本数以节省内存
        max_fisher_samples = min(len(calib_loader), 256)  # 最多使用 256 个样本
        
        for idx, batch in enumerate(tqdm(calib_loader, desc="Fisher backward pass")):
            if idx >= max_fisher_samples:
                break
                
            if isinstance(batch, (tuple, list)):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device) if len(batch) > 1 else input_ids.clone()
            elif isinstance(batch, dict):
                input_ids = batch['input_ids'].to(device)
                labels = batch.get('labels', input_ids.clone()).to(device)
            else:
                input_ids = batch.to(device)
                labels = input_ids.clone()
            
            # 语言模型损失
            model.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # 每个 batch 后清理内存
            del outputs, loss, input_ids, labels
            torch.cuda.empty_cache()
        
        model.eval()
        
        # 清理 hooks
        for h in hooks:
            h.remove()
        
        # 收集并归一化
        f_mat = {}
        for i, layer in enumerate(layers):
            layer_f = {}
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    if use_diagonal and hasattr(module, '_fisher_diag'):
                        # 将对角向量转换为对角矩阵
                        f_diag = module._fisher_diag / max(module._fisher_count, 1)
                        # 创建对角矩阵
                        F = torch.diag(f_diag)
                        layer_f[name] = F
                        del module._fisher_diag
                        del module._fisher_count
                    elif hasattr(module, '_fisher_acc'):
                        F = module._fisher_acc / max(module._fisher_count, 1)
                        layer_f[name] = F
                        del module._fisher_acc
                        del module._fisher_count
            f_mat[i] = layer_f
        
        torch.cuda.empty_cache()
        print(f"Fisher collection completed for {len(f_mat)} layers.\n")
        return f_mat
    
    def compute_sqrt_matrix(self, M: torch.Tensor, damp: float = None) -> torch.Tensor:
        """
        计算矩阵的平方根 M^(1/2)
        使用特征分解: M = V @ Λ @ V^T => M^(1/2) = V @ Λ^(1/2) @ V^T
        
        对于对角矩阵，使用更高效的逐元素开方
        """
        if damp is None:
            damp = self.damp
        
        M = M.float()
        
        # 检查是否接近零
        if M.abs().max() < 1e-10:
            return torch.eye(M.size(0), device=M.device, dtype=M.dtype)
        
        n = M.size(0)
        
        # 检测是否为对角矩阵（更高效的处理）
        off_diag_norm = (M - torch.diag(M.diag())).abs().max()
        if off_diag_norm < 1e-8:
            # 对角矩阵：直接对对角线元素开方
            diag_vals = M.diag()
            damp_val = max(damp * diag_vals.abs().mean().item(), 1e-6)
            diag_sqrt = torch.sqrt(torch.clamp(diag_vals + damp_val, min=1e-6))
            return torch.diag(diag_sqrt)
        
        M = (M + M.T) / 2.0  # 确保对称
        
        # 添加阻尼
        diag_mean = M.diag().abs().mean()
        damp_val = max(damp * diag_mean.item(), 1e-6)
        M_damped = M + damp_val * torch.eye(n, device=M.device, dtype=M.dtype)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M_damped)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            M_sqrt = eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ eigenvectors.T
        except RuntimeError:
            print("Warning: Eigendecomposition failed, using identity")
            M_sqrt = torch.eye(n, device=M.device, dtype=M.dtype)
        
        return M_sqrt
    
    def compute_inv_sqrt_matrix(self, M: torch.Tensor, damp: float = None) -> torch.Tensor:
        """
        计算矩阵的逆平方根 M^(-1/2)
        
        对于对角矩阵，使用更高效的逐元素计算
        """
        if damp is None:
            damp = self.damp
        
        M = M.float()
        
        if M.abs().max() < 1e-10:
            return torch.eye(M.size(0), device=M.device, dtype=M.dtype)
        
        n = M.size(0)
        
        # 检测是否为对角矩阵
        off_diag_norm = (M - torch.diag(M.diag())).abs().max()
        if off_diag_norm < 1e-8:
            # 对角矩阵：直接对对角线元素求逆平方根
            diag_vals = M.diag()
            damp_val = max(damp * diag_vals.abs().mean().item(), 1e-6)
            diag_inv_sqrt = 1.0 / torch.sqrt(torch.clamp(diag_vals + damp_val, min=1e-6))
            return torch.diag(diag_inv_sqrt)
        
        M = (M + M.T) / 2.0
        
        diag_mean = M.diag().abs().mean()
        damp_val = max(damp * diag_mean.item(), 1e-6)
        M_damped = M + damp_val * torch.eye(n, device=M.device, dtype=M.dtype)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M_damped)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            M_inv_sqrt = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
        except RuntimeError:
            print("Warning: Inverse sqrt computation failed, using identity")
            M_inv_sqrt = torch.eye(n, device=M.device, dtype=M.dtype)
        
        return M_inv_sqrt


class WeightedSVDComputer:
    """
    加权 SVD 计算器
    
    对 W̃ = F^(1/2) @ W @ H^(1/2) 进行 SVD 分解
    得到语义重要性加权的奇异值
    """
    
    def __init__(self, damp: float = 0.01, use_fisher: bool = True):
        self.damp = damp
        self.use_fisher = use_fisher
        self.fh_computer = FisherHessianComputer(damp)
    
    def weighted_svd(self,
                     W: torch.Tensor,
                     H: torch.Tensor,
                     F: Optional[torch.Tensor] = None,
                     device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
        """
        执行加权 SVD
        
        W̃ = F^(1/2) @ W @ H^(1/2)
        W̃ = U @ Σ @ V^T
        
        Returns:
            U, S, Vt: SVD 分解结果
            H_sqrt, F_sqrt: 用于逆变换的矩阵
        """
        W = W.float().to(device)
        H = H.float().to(device)
        
        # 计算 H^(1/2)
        H_sqrt = self.fh_computer.compute_sqrt_matrix(H, self.damp)
        
        # Hessian 空间白化: W @ H^(1/2)
        W_h = W @ H_sqrt
        
        # 可选: Fisher 空间白化
        if self.use_fisher and F is not None:
            F = F.float().to(device)
            F_sqrt = self.fh_computer.compute_sqrt_matrix(F, self.damp)
            W_weighted = F_sqrt @ W_h
        else:
            F_sqrt = torch.eye(W.size(0), device=device, dtype=W.dtype)
            W_weighted = W_h
        
        # SVD 分解
        try:
            U, S, Vt = torch.linalg.svd(W_weighted, full_matrices=False)
        except RuntimeError as e:
            print(f"Warning: SVD failed: {e}, using unweighted SVD")
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            H_sqrt = torch.eye(W.size(1), device=device, dtype=W.dtype)
            F_sqrt = torch.eye(W.size(0), device=device, dtype=W.dtype)
        
        return U, S, Vt, H_sqrt, F_sqrt
    
    def unweight_transform(self,
                           U: torch.Tensor,
                           Vt: torch.Tensor,
                           H_sqrt: torch.Tensor,
                           F_sqrt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        逆变换：将加权 SVD 结果转换回原始空间
        
        W' = F^(-1/2) @ U @ Σ' @ V^T @ H^(-1/2)
        
        Returns:
            U_final, Vt_final: 原始空间的 U 和 V^T
        """
        device = U.device
        
        # F^(-1/2) @ U
        F_inv_sqrt = self.fh_computer.compute_inv_sqrt_matrix(
            F_sqrt @ F_sqrt,  # 从 F^(1/2) 恢复 F
            self.damp
        )
        U_final = F_inv_sqrt @ U
        
        # V^T @ H^(-1/2)
        H_inv_sqrt = self.fh_computer.compute_inv_sqrt_matrix(
            H_sqrt @ H_sqrt,
            self.damp
        )
        Vt_final = Vt @ H_inv_sqrt
        
        return U_final, Vt_final


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


class RecursiveErrorCompensator:
    """
    递归误差吸收器
    
    利用下一层的 Hessian 逆信息，将当前层的截断误差压入下一层权重
    
    理论：
    Δ_l = W_l @ X - W'_l @ X  (当前层误差)
    W_{l+1}' = W_{l+1} + H_{l+1}^(-1) @ Δ_l^T @ ...
    """
    
    def __init__(self, damp: float = 0.01):
        self.damp = damp
        self.fh_computer = FisherHessianComputer(damp)
    
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
        
        # 计算 Hessian 逆
        H_next = H_next.to(device).float()
        H_inv = self.fh_computer.compute_inv_sqrt_matrix(H_next @ H_next)  # 近似 H^(-1)
        
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
    1. 初始化与多维统计收集 (Hessian + Fisher)
    2. 全局边际收益建模 (加权 SVD)
    3. 严格压缩率约束下的秩分配 (贪心选择)
    4. 顺序压缩与跨层误差补偿
    5. 模型重构
    """
    
    def __init__(self,
                 damp: float = 0.01,
                 use_fisher: bool = True,
                 use_compensation: bool = True,
                 compensation_strength: float = 0.1):
        """
        Args:
            damp: 阻尼系数
            use_fisher: 是否使用 Fisher 信息
            use_compensation: 是否启用误差补偿
            compensation_strength: 误差补偿强度
        """
        self.damp = damp
        self.use_fisher = use_fisher
        self.use_compensation = use_compensation
        self.compensation_strength = compensation_strength
        
        self.fh_computer = FisherHessianComputer(damp)
        self.svd_computer = WeightedSVDComputer(damp, use_fisher)
        self.rank_allocator = MarginalUtilityRankAllocator()
        self.error_compensator = RecursiveErrorCompensator(damp)
    
    def compress(self,
                 model,
                 calib_loader,
                 target_ratio: float,
                 device: str,
                 model_name: str) -> nn.Module:
        """
        执行 C-GSVR 压缩
        
        Args:
            model: 待压缩的模型
            calib_loader: 校准数据加载器
            target_ratio: 目标参数保留率
            device: 计算设备
            model_name: 模型名称
            
        Returns:
            compressed_model: 压缩后的模型
        """
        print(f"\n{'='*70}")
        print(f"C-GSVR: Compensated Global Semantic Variable-Rank Compression")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Target ratio: {target_ratio:.2%}")
        print(f"Use Fisher: {self.use_fisher}")
        print(f"Use Compensation: {self.use_compensation}")
        print(f"{'='*70}\n")
        
        # ========== Step 1: 收集统计信息 ==========
        print(f"\n[Step 1/5] Collecting Statistics...")
        h_mat = self.fh_computer.collect_hessian(model, calib_loader, device, model_name)
        
        f_mat = None
        if self.use_fisher:
            f_mat = self.fh_computer.collect_fisher(model, calib_loader, device, model_name)
        
        # ========== Step 2: 加权 SVD ==========
        print(f"\n[Step 2/5] Computing Weighted SVD...")
        if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
            layers = model.model.layers
        elif "opt" in model_name:
            layers = model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        sigma_dict = {}
        shapes_dict = {}
        svd_cache = {}  # {global_name: (U, S, Vt, H_sqrt, F_sqrt)}
        layer_module_map = {}
        total_original_params = 0
        
        for i in tqdm(range(len(layers)), desc="SVD computation"):
            layer = layers[i]
            layer_h = h_mat[i]
            layer_f = f_mat[i] if f_mat else {}
            
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear) and name in layer_h:
                    W = module.weight.data
                    H = layer_h[name]
                    F = layer_f.get(name, None)
                    
                    global_name = f"layer_{i}.{name}"
                    
                    # 执行加权 SVD
                    U, S, Vt, H_sqrt, F_sqrt = self.svd_computer.weighted_svd(
                        W, H, F, device
                    )
                    
                    # 保存结果
                    sigma_dict[global_name] = S.cpu()
                    shapes_dict[global_name] = W.shape
                    svd_cache[global_name] = (U.cpu(), S.cpu(), Vt.cpu(), 
                                              H_sqrt.cpu(), F_sqrt.cpu())
                    layer_module_map[global_name] = (i, name)
                    total_original_params += W.numel()
            
            torch.cuda.empty_cache()
        
        # ========== Step 3: 全局秩分配 ==========
        print(f"\n[Step 3/5] Global Rank Allocation...")
        self.rank_allocator.build_global_utility_table(
            sigma_dict, shapes_dict, layer_module_map
        )
        rank_allocation = self.rank_allocator.allocate_ranks(
            total_original_params, target_ratio
        )
        
        # ========== Step 4: 顺序压缩与跨层误差补偿 ==========
        print(f"\n[Step 4/5] Sequential Compression & Cross-layer Compensation...")
        compressed_modules = {}
        
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
                        U, S, Vt, H_sqrt, F_sqrt = svd_cache[global_name]
                        W_original = module.weight.data.clone()
                        
                        # 截断到分配的秩
                        U_trunc = U[:, :rank]
                        S_trunc = S[:rank]
                        Vt_trunc = Vt[:rank, :]
                        
                        # 逆变换到原始空间
                        U_final, Vt_final = self.svd_computer.unweight_transform(
                            U_trunc.to(device), Vt_trunc.to(device),
                            H_sqrt.to(device), F_sqrt.to(device)
                        )
                        
                        # 计算本层压缩误差（用于传递给下一层）
                        if self.use_compensation and i < len(layers) - 1:
                            W_compressed = (U_final @ torch.diag(S_trunc.to(device))) @ Vt_final
                            delta_W = W_original.to(device).float() - W_compressed
                            layer_error[name] = delta_W.cpu()
                        
                        compressed_modules[global_name] = (
                            U_final.cpu(), S_trunc.cpu(), Vt_final.cpu()
                        )
                    else:
                        compressed_modules[global_name] = None
                    
                    # 释放 SVD 缓存
                    del svd_cache[global_name]
            
            # ===== 跨层误差补偿：将本层误差传递给下一层 =====
            if self.use_compensation and i < len(layers) - 1 and layer_error:
                next_layer = layers[i + 1]
                next_layer_h = h_mat.get(i + 1, {})
                
                # 对下一层的权重进行补偿
                self._apply_error_compensation(
                    next_layer, layer_error, next_layer_h, 
                    layer_inputs.get(i + 1, None), device
                )
            
            torch.cuda.empty_cache()
        
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
