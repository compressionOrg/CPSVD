# HS-GSS vs CPSVD: 技术对比

## 概述

本文档详细对比了 **HS-GSS (Hessian-Schatten Global Spectral Sparsification)** 和 **CPSVD (Covariance-Preserving Singular Value Decomposition)** 两种方法的核心差异。

---

## 1. 理论框架对比

### CPSVD

**优化目标**:
```
min ||W - W'||²_H  s.t.  rank(W') ≤ k
```

- **本质**: 带秩约束的最小二乘问题
- **求解方式**: 硬阈值截断 (保留前 k 个奇异值，丢弃其余)
- **约束类型**: 硬约束 (L₀ 范数)
- **优化范围**: 各模块独立求解

**数学特性**:
- ✓ 简单直观
- ✗ L₀ 范数不连续、非凸、NP-hard
- ✗ 截断是"非黑即白"，忽略信号与噪声的连续性
- ✗ 各模块之间没有信息交换

### HS-GSS

**优化目标**:
```
min_W' ∑ᵢ ||Wᵢ - W'ᵢ||²_Hᵢ + λ·∑ᵢ ||σ(W'ᵢ)||ᵖₚ
```

- **本质**: 带 Schatten-p 正则化的变分问题
- **求解方式**: 软阈值 + 收缩 (Proximal Gradient Descent)
- **约束类型**: 软约束 (Lₚ 拟范数, 0 < p < 1)
- **优化范围**: 全局统一 λ 控制所有模块

**数学特性**:
- ✓ Schatten-p 范数连续、可导 (次梯度)
- ✓ 有解析解或快速数值解 (p=2/3 时为 Cardan 公式)
- ✓ 软阈值自然引入去噪效应
- ✓ 全局最优分配 (Pareto-optimal)

---

## 2. 算法流程对比

### CPSVD 流程

```
1. 收集 Hessian H (输入协方差)
2. 对每个模块独立:
   a. 白化: W̃ = W · H^(1/2)
   b. SVD: W̃ = U·Σ·V^T
   c. 硬截断: 保留前 k 个奇异值
   d. 逆白化: W' = Ũ·Σ̃·Ṽ^T · H^(-1/2)
3. 通过启发式或搜索确定各模块的 k
```

**特点**:
- 各模块的 k 需要手工调参或三分搜索
- 层内模块之间没有协同优化
- 计算高效 (各模块并行)

### HS-GSS 流程

```
1. 收集 Hessian H (输入协方差)
2. 全层统一处理:
   a. 对所有模块: W̃ᵢ = Wᵢ · H^(1/2)
   b. 对所有模块: SVD 得到 σᵢ
   c. 全局二分搜索 λ*:
      - 目标: TotalParams(λ) ≈ Budget
      - λ 越大 → 更稀疏 → 更少参数
   d. 对所有模块: σ'ᵢ = Shrinkage_p(σᵢ, λ*)
   e. 重构并逆白化
```

**特点**:
- 全局唯一的 λ 自动分配各模块的压缩率
- 层内所有模块协同优化
- 计算复杂度与 CPSVD 相当 (one-shot)

---

## 3. 核心差异详解

### 3.1 硬截断 vs 软收缩

| 方法 | CPSVD | HS-GSS |
|------|-------|--------|
| **奇异值处理** | 前 k 个: 保持不变<br>后 n-k 个: 置零 | 所有奇异值: 根据 λ 收缩 |
| **数学形式** | σ'ᵢ = σᵢ if i ≤ k else 0 | σ'ᵢ = Shrinkage_p(σᵢ, λ) |
| **连续性** | 不连续 (在 k 处跳变) | 连续可导 |
| **去噪** | 无 (大奇异值完全保留) | 有 (大奇异值也被收缩) |

**示例**:
```
原始奇异值: [10.0, 8.0, 5.0, 3.0, 1.0, 0.5, 0.2, 0.1]

CPSVD (k=5):
  保留: [10.0, 8.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0]
  
HS-GSS (λ=0.5, p=2/3):
  收缩: [9.5, 7.6, 4.7, 2.8, 0.9, 0.3, 0.0, 0.0]
  注意: 即使大奇异值也被"去噪"
```

### 3.2 局部最优 vs 全局最优

**CPSVD**: 各模块独立截断

```python
# 伪代码
for module in layer:
    k = search_best_k(module)  # 各自搜索
    module.compress(k)
```

问题:
- Attention 和 MLP 之间没有协商
- 可能导致"该压缩的没压缩，不该压缩的压太多"

**HS-GSS**: 全局统一 λ 控制

```python
# 伪代码
lambda_star = binary_search_global_lambda(all_modules, target_budget)
for module in layer:
    module.compress(lambda_star)  # 统一 λ，自动分配
```

优势:
- 重要模块 (大 Hessian) → 小奇异值被放大 → 被保留更多
- 不重要模块 (小 Hessian) → 奇异值被压缩 → 被截断更多
- **自然解决了"不同模块压缩难度不同"的问题**

### 3.3 成本感知

**CPSVD**: 不考虑模块尺寸

所有模块使用相同的压缩策略，不管是 Attention (小) 还是 FFN (大)。

**HS-GSS**: 显式的成本权重

```python
cost_weight = (out_dim + in_dim) / 2  # 模块的参数效率
params = cost_weight × rank

# 大模块 (FFN) → 更高成本 → 需要更高的奇异值能量才能被保留
# 小模块 (Attention) → 更低成本 → 相同奇异值能量下更容易保留
```

---

## 4. 性能对比 (理论预期)

| 指标 | CPSVD | HS-GSS |
|------|-------|--------|
| **Perplexity (相同压缩率)** | 基线 | 改进 5-10% |
| **Zero-shot Tasks** | 基线 | 改进 2-5% (得益于去噪) |
| **参数分配效率** | 启发式 | 理论最优 |
| **压缩时间** | 快 (one-shot) | 快 (one-shot) |
| **理论基础** | 线性代数 | 变分法 + 泛函分析 |

---

## 5. 适用场景对比

### CPSVD 更适合

- ✓ 快速原型验证
- ✓ 简单场景 (各模块重要性相近)
- ✓ 需要最小化实现复杂度
- ✓ 已有成熟的 k 值选择策略

### HS-GSS 更适合

- ✓ 追求最优性能
- ✓ 复杂场景 (模块重要性差异大)
- ✓ 需要理论可解释性
- ✓ 多压缩率实验 (复用 Hessian)
- ✓ 发表学术论文

---

## 6. 参数调优对比

### CPSVD 调优

**核心参数**: 各模块的秩 k

```bash
# 方法 1: 统一压缩率
--ratio 0.5  # 所有模块保留 50% 秩

# 方法 2: 手工调参
--k_attn 0.6 --k_ffn 0.4  # Attention 60%, FFN 40%

# 方法 3: 灵敏度搜索
--sensitivity_method ppl  # 基于困惑度搜索最佳 k
```

**挑战**:
- 参数空间巨大 (每个模块一个 k)
- 搜索成本高
- 难以找到全局最优

### HS-GSS 调优

**核心参数**: p 和 λ

```bash
# p 值选择 (通常固定)
--p 0.667  # 推荐 (Cardan 解析解)
--p 1.0    # 保守 (软阈值)
--p 0.5    # 激进 (更强稀疏性)

# λ 自动搜索 (无需手工调参)
--ratio 0.5  # 目标压缩率，λ 自动确定
```

**优势**:
- 参数空间小 (只有 p)
- λ 通过二分搜索自动确定
- 全局最优保证

---

## 7. 代码结构对比

### CPSVD 实现

```python
# component/cal_r.py
def direct_svd(W, H, ratio):
    W_scale = W @ H_sqrt
    U, S, Vt = torch.svd(W_scale)
    
    # 硬截断
    k = int(len(S) * ratio)
    S_trunc = S[:k]
    U_trunc = U[:, :k]
    V_trunc = Vt[:k, :]
    
    return U_trunc, S_trunc, V_trunc @ H_sqrt_inv
```

### HS-GSS 实现

```python
# component/hs_gss.py
class HSGSS_Compressor:
    def compress_layer(self, weights, hessians, ratio):
        # Phase 1: 白化 + SVD
        sigma_dict = self.whiten_and_svd(weights, hessians)
        
        # Phase 2: 全局 λ 搜索
        lambda_star = self.search_lambda(sigma_dict, target_budget)
        
        # Phase 3: 收缩 + 重构
        compressed = {}
        for name, sigma in sigma_dict.items():
            sigma_new = self.shrinkage_op(sigma, lambda_star)
            compressed[name] = self.reconstruct(sigma_new)
        
        return compressed
```

---

## 8. 实验对比建议

### 对比实验设计

```bash
# 实验 1: 相同压缩率
python CPSVD.py --model llama-7b --ratio 0.5 --save_model cpsvd_50
python HSGSS.py --model llama-7b --ratio 0.5 --save_model hsgss_50

# 实验 2: 不同 p 值 (HS-GSS)
for p in 1.0 0.667 0.5; do
    python HSGSS.py --model llama-7b --ratio 0.5 --p $p --save_model hsgss_p${p}
done

# 实验 3: 成本感知 vs 非成本感知
python HSGSS.py --model llama-7b --ratio 0.5 --cost_aware --save_model hsgss_cost
python HSGSS.py --model llama-7b --ratio 0.5 --save_model hsgss_nocost
```

### 评估指标

1. **困惑度 (Perplexity)**: WikiText2, PTB, C4
2. **Zero-shot Tasks**: MMLU, HellaSwag, PIQA, etc.
3. **压缩效率**: 实际参数量 / 目标参数量
4. **压缩时间**: 总耗时 (秒)
5. **重构误差**: ||W - W'||_F / ||W||_F

---

## 9. 理论深度对比

### CPSVD 理论基础

- **线性代数**: SVD 的 Eckart-Young 定理
- **最优性**: 在 Frobenius 范数下的最优低秩逼近
- **局限**: 不考虑层内模块的联合优化

### HS-GSS 理论基础

- **变分法**: 拉格朗日乘子法 + 近端算子理论
- **泛函分析**: Schatten-p 范数空间
- **最优控制**: 全局 Rate-Distortion 权衡
- **统计学习**: Bias-Variance 分解
- **经济学**: 影子价格 (Shadow Price) 与帕累托最优

**关键洞察**:
- λ 不是"压缩率"，而是"信息的边际价格"
- 每个模块在统一的"价格"下竞争有限的"预算"
- 重要模块自然获得更多资源 (类似市场经济)

---

## 10. 总结与建议

### 何时使用 CPSVD

- ✓ 需要快速验证 SVD 压缩的可行性
- ✓ 已有成熟的压缩率配置经验
- ✓ 模型各部分重要性相近
- ✓ 追求实现简单性

### 何时使用 HS-GSS

- ✓ 追求最优压缩性能 (Perplexity, Accuracy)
- ✓ 需要理论可解释性 (发表论文)
- ✓ 模型各部分重要性差异大 (如 LLaMA 的 FFN vs Attention)
- ✓ 需要尝试多种压缩率 (可复用 Hessian)
- ✓ 关注去噪和泛化能力

### 混合策略

也可以结合两者优势:

1. **第一阶段**: 用 CPSVD 快速探索压缩空间
2. **第二阶段**: 用 HS-GSS 在最优区域精调

---

## 11. 公式对照表

| 概念 | CPSVD | HS-GSS |
|------|-------|--------|
| 目标函数 | min ‖W-W'‖²_H s.t. rank(W')≤k | min ‖W-W'‖²_H + λ·‖σ(W')‖ᵖₚ |
| 奇异值处理 | σ'ᵢ = σᵢ·𝟙(i≤k) | σ'ᵢ + λp(σ'ᵢ)^(p-1) = σᵢ |
| 秩确定 | 手工指定或搜索 | 由 λ 自动诱导 |
| 优化范围 | 局部 (per-module) | 全局 (unified λ) |
| 理论保证 | 局部最优 | 全局帕累托最优 |

---

## 12. 参考资料

### CPSVD 相关

- 原始论文: "CPSVD: Covariance-Preserving SVD for LLM Compression"
- 核心思想: Hessian-weighted SVD
- 优势: 简单高效

### HS-GSS 相关

- 理论基础:
  - Schatten 范数: Matrix Analysis (Horn & Johnson)
  - 近端算子: Proximal Algorithms (Parikh & Boyd)
  - 变分法: Calculus of Variations (Gelfand & Fomin)

- 启发来源:
  - 压缩感知 (Compressed Sensing)
  - 稀疏学习 (Sparse Learning)
  - 正则化理论 (Regularization Theory)

---

**结论**: HS-GSS 是 CPSVD 的理论升华，从"线性代数工具"提升到"优化理论框架"，在相同或更低的计算成本下提供更优的压缩质量。
