# C-GSVR: Compensated Global Semantic Variable-Rank

## 补偿性全局语义变秩法 - LLM 压缩框架

C-GSVR 是一种基于数学理论的大语言模型压缩方法，通过 Fisher-Hessian 联合空间和边际效用等价原理，实现**严格压缩率保证**的同时达到**最优性能**。

---

## 核心理论

### 1. 边际效用等价原理 (Marginal Utility Equivalence)

在约束优化中，当每一层增加一个"秩"所带来的误差下降量（效用）与该秩消耗的参数量（成本）之比相等时，全局重建误差达到最小：

$$
\frac{\partial \mathcal{E}_l}{\partial r_l} \cdot \frac{1}{C_l} = \lambda, \quad \forall l
$$

其中：
- $\mathcal{E}_l$ 是层 $l$ 的重建误差
- $r_l$ 是层 $l$ 的保留秩
- $C_l = \text{out\_dim} + \text{in\_dim}$ 是增加一个秩的参数成本
- $\lambda$ 是全局拉格朗日乘子

### 2. Fisher-Hessian 联合空间

定义重要性得分为经过 Fisher 信息（输出敏感度）和 Hessian（输入能量）双重加权后的奇异值平方：

$$
\tilde{W} = F^{1/2} \cdot W \cdot H^{1/2}
$$

其中：
- $H = \mathbb{E}[X^T X]$ 是输入协方差矩阵 (Hessian)
- $F = \mathbb{E}[(\nabla_Y \mathcal{L})^T (\nabla_Y \mathcal{L})]$ 是 Fisher 信息矩阵

### 3. 递归误差吸收 (Recursive Error Absorption)

通过二阶梯度信息（Hessian 的逆），将当前层的截断误差"压"入下一层的权重空间：

$$
W'_{l+1} = W_{l+1} - \alpha \cdot H_{l+1}^{-1} \cdot \Delta_l^T
$$

---

## 算法流程

### Phase 1: 初始化与多维统计收集

```python
# 1. 收集 Hessian（输入协方差）
H_l = (1/N) * Σ X_l^T @ X_l

# 2. 收集 Fisher（输出敏感度）
F_l = (1/N) * Σ (∂L/∂Y_l)^T @ (∂L/∂Y_l)
```

### Phase 2: 全局边际收益建模

```python
# 1. 加权 SVD
W̃ = F^(1/2) @ W @ H^(1/2)
U, Σ, V^T = SVD(W̃)

# 2. 构建效用表
MU(l, k) = σ_k² / (out_dim + in_dim)

# 3. 全局排序
sort(MU, descending=True)
```

### Phase 3: 严格压缩率约束下的秩分配

```python
# 贪心选择：按边际效用降序选取，直到达到预算
target_budget = total_params * ratio
current_budget = 0

for entry in sorted_utility_pool:
    if current_budget + entry.cost <= target_budget:
        allocate_rank(entry.layer, entry.module)
        current_budget += entry.cost
```

### Phase 4: 顺序压缩与跨层误差补偿

```python
for l in range(n_layers):
    # 1. 执行截断
    W'_l = U[:, :r_l] @ Σ[:r_l] @ V^T[:r_l, :]
    
    # 2. 计算误差
    Δ_l = W_l @ X - W'_l @ X
    
    # 3. 补偿下一层（可选）
    W_{l+1} -= α * H_{l+1}^(-1) @ Δ_l^T
```

### Phase 5: 模型重构

```python
# 逆变换还原
U_final = F^(-1/2) @ U
V^T_final = V^T @ H^(-1/2)

# 替换为低秩结构
nn.Linear -> (v_proj: in_dim → rank, u_proj: rank → out_dim)
```

---

## 使用方法

### 快速开始

```bash
# 使用默认配置（Fisher + 补偿）
python CGSVR.py \
    --model meta-llama/Llama-2-7b-hf \
    --ratio 0.5 \
    --nsamples 128

# 或使用快速启动脚本
MODEL=meta-llama/Llama-2-7b-hf RATIO=0.5 bash quickstart_cgsvr.sh
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | (必需) | HuggingFace 模型名称或路径 |
| `--ratio` | 0.5 | 目标参数保留率 (0.0-1.0) |
| `--use_fisher` | True | 使用 Fisher 信息加权 |
| `--use_compensation` | True | 启用误差补偿 |
| `--damp` | 0.01 | 阻尼系数（数值稳定性） |
| `--nsamples` | 128 | 校准样本数量 |
| `--calib_data` | wikitext2 | 校准数据集 |
| `--seqlen` | 2048 | 序列长度 |

### 高级配置

```bash
# 禁用 Fisher 信息（更快，但可能精度略低）
python CGSVR.py --model MODEL --ratio 0.5 --no_fisher

# 禁用误差补偿（更快，适合浅层模型）
python CGSVR.py --model MODEL --ratio 0.5 --no_compensation

# 调整阻尼系数（解决数值不稳定问题）
python CGSVR.py --model MODEL --ratio 0.5 --damp 0.05

# 保存压缩后的模型
python CGSVR.py --model MODEL --ratio 0.5 --save_model --save_path ./compressed/
```

---

## 测试

```bash
# 运行所有单元测试
python test_cgsvr.py

# 运行特定测试
python test_cgsvr.py --test allocator  # 边际效用分配器
python test_cgsvr.py --test ratio       # 压缩率控制
python test_cgsvr.py --test distribution # 层间分配
```

---

## 与 HS-GSS 的对比

| 特性 | HS-GSS | C-GSVR |
|------|--------|--------|
| 压缩率控制 | 基于 λ 搜索（近似） | 贪心选择（精确） |
| 秩分配策略 | 全局统一 λ | 边际效用等价 |
| 信息利用 | 仅 Hessian | Fisher + Hessian |
| 误差处理 | 无 | 递归补偿 |
| 压缩率误差 | ~1-5% | <1% |
| 理论保证 | Schatten-p 正则化 | 拉格朗日对偶 |

---

## 代码结构

```
CPSVD/
├── CGSVR.py                 # 主入口
├── quickstart_cgsvr.sh      # 快速启动脚本
├── test_cgsvr.py            # 单元测试
├── component/
│   ├── c_gsvr.py            # C-GSVR 核心实现
│   │   ├── FisherHessianComputer    # Fisher/Hessian 收集
│   │   ├── WeightedSVDComputer      # 加权 SVD
│   │   ├── MarginalUtilityRankAllocator  # 秩分配
│   │   ├── RecursiveErrorCompensator    # 误差补偿
│   │   └── CGSVRCompressor          # 主压缩器
│   ├── svd_llama.py         # Llama SVD 模块
│   └── ...
└── utils/
    ├── data_utils.py        # 数据加载
    └── model_utils.py       # 模型加载
```

---

## 实现细节

### 数值稳定性

1. **阻尼系数**：在计算 $H^{-1}$ 和 $F^{-1}$ 时添加岭回归系数
   ```python
   H_damped = H + damp * mean(diag(H)) * I
   ```

2. **异常处理**：SVD 失败时回退到未加权版本

3. **梯度裁剪**：补偿步骤使用小步长避免过度修正

### 内存优化

1. **分批处理**：SVD 计算逐层进行，及时释放中间结果
2. **CPU 缓存**：大型张量保存到 CPU，使用时再移到 GPU
3. **增量更新**：Hessian/Fisher 使用在线累加方式

---

## 常见问题

### Q: 压缩率不准怎么办？
A: C-GSVR 通过贪心选择保证压缩率误差在单个秩的参数量内（通常 <0.1%）。

### Q: 遇到数值不稳定（NaN/Inf）怎么办？
A: 增大 `--damp` 参数（如 0.05 或 0.1）。

### Q: Fisher 信息收集太慢？
A: 使用 `--no_fisher` 禁用，仅使用 Hessian 加权。

### Q: 压缩后 PPL 爆炸？
A: 确保使用 `--use_compensation`，并增加校准样本数。

---

## 引用

如果这个工作对你有帮助，请引用：

```bibtex
@misc{cgsvr2024,
  title={C-GSVR: Compensated Global Semantic Variable-Rank for LLM Compression},
  author={...},
  year={2024}
}
```

---

## License

MIT License
