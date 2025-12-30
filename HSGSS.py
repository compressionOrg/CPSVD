"""
HS-GSS: Hessian-Schatten Global Spectral Sparsification for LLM Compression

主入口文件: 使用 HS-GSS 方法压缩大语言模型

用法:
    python HSGSS.py --model meta-llama/Llama-2-7b-hf --ratio 0.5 --p 0.667
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np

from utils.data_utils import *
from utils.model_utils import get_model_from_huggingface, find_layers
from evaluater import evaluate_perplexity
from component.hs_gss import (
    HSGSS_Compressor,
    apply_hsgss_to_llama_layer,
    analyze_compression_distribution
)

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)


def get_llm(model_name, seqlen):
    """加载大语言模型"""
    model, tokenizer = get_model_from_huggingface(model_name, device_map='auto')
    model.seqlen = seqlen
    model.eval()
    return model, tokenizer


def eval_ppl(model, test_loader, device):
    """评估困惑度"""
    # test_loader 是 (input, target) 元组的列表
    input_ids = torch.cat([batch[0] for batch in test_loader], 0)
    return evaluate_perplexity(model, input_ids, len(test_loader))


@torch.no_grad()
def collect_hessian_matrices(model, calib_loader, dev, model_name):
    """
    收集每一层每个模块的 Hessian (输入协方差矩阵)
    
    Returns:
        h_mat: {layer_idx: {module_name: H_matrix}}
    """
    if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
        layers = model.model.layers
    elif "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(dev)
    print(f"\n{'='*70}")
    print(f"Collecting Hessian (Input Covariance) Matrices...")
    print(f"{'='*70}")
    
    # 注册 hook 来收集输入统计
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:  # for OPT
            inp = inp.unsqueeze(0)
        
        # 计算 X^T @ X (协方差矩阵)
        adds = torch.matmul(inp.transpose(1, 2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_hessian += adds_sum
        
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    
    # 为所有线性层注册 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_hessian = 0
            module.register_forward_hook(hook)
    
    # 前向传播收集统计
    for batch in tqdm(calib_loader, desc="Forward pass"):
        # batch 是 (input_ids, target) 元组
        if isinstance(batch, (tuple, list)):
            input_ids = batch[0].to(dev)
            attention_mask = torch.ones_like(input_ids)
            model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # 如果是字典格式
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
    
    # 清理 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    
    torch.cuda.empty_cache()
    
    # 收集 Hessian 矩阵
    h_mat = {}
    print("\nCollecting Hessian matrices from layers...")
    
    for i in tqdm(range(len(layers)), desc="Extracting"):
        layer_h = {}
        
        # 创建名称映射：从模块对象到规范名称
        for full_name, module in layers[i].named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'raw_hessian'):
                # 规范化名称：self_attn.q_proj, mlp.gate_proj 等
                layer_h[full_name] = module.raw_hessian.cpu()
                del module.raw_hessian
        
        h_mat[i] = layer_h
        torch.cuda.empty_cache()
    
    print(f"Hessian collection completed for {len(h_mat)} layers.\n")
    return h_mat


@torch.no_grad()
def compress_model_with_hsgss(model, h_mat, args):
    """
    使用 HS-GSS 方法压缩整个模型（真正的全局优化）
    
    Args:
        model: 待压缩的模型
        h_mat: Hessian 矩阵字典 {layer_idx: {module_name: H}}
        args: 命令行参数
    """
    if "llama" in args.model or "mistral" in args.model or "vicuna" in args.model:
        layers = model.model.layers
    elif "opt" in args.model:
        layers = model.model.decoder.layers
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    print(f"\n{'='*70}")
    print(f"HS-GSS Compression Pipeline (Global Optimization)")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Target compression ratio: {args.ratio:.2%}")
    print(f"Schatten-p parameter: {args.p}")
    print(f"Number of layers: {len(layers)}")
    print(f"{'='*70}\n")
    
    # 创建全局压缩器
    compressor = HSGSS_Compressor(
        p=args.p,
        damp=args.damp,
        cost_aware=args.cost_aware
    )
    
    # ========== Phase 1: 全局数据收集 ==========
    print(f"{'='*70}")
    print(f"Phase 1: Global Data Collection")
    print(f"{'='*70}\n")
    
    all_weights = {}
    all_hessians = {}
    all_shapes = {}
    total_original_params = 0
    
    for i in tqdm(range(len(layers)), desc="Collecting weights"):
        layer = layers[i]
        layer_hessians = h_mat[i]
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and name in layer_hessians:
                W = module.weight.data
                # 使用层索引+模块名作为全局唯一标识
                global_name = f"layer_{i}.{name}"
                # 将权重克隆到CPU，减少GPU内存占用
                all_weights[global_name] = W.cpu().clone()
                # Hessian已经在CPU上
                all_hessians[global_name] = layer_hessians[name]
                all_shapes[global_name] = W.shape
                total_original_params += W.numel()
    
    print(f"\nTotal modules: {len(all_weights)}")
    print(f"Total original parameters: {total_original_params:,}\n")
    
    # ========== Phase 2: 全局白化与SVD（分批处理，节省GPU内存）==========
    print(f"{'='*70}")
    print(f"Phase 2: Global Whitening & SVD (Memory Efficient)")
    print(f"{'='*70}\n")
    
    global_sigma_dict = {}
    global_whitened_svd = {}
    global_h_sqrt_dict = {}
    
    # 获取计算设备
    compute_device = args.device if hasattr(args, 'device') else 'cuda'
    
    for global_name, W in tqdm(all_weights.items(), desc="Whitening & SVD"):
        H = all_hessians[global_name]
        
        # 将数据移动到GPU进行计算
        W_gpu = W.float().to(compute_device)
        H_gpu = H.float().to(compute_device)
        
        # 白化（在GPU上）
        W_whitened, H_sqrt = compressor.whitening.whiten(W_gpu, H_gpu, compressor.damp)
        
        # SVD（在GPU上）
        U, S, Vt = torch.linalg.svd(W_whitened, full_matrices=False)
        
        # 立即将结果移回CPU，释放GPU内存
        global_sigma_dict[global_name] = S.cpu()
        global_whitened_svd[global_name] = (U.cpu(), S.cpu(), Vt.cpu())
        global_h_sqrt_dict[global_name] = H_sqrt.cpu()
        
        # 清理GPU内存
        del W_gpu, H_gpu, W_whitened, H_sqrt, U, S, Vt
        torch.cuda.empty_cache()
    
    # ========== Phase 3: 全局Lambda搜索 ==========
    print(f"\n{'='*70}")
    print(f"Phase 3: Global Lambda Search")
    print(f"{'='*70}\n")
    
    target_budget = int(total_original_params * args.ratio)
    print(f"Target parameter budget: {target_budget:,} ({args.ratio:.2%})\n")
    
    # 成本权重
    cost_weights = {}
    if compressor.cost_aware:
        for name, shape in all_shapes.items():
            cost = (shape[0] + shape[1]) / 2.0
            cost_weights[name] = cost / 1000.0
    
    # 全局二分搜索
    from component.hs_gss import GlobalLambdaSearcher
    searcher = GlobalLambdaSearcher(compressor.p, cost_weights)
    optimal_lambda = searcher.search_lambda(
        global_sigma_dict, all_shapes, target_budget
    )
    
    print(f"\nOptimal global λ: {optimal_lambda:.6e}\n")
    
    # ========== Phase 4: 全局压缩与重构 ==========
    print(f"{'='*70}")
    print(f"Phase 4: Global Compression & Reconstruction")
    print(f"{'='*70}\n")
    
    global_compressed = {}
    total_compressed_params = 0
    
    for global_name in tqdm(all_weights.keys(), desc="Compressing modules"):
        U, S_old, Vt = global_whitened_svd[global_name]
        H_sqrt = global_h_sqrt_dict[global_name]
        
        # cost_aware: 对不同模块应用不同的等效lambda
        # （与lambda搜索时保持一致！）
        cost_weight = cost_weights.get(global_name, 1.0)
        effective_lambda = optimal_lambda * cost_weight
        
        # 应用收缩（使用等效lambda）
        S_new = compressor.shrinkage_op.shrink(S_old, effective_lambda)
        rank = torch.sum(S_new > 1e-8).item()
        
        if rank > 0:
            U_trunc = U[:, :rank]
            S_trunc = S_new[:rank]
            Vt_trunc = Vt[:rank, :]
            
            # 逆白化（在GPU上完成，然后移回CPU）
            Vt_trunc_gpu = Vt_trunc.to(compute_device)
            H_sqrt_gpu = H_sqrt.to(compute_device)
            Vt_final_gpu = compressor.whitening.unwhiten(Vt_trunc_gpu, H_sqrt_gpu)
            Vt_final = Vt_final_gpu.cpu()
            
            # 清理
            del Vt_trunc_gpu, H_sqrt_gpu, Vt_final_gpu
            torch.cuda.empty_cache()
            
            global_compressed[global_name] = (U_trunc, S_trunc, Vt_final)
            
            out_dim, in_dim = all_shapes[global_name]
            total_compressed_params += rank * (out_dim + in_dim)
        else:
            global_compressed[global_name] = None
        
        # 释放已处理的SVD结果，节省内存
        del global_whitened_svd[global_name]
        del global_h_sqrt_dict[global_name]
    
    # 清理
    global_whitened_svd.clear()
    global_h_sqrt_dict.clear()
    torch.cuda.empty_cache()
    
    # ========== Phase 5: 替换模型层 ==========
    print(f"\n{'='*70}")
    print(f"Phase 5: Replacing Model Layers")
    print(f"{'='*70}\n")
    
    for i in tqdm(range(len(layers)), desc="Replacing layers"):
        layer = layers[i]
        
        # 收集该层的压缩结果
        layer_compressed = {}
        layer_shapes = {}
        
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                global_name = f"layer_{i}.{name}"
                if global_name in global_compressed:
                    layer_compressed[name] = global_compressed[global_name]
                    layer_shapes[name] = all_shapes[global_name]
        
        # 替换层（传递 layer_idx）
        if layer_compressed:
            replace_modules_in_layer(layer, layer_compressed, args, layer_idx=i)
    
    # ========== 最终统计 ==========
    print(f"\n{'='*70}")
    print(f"Global Compression Summary")
    print(f"{'='*70}")
    print(f"Original parameters: {total_original_params:,}")
    print(f"Compressed parameters: {total_compressed_params:,}")
    print(f"Actual compression ratio: {total_compressed_params/total_original_params:.2%}")
    print(f"Parameter reduction: {(1 - total_compressed_params/total_original_params):.2%}")
    print(f"Target ratio: {args.ratio:.2%}")
    print(f"Ratio error: {abs(total_compressed_params/total_original_params - args.ratio)/args.ratio*100:.2f}%")
    print(f"Global λ: {optimal_lambda:.6e}")
    print(f"{'='*70}\n")
    
    return model
    print(f"Parameter reduction: {(1 - total_compressed_params/total_original_params):.2%}")
    print(f"{'='*70}\n")
    
    return model


def replace_modules_in_layer(layer, compressed_modules, args, layer_idx=0):
    """
    将压缩后的 SVD 分解替换到层中
    
    Args:
        layer: 要替换的层
        compressed_modules: 压缩后的模块字典 {name: (U, S, Vt)}
        args: 命令行参数
        layer_idx: 层索引（用于 DynamicCache 兼容）
    """
    from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
    from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
    from component.svd_opt import SVDOPTDecoderLayer
    
    # 根据模型类型选择对应的 SVD 模块
    if "llama" in args.model or "vicuna" in args.model:
        # 检查是否需要替换 Attention
        attn_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
        if all(name in compressed_modules for name in attn_modules):
            # 检查是否所有模块都被压缩（不为 None）
            if all(compressed_modules[name] is not None for name in attn_modules):
                config = layer.self_attn.config
                # 获取原始层的dtype和device
                original_dtype = layer.self_attn.q_proj.weight.dtype
                original_device = layer.self_attn.q_proj.weight.device
                
                # 创建 SVD_LlamaAttention 并传递 layer_idx
                svd_attn = SVD_LlamaAttention(config, ratio=1.0, layer_idx=layer_idx)
                
                for proj in ['q', 'k', 'v', 'o']:
                    module_name = f'self_attn.{proj}_proj'
                    U, S, Vt = compressed_modules[module_name]  # Vt is [rank, in_dim]
                    rank = len(S)
                    
                    # 获取维度：U [out_dim, rank], Vt [rank, in_dim]
                    out_dim = U.shape[0]
                    in_dim = Vt.shape[1]
                    
                    # v_proj: input [batch, seq, in_dim] -> output [batch, seq, rank]
                    # u_proj: input [batch, seq, rank] -> output [batch, seq, out_dim]
                    v_proj = nn.Linear(in_dim, rank, bias=False)
                    u_proj = nn.Linear(rank, out_dim, bias=False)
                    
                    # nn.Linear weight shape: [out_features, in_features]
                    sqrt_S = torch.sqrt(torch.diag(S))
                    # v_proj.weight: [rank, in_dim] = sqrt(S) @ Vt
                    v_proj.weight.data = (sqrt_S @ Vt).to(dtype=original_dtype, device=original_device)
                    # u_proj.weight: [out_dim, rank] = U @ sqrt(S)
                    u_proj.weight.data = (U @ sqrt_S).to(dtype=original_dtype, device=original_device)
                    
                    # 移动到正确设备和dtype
                    v_proj = v_proj.to(dtype=original_dtype, device=original_device)
                    u_proj = u_proj.to(dtype=original_dtype, device=original_device)
                    
                    setattr(svd_attn, f'{proj}_v_proj', v_proj)
                    setattr(svd_attn, f'{proj}_u_proj', u_proj)
                
                # 将整个 svd_attn 移动到正确设备（包括 rotary_emb 的缓存）
                svd_attn = svd_attn.to(dtype=original_dtype, device=original_device)
                layer.self_attn = svd_attn
        
        # 检查是否需要替换 MLP
        mlp_modules = ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
        if all(name in compressed_modules for name in mlp_modules):
            if all(compressed_modules[name] is not None for name in mlp_modules):
                hidden_size = layer.mlp.gate_proj.weight.shape[1]
                intermediate_size = layer.mlp.gate_proj.weight.shape[0]
                hidden_act = getattr(layer.mlp.config, 'hidden_act', 'silu')
                
                # 获取原始层的dtype和device
                original_dtype = layer.mlp.gate_proj.weight.dtype
                original_device = layer.mlp.gate_proj.weight.device
                
                svd_mlp = SVD_LlamaMLP(
                    hidden_size, intermediate_size, hidden_act, ratio=1.0
                )
                
                for proj in ['gate', 'up', 'down']:
                    module_name = f'mlp.{proj}_proj'
                    U, S, Vt = compressed_modules[module_name]  # Vt is [rank, in_dim]
                    rank = len(S)
                    
                    out_dim = U.shape[0]
                    in_dim = Vt.shape[1]
                    
                    # v_proj: in_dim -> rank, u_proj: rank -> out_dim
                    v_proj = nn.Linear(in_dim, rank, bias=False)
                    u_proj = nn.Linear(rank, out_dim, bias=False)
                    
                    sqrt_S = torch.sqrt(torch.diag(S))
                    # v_proj.weight: [rank, in_dim] = sqrt(S) @ Vt
                    v_proj.weight.data = (sqrt_S @ Vt).to(dtype=original_dtype, device=original_device)
                    # u_proj.weight: [out_dim, rank] = U @ sqrt(S) (无转置!)
                    u_proj.weight.data = (U @ sqrt_S).to(dtype=original_dtype, device=original_device)
                    
                    # 移动到正确设备和dtype
                    v_proj = v_proj.to(dtype=original_dtype, device=original_device)
                    u_proj = u_proj.to(dtype=original_dtype, device=original_device)
                    
                    setattr(svd_mlp, f'{proj}_v_proj', v_proj)
                    setattr(svd_mlp, f'{proj}_u_proj', u_proj)
                
                # 将整个 svd_mlp 移动到正确设备
                svd_mlp = svd_mlp.to(dtype=original_dtype, device=original_device)
                layer.mlp = svd_mlp
    
    elif "mistral" in args.model:
        # Mistral 模型的处理 (类似 Llama)
        if all(f'self_attn.{proj}_proj' in compressed_modules 
               for proj in ['q', 'k', 'v', 'o']):
            config = layer.self_attn.config
            svd_attn = SVD_MistralAttention(config, ratio=args.ratio)
            # ... (类似的权重设置逻辑)
            layer.self_attn = svd_attn
        
        if all(f'mlp.{proj}_proj' in compressed_modules 
               for proj in ['gate', 'up', 'down']):
            # ... (类似的 MLP 处理)
            pass
    
    elif "opt" in args.model:
        # OPT 模型使用不同的结构
        # 需要根据 SVDOPTDecoderLayer 的具体实现来设置
        pass


def main():
    parser = argparse.ArgumentParser(description="HS-GSS: Hessian-Schatten Global Spectral Sparsification")
    
    # 模型相关参数
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    
    # 压缩参数
    parser.add_argument("--ratio", type=float, default=0.5,
                       help="Target parameter retention ratio (default: 0.5 = 50%%)")
    parser.add_argument("--p", type=float, default=2/3,
                       help="Schatten-p norm parameter (default: 2/3, recommended: [0.5, 1.0])")
    parser.add_argument("--damp", type=float, default=0.01,
                       help="Hessian damping coefficient for numerical stability (default: 0.01)")
    parser.add_argument("--cost_aware", action="store_true",
                       help="Enable cost-aware compression (prioritize expensive modules)")
    
    # 校准数据参数
    parser.add_argument("--dataset", type=str, default="wikitext2",
                       choices=["wikitext2", "c4", "ptb"],
                       help="Calibration dataset")
    parser.add_argument("--nsamples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--seqlen", type=int, default=2048,
                       help="Sequence length")
    
    # 评估参数
    parser.add_argument("--eval", action="store_true",
                       help="Evaluate perplexity after compression")
    parser.add_argument("--eval_datasets", type=str, nargs="+",
                       default=["wikitext2", "ptb", "c4"],
                       help="Datasets for evaluation")
    
    # 保存/加载参数
    parser.add_argument("--save_model", type=str, default=None,
                       help="Path to save compressed model")
    parser.add_argument("--load_hessian", type=str, default=None,
                       help="Path to load pre-computed Hessian matrices")
    parser.add_argument("--save_hessian", type=str, default=None,
                       help="Path to save Hessian matrices")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建 cache 目录
    os.makedirs("cache", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"HS-GSS: Hessian-Schatten Global Spectral Sparsification")
    print(f"{'='*70}")
    print(f"Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"{'='*70}\n")
    
    # 检查是否已有压缩模型
    if args.save_model and os.path.exists(args.save_model):
        print(f"Found existing compressed model at {args.save_model}")
        print("Loading compressed model...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        compressed_model = AutoModelForCausalLM.from_pretrained(
            args.save_model,
            device_map='auto',
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.save_model)
        compressed_model.eval()
        print(f"Compressed model loaded from disk.\n")
    else:
        # 加载模型
        print("Loading model...")
        model, tokenizer = get_llm(args.model, args.seqlen)
        model.eval()
        print(f"Model loaded: {model.config._name_or_path}")
        print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters\n")
        
        # 如果save_hessian存在但load_hessian未指定，自动使用save_hessian
        if not args.load_hessian and args.save_hessian and os.path.exists(args.save_hessian):
            print(f"Found existing Hessian at {args.save_hessian}")
            args.load_hessian = args.save_hessian
        
        # 加载校准数据
        print(f"Loading calibration data ({args.dataset}, {args.nsamples} samples)...")
        dataloader, _ = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
        print(f"Calibration data loaded.\n")
        
        # 收集或加载 Hessian 矩阵
        if args.load_hessian and os.path.exists(args.load_hessian):
            print(f"Loading pre-computed Hessian from {args.load_hessian}...")
            h_mat = torch.load(args.load_hessian, map_location="cpu")
            print("Hessian loaded.\n")
        else:
            h_mat = collect_hessian_matrices(
                model, dataloader, args.device, args.model
            )
            
            if args.save_hessian:
                print(f"Saving Hessian to {args.save_hessian}...")
                os.makedirs(os.path.dirname(args.save_hessian), exist_ok=True)
                torch.save(h_mat, args.save_hessian)
                print("Hessian saved.\n")
        
        # 执行 HS-GSS 压缩
        compressed_model = compress_model_with_hsgss(model, h_mat, args)
    
    # 评估
    if args.eval:
        print(f"\n{'='*70}")
        print(f"Evaluation")
        print(f"{'='*70}\n")
        
        for dataset in args.eval_datasets:
            print(f"Evaluating on {dataset}...")
            test_loader, test_dataset = get_loaders(
                dataset,
                seed=args.seed,
                seqlen=args.seqlen,
                tokenizer=tokenizer
            )
            
            ppl = eval_ppl(compressed_model, test_loader, args.device)
            print(f"{dataset} perplexity: {ppl:.4f}\n")
    
    # 保存模型
    if args.save_model and not os.path.exists(args.save_model):
        print(f"Saving compressed model to {args.save_model}...")
        os.makedirs(args.save_model, exist_ok=True)
        compressed_model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model saved.\n")
    elif args.save_model:
        print(f"Compressed model already exists at {args.save_model}, skipping save.\n")
    
    print(f"\n{'='*70}")
    print(f"HS-GSS Compression Completed Successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
