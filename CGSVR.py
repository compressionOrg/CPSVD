"""
C-GSVR: Compensated Global Semantic Variable-Rank

主入口文件: 使用 C-GSVR 方法压缩大语言模型

用法:
    python CGSVR.py --model meta-llama/Llama-2-7b-hf --ratio 0.5

理论优势:
1. 边际效用等价原理 - 全局最优的秩分配
2. Fisher-Hessian 联合空间 - 语义感知的重要性评分
3. 递归误差吸收 - 缓解深层 PPL 爆炸
4. 严格压缩率保证 - 通过贪心选择实现硬约束
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import numpy as np

from utils.data_utils import get_wikitext2, get_ptb, get_c4
from utils.model_utils import get_model_from_huggingface, find_layers
from evaluater import evaluate_perplexity, ppl_eval
from component.c_gsvr import CGSVRCompressor

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
    input_ids = torch.cat([batch[0] for batch in test_loader], 0)
    return evaluate_perplexity(model, input_ids, len(test_loader))


def get_calibration_data(tokenizer, args):
    """获取校准数据"""
    print(f"\nLoading calibration data: {args.calib_data}")
    
    if args.calib_data == "wikitext2":
        calib_loader, _ = get_wikitext2(
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
    elif args.calib_data == "ptb":
        calib_loader, _ = get_ptb(
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
    elif args.calib_data == "c4":
        calib_loader, _ = get_c4(
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
    else:
        raise ValueError(f"Unknown calibration data: {args.calib_data}")
    
    print(f"Loaded {len(calib_loader)} calibration samples")
    return calib_loader


def evaluate_model(model, tokenizer, args):
    """评估压缩后模型的困惑度"""
    print(f"\n{'='*70}")
    print(f"Evaluating Model Perplexity")
    print(f"{'='*70}")
    
    # 使用 ppl_eval 进行评估，保持与 CPSVD.py 一致
    results = ppl_eval(
        model, 
        tokenizer, 
        datasets=args.eval_data, 
        model_seq_len=args.seqlen, 
        batch_size=getattr(args, 'eval_batch_size', 32),
        device=args.device
    )
    
    return results


def save_model(model, tokenizer, args, results):
    """保存压缩后的模型"""
    if args.save_model:
        save_path = args.save_path or f"./compressed_models/{args.model.split('/')[-1]}_cgsvr_{args.ratio}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\nSaving model to {save_path}...")
        
        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'args': vars(args),
            'results': results
        }
        torch.save(save_dict, save_path + ".pt")
        
        # 保存配置
        with open(save_path + "_config.json", 'w') as f:
            json.dump({
                'model': args.model,
                'ratio': args.ratio,
                'use_fisher': args.use_fisher,
                'use_compensation': args.use_compensation,
                'damp': args.damp,
                'results': results
            }, f, indent=2)
        
        print(f"Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='C-GSVR: Compensated Global Semantic Variable-Rank')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for calibration')
    
    # 压缩参数
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Target parameter retention ratio (0.0-1.0)')
    parser.add_argument('--use_fisher', action='store_true', default=True,
                        help='Use Fisher information for weighting')
    parser.add_argument('--no_fisher', action='store_true',
                        help='Disable Fisher information')
    parser.add_argument('--use_compensation', action='store_true', default=False,
                        help='Enable error compensation')
    parser.add_argument('--no_compensation', action='store_true',
                        help='Disable error compensation')
    parser.add_argument('--damp', type=float, default=0.01,
                        help='Damping factor for numerical stability')
    parser.add_argument('--compensation_strength', type=float, default=0.1,
                        help='Error compensation strength')
    
    # 秩分配模式
    parser.add_argument('--fixed_rank', action='store_true', default=False,
                        help='Use fixed rank for all layers (if not set, use variable rank based on marginal utility)')
    parser.add_argument('--fixed_rank_value', type=int, default=None,
                        help='Fixed rank value for all layers (only used when --fixed_rank is set). If not specified, will be computed from ratio.')
    
    # 校准数据参数
    parser.add_argument('--calib_data', type=str, default='wikitext2',
                        choices=['wikitext2', 'ptb', 'c4'],
                        help='Calibration dataset')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # 评估参数
    parser.add_argument('--eval_data', type=str, nargs='+',
                        default=['wikitext2'],
                        help='Evaluation datasets')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip evaluation after compression')
    
    # 保存参数
    parser.add_argument('--save_model', action='store_true',
                        help='Save compressed model')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save compressed model')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to save/load intermediate results')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    
    args = parser.parse_args()
    
    # 处理参数逻辑
    if args.no_fisher:
        args.use_fisher = False
    if args.no_compensation:
        args.use_compensation = False
    
    # 自动设置默认的 checkpoint 路径，以利用缓存
    if args.checkpoint_path is None:
        model_name_safe = args.model.split('/')[-1]
        args.checkpoint_path = f"./checkpoints/{model_name_safe}"
        print(f"Auto-configured checkpoint path: {args.checkpoint_path}")
    
    # 打印配置
    print(f"\n{'='*70}")
    print(f"C-GSVR Configuration")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Target ratio: {args.ratio:.2%}")
    print(f"Use Fisher: {args.use_fisher}")
    print(f"Use Compensation: {args.use_compensation}")
    print(f"Damping: {args.damp}")
    print(f"Rank Mode: {'Fixed' if args.fixed_rank else 'Variable (Marginal Utility)'}")
    if args.fixed_rank and args.fixed_rank_value:
        print(f"Fixed Rank Value: {args.fixed_rank_value}")
    print(f"Calibration data: {args.calib_data} ({args.nsamples} samples)")
    print(f"Sequence length: {args.seqlen}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Device: {args.device}")
    if args.checkpoint_path:
        print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"{'='*70}\n")
    
    # 加载模型
    print("Loading model...")
    model, tokenizer = get_llm(args.model, args.seqlen)
    
    # 获取校准数据
    calib_loader = get_calibration_data(tokenizer, args)
    
    # 创建压缩器
    compressor = CGSVRCompressor(
        damp=args.damp,
        use_fisher=args.use_fisher,
        use_compensation=args.use_compensation,
        compensation_strength=args.compensation_strength,
        fixed_rank=args.fixed_rank,
        fixed_rank_value=args.fixed_rank_value
    )
    
    # 执行压缩
    model = compressor.compress(
        model=model,
        calib_loader=calib_loader,
        target_ratio=args.ratio,
        device=args.device,
        model_name=args.model,
        checkpoint_path=args.checkpoint_path
    )
    # 保存模型
    if args.save_model:
        save_model(model, tokenizer, args, results)
    # 评估
    results = {}
    if not args.skip_eval:
        results = evaluate_model(model, tokenizer, args)
    
    print(f"\n{'='*70}")
    print(f"C-GSVR Compression Complete!")
    print(f"{'='*70}\n")
    
    return model, tokenizer, results


if __name__ == "__main__":
    main()
