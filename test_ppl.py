"""
测试 PPL 评估函数
用于调试 evaluate_perplexity 函数的独立测试脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

# 添加当前路径
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from utils.data_utils import get_loaders
from utils.model_utils import get_model_from_huggingface
from evaluater import evaluate_perplexity


def _get_model_device(model):
    """获取模型的设备，兼容不同模型类型"""
    if hasattr(model, 'device'):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_model_forward(model, tokenizer, device):
    """测试模型前向传播"""
    print("\n" + "="*70)
    print("测试模型前向传播")
    print("="*70)
    
    # 创建简单的测试输入
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    print(f"输入文本: {test_text}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input device: {input_ids.device}")
    print(f"Model device: {_get_model_device(model)}")
    
    # 检查模型结构
    print(f"\n模型类型: {type(model).__name__}")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(f"层数: {len(model.model.layers)}")
        first_layer = model.model.layers[0]
        print(f"第一层 self_attn 类型: {type(first_layer.self_attn).__name__}")
        print(f"第一层 mlp 类型: {type(first_layer.mlp).__name__}")
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        print(f"\n✓ 前向传播成功!")
        print(f"输出 logits shape: {outputs[0].shape}")
        return True
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ppl_evaluation(model, tokenizer, dataset_name, seqlen, device):
    """测试 PPL 评估"""
    print("\n" + "="*70)
    print(f"测试 PPL 评估 ({dataset_name})")
    print("="*70)
    
    # 加载测试数据
    print(f"加载测试数据...")
    test_loader, _ = get_loaders(
        dataset_name,
        seed=0,
        seqlen=seqlen,
        tokenizer=tokenizer
    )
    
    # 合并输入
    input_ids = torch.cat([batch[0] for batch in test_loader], 0)
    print(f"测试数据 shape: {input_ids.shape}")
    print(f"样本数: {input_ids.shape[0]}")
    print(f"序列长度: {input_ids.shape[1]}")
    
    # 测试单个样本
    print(f"\n测试单个样本前向传播...")
    try:
        model_device = _get_model_device(model)
        single_input = input_ids[0:1, :-1].to(model_device)
        print(f"单样本输入 shape: {single_input.shape}")
        print(f"单样本输入 device: {single_input.device}")
        
        with torch.no_grad():
            logits = model(input_ids=single_input)[0]
        print(f"✓ 单样本前向传播成功! Logits shape: {logits.shape}")
    except Exception as e:
        print(f"✗ 单样本前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 测试完整 PPL 评估（只用几个样本）
    print(f"\n测试 PPL 评估（前 5 个样本）...")
    try:
        ppl = evaluate_perplexity(model, input_ids, limit=5)
        print(f"✓ PPL 评估成功! PPL = {ppl:.4f}")
        return ppl
    except Exception as e:
        print(f"✗ PPL 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="测试 PPL 评估函数")
    parser.add_argument("--model", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="测试数据集")
    parser.add_argument("--seqlen", type=int, default=2048, help="序列长度")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--full_eval", action="store_true", help="完整评估（所有样本）")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PPL 评估函数测试")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"序列长度: {args.seqlen}")
    print(f"设备: {args.device}")
    
    # 加载模型
    print(f"\n加载模型...")
    model, tokenizer = get_model_from_huggingface(args.model, device_map='auto')
    model.eval()
    print(f"模型加载完成")
    
    # 测试前向传播
    success = test_model_forward(model, tokenizer, args.device)
    if not success:
        print("\n前向传播测试失败，退出")
        return
    
    # 测试 PPL 评估
    ppl = test_ppl_evaluation(model, tokenizer, args.dataset, args.seqlen, args.device)
    
    if ppl is not None and args.full_eval:
        print(f"\n执行完整评估...")
        test_loader, _ = get_loaders(
            args.dataset,
            seed=0,
            seqlen=args.seqlen,
            tokenizer=tokenizer
        )
        input_ids = torch.cat([batch[0] for batch in test_loader], 0)
        full_ppl = evaluate_perplexity(model, input_ids, limit=len(test_loader))
        print(f"完整评估 PPL: {full_ppl:.4f}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == "__main__":
    main()
