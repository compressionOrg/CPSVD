#!/bin/bash

# HS-GSS 快速开始脚本
# 此脚本演示如何使用 HS-GSS 压缩 LLaMA 模型

echo "=================================================="
echo "HS-GSS Quick Start"
echo "=================================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null
then
    echo "Error: Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Step 1: Testing HS-GSS components..."
echo "--------------------------------------------------"
python test_hsgss.py

if [ $? -ne 0 ]; then
    echo "Error: Component tests failed. Please check the output."
    exit 1
fi

echo ""
echo "Step 2: Running a simple compression example..."
echo "--------------------------------------------------"
echo ""
echo "This will compress a model to 50% of its original size using HS-GSS."
echo "Note: This is a dry run. For real compression, uncomment the command below."
echo ""

# 示例命令 (注释掉，避免实际运行)
cat << 'EOF'
# Uncomment and modify the following command to compress your model:

python HSGSS.py \
    --model meta-llama/Llama-2-7b-hf \
    --ratio 0.5 \
    --p 0.667 \
    --dataset wikitext2 \
    --nsamples 128 \
    --cost_aware \
    --eval \
    --eval_datasets wikitext2 ptb c4 \
    --save_model ./compressed_models/llama2_7b_hsgss_50percent \
    --save_hessian ./cache/llama2_7b_hessian.pt

# Key parameters:
# --ratio 0.5     : Retain 50% of parameters (2x compression)
# --p 0.667       : Schatten-2/3 norm (recommended for balance)
# --cost_aware    : Prioritize compressing expensive modules (FFN)
# --eval          : Evaluate perplexity after compression
# --save_hessian  : Save Hessian for reuse with different compression ratios

EOF

echo ""
echo "=================================================="
echo "Quick Start Guide"
echo "=================================================="
echo ""
echo "1. Basic Usage:"
echo "   python HSGSS.py --model <model_name> --ratio <compression_ratio>"
echo ""
echo "2. Recommended Parameters:"
echo "   - For balanced compression: --ratio 0.5 --p 0.667 --cost_aware"
echo "   - For aggressive compression: --ratio 0.3 --p 0.5 --cost_aware"
echo "   - For conservative compression: --ratio 0.7 --p 1.0"
echo ""
echo "3. Two-Stage Workflow (Recommended for experiments):"
echo "   Stage 1: Compute and save Hessian"
echo "     python HSGSS.py --model <model> --ratio 0.5 --save_hessian ./cache/hessian.pt"
echo ""
echo "   Stage 2: Try different compression ratios"
echo "     python HSGSS.py --model <model> --ratio 0.5 --load_hessian ./cache/hessian.pt"
echo "     python HSGSS.py --model <model> --ratio 0.3 --load_hessian ./cache/hessian.pt"
echo ""
echo "4. Supported Models:"
echo "   - LLaMA (meta-llama/Llama-2-7b-hf, etc.)"
echo "   - Mistral (mistralai/Mistral-7B-v0.1, etc.)"
echo "   - Vicuna (lmsys/vicuna-7b-v1.5, etc.)"
echo "   - OPT (facebook/opt-6.7b, etc.)"
echo ""
echo "5. Key Advantages over CPSVD:"
echo "   ✓ Global Pareto-optimal allocation (vs local optimization)"
echo "   ✓ Denoising effect (shrinks large singular values)"
echo "   ✓ Cost-aware compression (prioritizes expensive modules)"
echo "   ✓ Theoretical foundation (Schatten-p regularization)"
echo ""
echo "=================================================="
echo "For more details, see HSGSS_README.md"
echo "=================================================="
echo ""
