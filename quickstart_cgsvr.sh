#!/bin/bash
# C-GSVR 快速启动脚本
# Compensated Global Semantic Variable-Rank Compression

set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 默认参数
MODEL=${MODEL:-"meta-llama/Llama-2-7b-hf"}
RATIO=${RATIO:-0.8}
NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-2048}
CALIB_DATA=${CALIB_DATA:-"wikitext2"}
# 设置 USE_FISHER=0 可禁用 Fisher 信息收集（节省 GPU 内存）
USE_FISHER=${USE_FISHER:-1}

echo "=========================================="
echo "C-GSVR: Compensated Global Semantic"
echo "        Variable-Rank Compression"
echo "=========================================="
echo "Model: $MODEL"
echo "Target ratio: $RATIO"
echo "Calibration: $CALIB_DATA ($NSAMPLES samples)"
echo "Sequence length: $SEQLEN"
echo "Use Fisher: $USE_FISHER"
echo "=========================================="

# 构建 Fisher 参数
if [ "$USE_FISHER" = "1" ]; then
    FISHER_ARG="--use_fisher"
else
    FISHER_ARG="--no_fisher"
fi

# 基础运行
python CGSVR.py \
    --model $MODEL \
    --ratio $RATIO \
    --nsamples $NSAMPLES \
    --seqlen $SEQLEN \
    --calib_data $CALIB_DATA \
    $FISHER_ARG \
    --use_compensation \
    --damp 0.01 \
    --eval_data wikitext2 \
    "$@"

echo ""
echo "=========================================="
echo "C-GSVR Compression Complete!"
echo "=========================================="

set +x 