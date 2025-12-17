#!/bin/bash

# 加载 CUDA 12.4 模块
module load cuda/12.4

# 激活 conda 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate mlc-llm-env

# 检查 nvcc 是否可用
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc 已加载: $(nvcc --version | head -n 1)"
else
    echo "✗ 警告: nvcc 未找到"
fi

# 运行 MLC LLM chat
echo "正在启动 MLC LLM chat..."
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

