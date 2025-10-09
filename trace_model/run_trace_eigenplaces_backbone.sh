#!/bin/bash

# EigenPlaces模型trace运行脚本
# 使用方法: ./run_trace_eigenplaces.sh [模型路径]

# 设置默认参数
BACKBONE="ResNet50"
FC_OUTPUT_DIM=2048
INPUT_SIZE="512 512"
BATCH_SIZE=1
DEVICE="cuda"
TRACE_DIR="./results"
MODEL_NAME="eigenplaces_backbone_traced"

# 检查是否提供了模型路径参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供模型路径参数"
    echo "使用方法: $0 <模型路径> [选项]"
    echo ""
    echo "示例:"
    echo "  $0 ../models/eigenplaces_model.pth"
    echo "  $0 torchhub  # 使用PyTorch Hub预训练模型"
    echo ""
    echo "可选参数:"
    echo "  --backbone BACKBONE        骨架网络 (默认: ResNet50)"
    echo "  --fc_output_dim DIM        输出维度 (默认: 2048)"
    echo "  --input_size H W           输入尺寸 (默认: 512 512)"
    echo "  --batch_size SIZE          批大小 (默认: 1)"
    echo "  --device DEVICE            设备 (默认: cpu)"
    echo "  --trace_dir DIR            输出目录 (默认: ./results)"
    exit 1
fi

# 获取模型路径
RESUME_MODEL=$1
shift

echo "=========================================="
echo "EigenPlaces 模型 Trace 工具"
echo "=========================================="
echo "模型路径: $RESUME_MODEL"
echo "骨架网络: $BACKBONE"
echo "输出维度: $FC_OUTPUT_DIM"
echo "输入尺寸: $INPUT_SIZE"
echo "批大小: $BATCH_SIZE"
echo "设备: $DEVICE"
echo "输出目录: $TRACE_DIR"
echo "模型名称: $MODEL_NAME"
echo "=========================================="

echo "开始trace模型..."
python trace_eigenplaces_backbone.py \
    --resume_model "$RESUME_MODEL" \
    --backbone "$BACKBONE" \
    --fc_output_dim "$FC_OUTPUT_DIM" \
    --input_size $INPUT_SIZE \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --trace_dir "$TRACE_DIR" \
    --model_name "$MODEL_NAME"
