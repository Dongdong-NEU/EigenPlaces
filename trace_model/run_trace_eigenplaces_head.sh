#!/bin/bash

# EigenPlaces Head (aggregation) 模型trace运行脚本
# 使用方法: ./run_trace_eigenplaces_head.sh [模型路径]

# 设置默认参数
BACKBONE="ResNet50"
FC_OUTPUT_DIM=2048
FEATURE_SIZE="16 16"  # backbone输出的特征图大小
BATCH_SIZE=1
DEVICE="cpu"
TRACE_DIR="./results"
MODEL_NAME="eigenplaces_head_traced"

# 检查是否提供了模型路径参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供模型路径参数"
    echo "使用方法: $0 <模型路径> [选项]"
    echo ""
    echo "示例:"
    echo "  $0 ../models/eigenplaces_model.pth"
    echo "  $0 torchhub  # 使用PyTorch Hub预训练模型"
    echo "  $0 ../models/eigenplaces_model.pth --no_sqrt  # 使用DLA兼容版本"
    echo ""
    echo "可选参数:"
    echo "  --backbone BACKBONE        骨架网络 (默认: ResNet50)"
    echo "  --fc_output_dim DIM        输出维度 (默认: 2048)"
    echo "  --feature_size H W         特征图尺寸 (默认: 16 16)"
    echo "  --batch_size SIZE          批大小 (默认: 1)"
    echo "  --device DEVICE            设备 (默认: cpu)"
    echo "  --trace_dir DIR            输出目录 (默认: ./results)"
    echo "  --model_name NAME          模型名称 (默认: eigenplaces_head_traced)"
    echo "  --no_sqrt                  使用DLA兼容版本(无Sqrt算子)"
    echo ""
    echo "说明:"
    echo "  此脚本只trace EigenPlaces的head部分 (aggregation层)"
    echo "  Head包含: L2Norm -> GeM -> Flatten -> Linear -> L2Norm"
    echo "  输入: backbone输出的特征图 (B, C, H, W)"
    echo "  输出: 描述符向量 (B, fc_output_dim)"
    exit 1
fi

# 获取模型路径
RESUME_MODEL=$1
shift

echo "=========================================="
echo "EigenPlaces Head Trace 工具"
echo "=========================================="
echo "模型路径: $RESUME_MODEL"
echo "骨架网络: $BACKBONE"
echo "输出维度: $FC_OUTPUT_DIM"
echo "特征图尺寸: $FEATURE_SIZE"
echo "批大小: $BATCH_SIZE"
echo "设备: $DEVICE"
echo "输出目录: $TRACE_DIR"
echo "模型名称: $MODEL_NAME"
echo "=========================================="

echo "开始trace Head模块..."
python trace_eigenplaces_head.py \
    --resume_model "$RESUME_MODEL" \
    --backbone "$BACKBONE" \
    --fc_output_dim "$FC_OUTPUT_DIM" \
    --feature_size $FEATURE_SIZE \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --trace_dir "$TRACE_DIR" \
    --model_name "$MODEL_NAME" \
    "$@"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Trace成功完成!"
    echo "=========================================="
    echo "输出目录: $TRACE_DIR"
    echo "模型名称: $MODEL_NAME"
else
    echo ""
    echo "=========================================="
    echo "❌ Trace失败，请查看日志"
    echo "=========================================="
    exit 1
fi

