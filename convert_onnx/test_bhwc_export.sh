#!/bin/bash

# 测试BHWC输入格式的ONNX导出

echo "=========================================="
echo "测试 BHWC 输入格式的 ONNX 导出"
echo "=========================================="

# 导出BHWC格式的ONNX模型
python export_onnx.py \
    --backbone ResNet50 \
    --fc_output_dim 2048 \
    --resume_model torchhub \
    --output_path ../models/eigenplaces_bhwc_input.onnx \
    --input_size 512 512 \
    --batch_size 1 \
    --bhwc_input \
    --verify

echo ""
echo "=========================================="
echo "导出完成！"
echo ""
echo "导出的模型:"
echo "  - 输入格式: BHWC (batch, height, width, channels)"
echo "  - 输入形状: [1, 512, 512, 3]"
echo "  - 输出形状: [1, 2048]"
echo ""
echo "模型文件: ../models/eigenplaces_bhwc_input.onnx"
echo "=========================================="

