#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/.."

# 运行导出脚本
python convert_onnx/export_onnx.py \
--backbone ResNet50 \
--fc_output_dim 2048 \
--resume_model torchhub \
--simplify \
--verify \
--output_path models/eigenplaces_resnet50_fixedshape_GPU.onnx

# --no_sqrt \
# --dynamic_axes \