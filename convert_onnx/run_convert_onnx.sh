#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/.."

# 运行导出脚本
python convert_onnx/export_onnx.py \
--backbone ResNet50 \
--fc_output_dim 2048 \
--resume_model /home/xihuidong/Documents/workspace/EigenPlaces/models/eigenplaces_resnet50_2048.pth \
--batch_size 2 \
--bhwc_input \
--input_size 480 640 \
--simplify \
--verify \
--output_path models/eigenplaces_resnet50_fixedshape_480_640_GPU.onnx

# --no_sqrt \
# --dynamic_axes \