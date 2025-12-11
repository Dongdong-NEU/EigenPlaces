#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/.."

# 运行导出脚本
python convert_onnx/export_onnx.py \
--backbone ResNet50 \
--fc_output_dim 2048 \
--resume_model /home/xihuidong/Documents/workspace/EigenPlaces/models/eigenplaces_resnet50_2048.pth \
--bhwc_input \
--simplify \
--verify \
--batch_size 2 \
--input_size 180 480 \
--output_path models/eigenplaces_resnet50_fixedshape_180_480.onnx

# --no_sqrt \
# --dynamic_axes \
# --resume_model  /home/xihuidong/.cache/torch/hub/checkpoints/eigenplaces_ResNet50_2048_GB1_BAI_5_10.pth \
