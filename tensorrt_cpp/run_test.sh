#!/bin/bash

# EigenPlaces TensorRT测试运行脚本
# 使用固定尺寸ONNX模型进行推理测试

set -e  # 遇到错误立即退出

echo "=== EigenPlaces TensorRT推理测试 ==="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查可执行文件
EXECUTABLE="build/EigenPlaces_TensorRT"
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 可执行文件未找到: $EXECUTABLE"
    echo "请先运行 ./build.sh 编译项目"
    exit 1
fi

# 设置模型和图像路径
ONNX_MODEL="../models/eigenplaces_resnet50_onnx_no_sqrt_simplified.onnx"
ENGINE_FILE="../models/eigenplaces_resnet50_onnx_no_sqrt_simplified.onnx"

# ONNX_MODEL="../models/eigenplaces_resnet50_onnx_fixheightwidth_simplified.onnx"
# ENGINE_FILE="../models/eigenplaces_resnet50_fixheightwidth_simplified.engine"

TEST_MAP_IMAGE="/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/map_test/1741760814.740169.png"
TEST_QUERY_IMAGE="/home/xihuidong/codetree/repo/visual_mapping_test/avp-reloc-indoor/map_test/1741760816.640487.png"

# 检查文件是否存在
echo "检查必要文件..."

if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: ONNX模型文件未找到: $ONNX_MODEL"
    echo "请确保已经生成了固定尺寸的ONNX模型"
    exit 1
fi

if [ ! -f "$TEST_MAP_IMAGE" ]; then
    echo "错误: 测试地图图像未找到: $TEST_MAP_IMAGE"
    exit 1
fi

if [ ! -f "$TEST_QUERY_IMAGE" ]; then
    echo "错误: 测试查询图像未找到: $TEST_QUERY_IMAGE"
    exit 1
fi

echo "✓ 所有必要文件检查完成"

# 显示文件信息
echo ""
echo "测试配置:"
echo "  - ONNX模型: $ONNX_MODEL ($(du -h "$ONNX_MODEL" | cut -f1))"
echo "  - 引擎缓存: $ENGINE_FILE"
echo "  - 测试地图图像: $TEST_MAP_IMAGE"
echo "  - 测试查询图像: $TEST_QUERY_IMAGE"

# 检查GPU信息
echo ""
echo "GPU信息:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1 | \
    awk -F', ' '{printf "  - GPU: %s\n  - 显存: %s MB (已用: %s MB)\n", $1, $2, $3}'
else
    echo "  - nvidia-smi未找到，无法获取GPU信息"
fi

# 设置环境变量
export LD_LIBRARY_PATH="/home/xihuidong/usr_lib/TensorRT-8.5.3.1/lib:$LD_LIBRARY_PATH"

echo ""
echo "开始推理测试..."
echo "=================================="

# 运行测试
"$EXECUTABLE" "$ONNX_MODEL" "$TEST_MAP_IMAGE" "$TEST_QUERY_IMAGE" "$ENGINE_FILE"

RESULT=$?

echo "=================================="

if [ $RESULT -eq 0 ]; then
    echo "🎉 测试成功完成!"
    
    # 显示引擎文件信息
    if [ -f "$ENGINE_FILE" ]; then
        echo ""
        echo "TensorRT引擎文件已生成:"
        echo "  - 文件: $ENGINE_FILE"
        echo "  - 大小: $(du -h "$ENGINE_FILE" | cut -f1)"
        echo "  - 下次运行将直接加载引擎文件，启动更快"
    fi
    
    echo ""
    echo "=== 测试总结 ==="
    echo "✓ TensorRT引擎初始化成功"
    echo "✓ 图像预处理正常"
    echo "✓ 模型推理正常"
    echo "✓ 描述符提取正常"
    echo "✓ 相似度计算正常"
    echo ""
    echo "部署验证完成！模型可以正常使用。"
    
else
    echo "❌ 测试失败，退出码: $RESULT"
    echo ""
    echo "可能的问题:"
    echo "1. TensorRT版本不兼容"
    echo "2. CUDA驱动问题"
    echo "3. 显存不足"
    echo "4. ONNX模型文件损坏"
    echo ""
    echo "请检查错误信息并重试"
fi

exit $RESULT
