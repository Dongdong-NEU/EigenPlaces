#!/bin/bash

# EigenPlaces TensorRT C++项目编译脚本
# 需要在bash环境下运行

set -e  # 遇到错误立即退出

echo "=== EigenPlaces TensorRT C++编译脚本 ==="

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查必要的环境
echo "步骤1: 检查编译环境..."

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo "错误: g++编译器未找到"
    exit 1
fi

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: cmake未找到"
    exit 1
fi

# 检查CUDA
if ! command -v nvcc &> /dev/null; then
    echo "错误: CUDA编译器未找到"
    exit 1
fi

# 检查TensorRT
TENSORRT_ROOT="/home/xihuidong/usr_lib/TensorRT-8.5.3.1"
if [ ! -d "$TENSORRT_ROOT" ]; then
    echo "错误: TensorRT未找到: $TENSORRT_ROOT"
    exit 1
fi

# 检查OpenCV
if ! pkg-config --exists opencv4; then
    echo "警告: OpenCV4未找到，尝试查找OpenCV..."
    if ! pkg-config --exists opencv; then
        echo "错误: OpenCV未找到"
        exit 1
    fi
fi

echo "✓ 编译环境检查通过"

# 显示版本信息
echo "环境信息:"
echo "  - g++版本: $(g++ --version | head -n1)"
echo "  - cmake版本: $(cmake --version | head -n1)"
echo "  - CUDA版本: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
echo "  - OpenCV版本: $(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv)"

# 创建构建目录
echo ""
echo "步骤2: 创建构建目录..."
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "✓ 构建目录创建完成: $BUILD_DIR"

# 运行CMake配置
echo ""
echo "步骤3: 运行CMake配置..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT="$TENSORRT_ROOT"

if [ $? -ne 0 ]; then
    echo "错误: CMake配置失败"
    exit 1
fi
echo "✓ CMake配置完成"

# 编译项目
echo ""
echo "步骤4: 编译项目..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi
echo "✓ 编译完成"

# 检查可执行文件
EXECUTABLE="EigenPlaces_TensorRT"
if [ ! -f "$EXECUTABLE" ]; then
    echo "错误: 可执行文件未生成: $EXECUTABLE"
    exit 1
fi

echo ""
echo "🎉 编译成功!"
echo "可执行文件位置: $(pwd)/$EXECUTABLE"

# 显示文件大小
echo "可执行文件大小: $(du -h $EXECUTABLE | cut -f1)"

# 显示依赖库
echo ""
echo "依赖库检查:"
ldd "$EXECUTABLE" | grep -E "(tensorrt|cuda|opencv)" | head -10

echo ""
echo "=== 编译完成 ==="
echo "使用方法:"
echo "  cd $(dirname $(pwd))"
echo "  ./build/$EXECUTABLE <onnx_file> <test_map_image> <test_query_image> [engine_file]"
echo ""
echo "示例:"
echo "  ./build/$EXECUTABLE ../models/eigenplaces_resnet50_onnx_fixheightwidth.onnx ../test_map.png ../test_query.png"
