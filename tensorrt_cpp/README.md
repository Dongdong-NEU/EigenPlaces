# EigenPlaces TensorRT C++ 部署

本项目实现了EigenPlaces模型的TensorRT C++部署，支持固定尺寸ONNX模型的高性能推理。

## 🚀 功能特性

- **高性能推理**: 使用TensorRT优化的推理引擎
- **FP16支持**: 自动启用FP16精度以提升性能
- **引擎缓存**: 自动缓存TensorRT引擎文件，加速后续启动
- **完整流程**: 包含图像预处理、模型推理、后处理和相似度计算
- **性能测试**: 内置推理性能基准测试

## 📋 环境要求

### 系统要求
- Ubuntu 18.04/20.04/22.04
- CUDA 11.x 或 12.x
- TensorRT 8.5.3.1
- OpenCV 4.x
- CMake 3.12+
- g++ 7.0+

### 依赖库
- TensorRT 8.5.3.1 (安装在 `/home/xihuidong/usr_lib/TensorRT-8.5.3.1`)
- CUDA Runtime
- OpenCV (用于图像处理)
- cuDNN (TensorRT依赖)

## 🏗️ 项目结构

```
tensorrt_cpp/
├── README.md                 # 本文档
├── CMakeLists.txt            # CMake构建配置
├── build.sh                  # 编译脚本
├── run_test.sh               # 测试运行脚本
├── tensorrt_engine.h         # TensorRT引擎类头文件
├── tensorrt_engine.cpp       # TensorRT引擎类实现
├── image_processor.h         # 图像处理类头文件
├── image_processor.cpp       # 图像处理类实现
└── main.cpp                  # 主程序
```

## 🔧 编译步骤

### 1. 检查环境
确保所有依赖已正确安装：
```bash
# 检查CUDA
nvcc --version

# 检查OpenCV
pkg-config --modversion opencv4

# 检查TensorRT
ls /home/xihuidong/usr_lib/TensorRT-8.5.3.1/lib/
```

### 2. 编译项目
```bash
cd tensorrt_cpp
chmod +x build.sh
./build.sh
```

编译脚本会自动：
- 检查编译环境
- 创建构建目录
- 运行CMake配置
- 编译生成可执行文件

## 🚀 运行测试

### 快速测试
使用提供的测试脚本：
```bash
chmod +x run_test.sh
./run_test.sh
```

### 手动运行
```bash
./build/EigenPlaces_TensorRT <onnx_file> <test_map_image> <test_query_image> [engine_file]
```

**参数说明**:
- `onnx_file`: ONNX模型文件路径
- `test_map_image`: 测试地图图像路径
- `test_query_image`: 测试查询图像路径  
- `engine_file`: TensorRT引擎缓存文件路径（可选）

**示例**:
```bash
./build/EigenPlaces_TensorRT \
    ../models/eigenplaces_resnet50_onnx_fixheightwidth.onnx \
    ../test_map.png \
    ../test_query.png \
    ../models/eigenplaces_resnet50_fixheightwidth.engine
```

## 📊 输出结果

程序会输出详细的测试结果：

### 1. 初始化信息
- TensorRT引擎初始化状态
- 输入输出维度信息
- GPU和显存使用情况

### 2. 推理结果
- 每张图像的处理时间
- 描述符维度和数值范围
- L2归一化验证

### 3. 相似度分析
- **余弦相似度**: [-1, 1] 范围，越接近1越相似
- **欧氏距离**: 越小表示越相似
- **相似度解释**: 自动判断图像相似程度

### 4. 性能统计
- 平均推理时间
- 最小/最大推理时间
- 吞吐量 (FPS)

## 🔍 预期结果示例

```
=== 相似度分析结果 ===
余弦相似度: 0.923456
欧氏距离:   0.387123

=== 相似度解释 ===
🎯 结论: 两张图像非常相似 (余弦相似度 > 0.8)

=== 性能统计 ===
推理时间统计:
  - 平均时间: 12.34 ms
  - 最小时间: 11.23 ms  
  - 最大时间: 15.67 ms
  - 吞吐量:   81.0 FPS
```

## ⚡ 性能优化

### 1. 引擎缓存
首次运行会构建TensorRT引擎（耗时较长），后续运行会直接加载缓存的引擎文件，大大加速启动时间。

### 2. FP16精度
自动检测GPU是否支持FP16，并启用以提升推理性能。

### 3. CUDA流
使用CUDA流进行异步内存传输，提升整体吞吐量。

## 🐛 故障排除

### 编译错误
1. **TensorRT未找到**: 检查路径 `/home/xihuidong/usr_lib/TensorRT-8.5.3.1`
2. **CUDA未找到**: 确保CUDA已正确安装并添加到PATH
3. **OpenCV未找到**: 安装OpenCV开发包

### 运行时错误
1. **显存不足**: 减少批次大小或使用更小的模型
2. **库文件未找到**: 检查LD_LIBRARY_PATH设置
3. **ONNX模型不兼容**: 确保使用固定尺寸的ONNX模型

### 性能问题
1. **推理速度慢**: 检查是否启用了FP16精度
2. **内存泄漏**: 检查CUDA内存释放
3. **CPU使用率高**: 优化图像预处理流程

## 📝 技术细节

### 模型要求
- **输入格式**: RGB图像，尺寸512x512
- **输入数据类型**: FP32
- **预处理**: ImageNet标准化 (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **输出格式**: 2048维特征向量
- **后处理**: L2归一化

### 内存管理
- 自动管理GPU和CPU内存
- 使用CUDA流进行异步操作
- 析构函数确保内存正确释放

### 线程安全
- TensorRT引擎支持多线程推理
- 图像处理器是线程安全的
- 可以并行处理多张图像

## 📄 许可证

本项目遵循与EigenPlaces主项目相同的许可证。
