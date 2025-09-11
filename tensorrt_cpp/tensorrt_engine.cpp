#include "tensorrt_engine.h"
#include <cassert>
#include <chrono>

TensorRTEngine::TensorRTEngine(const std::string& onnx_file_path, 
                               const std::string& engine_file_path,
                               int max_batch_size, 
                               bool use_fp16)
    : onnx_file_path_(onnx_file_path)
    , engine_file_path_(engine_file_path)
    , max_batch_size_(max_batch_size)
    , use_fp16_(use_fp16)
    , gpu_input_buffer_(nullptr)
    , gpu_output_buffer_(nullptr)
    , cpu_input_buffer_(nullptr)
    , cpu_output_buffer_(nullptr)
    , stream_(nullptr) {
}

TensorRTEngine::~TensorRTEngine() {
    // 释放CUDA内存
    if (gpu_input_buffer_) {
        cudaFree(gpu_input_buffer_);
    }
    if (gpu_output_buffer_) {
        cudaFree(gpu_output_buffer_);
    }
    if (cpu_input_buffer_) {
        free(cpu_input_buffer_);
    }
    if (cpu_output_buffer_) {
        free(cpu_output_buffer_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TensorRTEngine::initialize() {
    std::cout << "[TensorRT] 初始化TensorRT引擎..." << std::endl;
    
    // 创建CUDA流
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "[TensorRT] 创建CUDA流失败" << std::endl;
        return false;
    }
    
    // 尝试从文件加载引擎
    bool engine_loaded = false;
    if (!engine_file_path_.empty()) {
        engine_loaded = loadEngineFromFile();
    }
    
    // 如果无法从文件加载，则从ONNX构建
    if (!engine_loaded) {
        std::cout << "[TensorRT] 从ONNX文件构建引擎..." << std::endl;
        if (!buildEngineFromOnnx()) {
            std::cerr << "[TensorRT] 从ONNX构建引擎失败" << std::endl;
            return false;
        }
        
        // 保存引擎到文件
        if (!engine_file_path_.empty()) {
            saveEngineToFile();
        }
    }
    
    // 创建推理上下文
    if (!createInferenceContext()) {
        std::cerr << "[TensorRT] 创建推理上下文失败" << std::endl;
        return false;
    }
    
    std::cout << "[TensorRT] 引擎初始化成功" << std::endl;
    std::cout << "[TensorRT] 输入维度: [";
    for (size_t i = 0; i < input_dims_.size(); ++i) {
        std::cout << input_dims_[i];
        if (i < input_dims_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "[TensorRT] 输出维度: [";
    for (size_t i = 0; i < output_dims_.size(); ++i) {
        std::cout << output_dims_[i];
        if (i < output_dims_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    return true;
}

bool TensorRTEngine::buildEngineFromOnnx() {
    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "[TensorRT] 创建builder失败" << std::endl;
        return false;
    }
    
    // 创建网络定义
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "[TensorRT] 创建网络定义失败" << std::endl;
        return false;
    }
    
    // 创建ONNX解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "[TensorRT] 创建ONNX解析器失败" << std::endl;
        return false;
    }
    
    // 解析ONNX文件
    std::cout << "[TensorRT] 解析ONNX文件: " << onnx_file_path_ << std::endl;
    if (!parser->parseFromFile(onnx_file_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[TensorRT] 解析ONNX文件失败" << std::endl;
        return false;
    }
    
    // 创建构建配置
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "[TensorRT] 创建构建配置失败" << std::endl;
        return false;
    }
    
    // 设置最大工作空间大小 (1GB)
    config->setMaxWorkspaceSize(1ULL << 30);
    
    // 如果使用FP16，启用FP16模式
    if (use_fp16_ && builder->platformHasFastFp16()) {
        std::cout << "[TensorRT] 启用FP16精度" << std::endl;
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else {
        std::cout << "[TensorRT] 使用FP32精度" << std::endl;
    }
    
    // 检查是否有动态维度，如果有则创建优化配置文件
    bool has_dynamic_shapes = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        auto dims = network->getInput(i)->getDimensions();
        for (int j = 0; j < dims.nbDims; ++j) {
            if (dims.d[j] == -1) {
                has_dynamic_shapes = true;
                break;
            }
        }
        if (has_dynamic_shapes) break;
    }
    
    if (has_dynamic_shapes) {
        std::cout << "[TensorRT] 检测到动态维度，创建优化配置文件..." << std::endl;
        
        // 创建优化配置文件
        auto profile = builder->createOptimizationProfile();
        
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto input = network->getInput(i);
            auto dims = input->getDimensions();
            const char* input_name = input->getName();
            
            // 创建维度配置 (min, opt, max)
            nvinfer1::Dims min_dims = dims;
            nvinfer1::Dims opt_dims = dims;
            nvinfer1::Dims max_dims = dims;
            
            // 设置batch维度
            if (dims.d[0] == -1) {  // 动态batch维度
                min_dims.d[0] = 1;
                opt_dims.d[0] = max_batch_size_;
                max_dims.d[0] = max_batch_size_;
            }
            
            // 设置其他动态维度
            for (int j = 1; j < dims.nbDims; ++j) {
                if (dims.d[j] == -1) {
                    // 假设是512x512的图像尺寸
                    if (j == 2 || j == 3) {  // 高度和宽度
                        min_dims.d[j] = 224;
                        opt_dims.d[j] = 512;
                        max_dims.d[j] = 1024;
                    } else {
                        min_dims.d[j] = 1;
                        opt_dims.d[j] = dims.d[j] > 0 ? dims.d[j] : 1;
                        max_dims.d[j] = dims.d[j] > 0 ? dims.d[j] : 1;
                    }
                }
            }
            
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims);
            
            std::cout << "[TensorRT] 输入 " << input_name << " 优化配置:" << std::endl;
            std::cout << "  - Min: ["; for (int k = 0; k < min_dims.nbDims; ++k) std::cout << min_dims.d[k] << (k < min_dims.nbDims-1 ? ", " : ""); std::cout << "]" << std::endl;
            std::cout << "  - Opt: ["; for (int k = 0; k < opt_dims.nbDims; ++k) std::cout << opt_dims.d[k] << (k < opt_dims.nbDims-1 ? ", " : ""); std::cout << "]" << std::endl;
            std::cout << "  - Max: ["; for (int k = 0; k < max_dims.nbDims; ++k) std::cout << max_dims.d[k] << (k < max_dims.nbDims-1 ? ", " : ""); std::cout << "]" << std::endl;
        }
        
        config->addOptimizationProfile(profile);
    }
    
    // 构建引擎
    std::cout << "[TensorRT] 构建TensorRT引擎..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    if (!engine_) {
        std::cerr << "[TensorRT] 构建引擎失败" << std::endl;
        return false;
    }
    
    std::cout << "[TensorRT] 引擎构建完成，耗时: " << duration.count() << " 秒" << std::endl;
    return true;
}

bool TensorRTEngine::loadEngineFromFile() {
    std::ifstream file(engine_file_path_, std::ios::binary);
    if (!file.good()) {
        std::cout << "[TensorRT] 引擎文件不存在: " << engine_file_path_ << std::endl;
        return false;
    }
    
    std::cout << "[TensorRT] 从文件加载引擎: " << engine_file_path_ << std::endl;
    
    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 读取引擎数据
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();
    
    // 创建运行时
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "[TensorRT] 创建运行时失败" << std::endl;
        return false;
    }
    
    // 反序列化引擎
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), file_size));
    
    if (!engine_) {
        std::cerr << "[TensorRT] 反序列化引擎失败" << std::endl;
        return false;
    }
    
    std::cout << "[TensorRT] 引擎加载成功" << std::endl;
    return true;
}

bool TensorRTEngine::saveEngineToFile() {
    if (!engine_ || engine_file_path_.empty()) {
        return false;
    }
    
    std::cout << "[TensorRT] 保存引擎到文件: " << engine_file_path_ << std::endl;
    
    // 序列化引擎
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    if (!serialized_engine) {
        std::cerr << "[TensorRT] 序列化引擎失败" << std::endl;
        return false;
    }
    
    // 写入文件
    std::ofstream file(engine_file_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[TensorRT] 无法创建引擎文件: " << engine_file_path_ << std::endl;
        return false;
    }
    
    file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    file.close();
    
    std::cout << "[TensorRT] 引擎保存成功" << std::endl;
    return true;
}

bool TensorRTEngine::createInferenceContext() {
    // 如果没有运行时，创建一个
    if (!runtime_) {
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) {
            std::cerr << "[TensorRT] 创建运行时失败" << std::endl;
            return false;
        }
    }
    
    // 创建执行上下文
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[TensorRT] 创建执行上下文失败" << std::endl;
        return false;
    }
    
    // 获取输入输出维度信息
    int num_bindings = engine_->getNbBindings();
    std::cout << "[TensorRT] 绑定数量: " << num_bindings << std::endl;
    
    for (int i = 0; i < num_bindings; ++i) {
        const char* binding_name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        bool is_input = engine_->bindingIsInput(i);
        
        std::cout << "[TensorRT] 绑定 " << i << ": " << binding_name 
                  << (is_input ? " (输入)" : " (输出)") << std::endl;
        
        // 计算尺寸
        int size = 1;
        std::vector<int> shape;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
            shape.push_back(dims.d[j]);
        }
        
        if (is_input) {
            input_dims_ = shape;
            input_size_ = size;
        } else {
            output_dims_ = shape;
            output_size_ = size;
        }
    }
    
    // 分配GPU内存
    if (cudaMalloc(&gpu_input_buffer_, input_size_ * sizeof(float)) != cudaSuccess) {
        std::cerr << "[TensorRT] 分配GPU输入内存失败" << std::endl;
        return false;
    }
    
    if (cudaMalloc(&gpu_output_buffer_, output_size_ * sizeof(float)) != cudaSuccess) {
        std::cerr << "[TensorRT] 分配GPU输出内存失败" << std::endl;
        return false;
    }
    
    // 分配CPU内存
    cpu_input_buffer_ = malloc(input_size_ * sizeof(float));
    cpu_output_buffer_ = malloc(output_size_ * sizeof(float));
    
    if (!cpu_input_buffer_ || !cpu_output_buffer_) {
        std::cerr << "[TensorRT] 分配CPU内存失败" << std::endl;
        return false;
    }
    
    return true;
}

bool TensorRTEngine::infer(const std::vector<float>& input_data, std::vector<float>& output_data) {
    if (input_data.size() != static_cast<size_t>(input_size_)) {
        std::cerr << "[TensorRT] 输入数据大小不匹配: " << input_data.size() 
                  << " vs " << input_size_ << std::endl;
        return false;
    }
    
    // 对于动态形状，设置输入维度
    if (engine_->hasImplicitBatchDimension() == false) {
        // 显式batch模式，需要设置输入形状
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) {
                // 设置输入维度为 [1, 3, 512, 512]
                nvinfer1::Dims input_dims;
                input_dims.nbDims = 4;
                input_dims.d[0] = 1;      // batch size
                input_dims.d[1] = 3;      // channels
                input_dims.d[2] = 512;    // height
                input_dims.d[3] = 512;    // width
                
                if (!context_->setBindingDimensions(i, input_dims)) {
                    std::cerr << "[TensorRT] 设置输入维度失败" << std::endl;
                    return false;
                }
            }
        }
        
        // 检查所有输入维度是否有效
        if (!context_->allInputDimensionsSpecified()) {
            std::cerr << "[TensorRT] 输入维度未完全指定" << std::endl;
            return false;
        }
    }
    
    // 复制输入数据到CPU缓冲区
    memcpy(cpu_input_buffer_, input_data.data(), input_size_ * sizeof(float));
    
    // 复制输入数据到GPU
    if (cudaMemcpyAsync(gpu_input_buffer_, cpu_input_buffer_, 
                       input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        std::cerr << "[TensorRT] 复制输入数据到GPU失败" << std::endl;
        return false;
    }
    
    // 设置绑定
    void* bindings[] = {gpu_input_buffer_, gpu_output_buffer_};
    
    // 执行推理
    if (!context_->enqueueV2(bindings, stream_, nullptr)) {
        std::cerr << "[TensorRT] 推理执行失败" << std::endl;
        return false;
    }
    
    // 复制输出数据到CPU
    if (cudaMemcpyAsync(cpu_output_buffer_, gpu_output_buffer_, 
                       output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
        std::cerr << "[TensorRT] 复制输出数据到CPU失败" << std::endl;
        return false;
    }
    
    // 等待CUDA操作完成
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        std::cerr << "[TensorRT] CUDA同步失败" << std::endl;
        return false;
    }
    
    // 复制输出数据
    output_data.resize(output_size_);
    memcpy(output_data.data(), cpu_output_buffer_, output_size_ * sizeof(float));
    
    return true;
}
