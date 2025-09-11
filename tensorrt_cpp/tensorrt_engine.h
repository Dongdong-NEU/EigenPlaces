#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

/**
 * TensorRT推理引擎类
 * 用于加载ONNX模型并进行TensorRT推理
 */
class TensorRTEngine {
public:
    /**
     * 构造函数
     * @param onnx_file_path ONNX模型文件路径
     * @param engine_file_path TensorRT引擎文件路径（可选，用于缓存）
     * @param max_batch_size 最大批次大小
     * @param use_fp16 是否使用FP16精度
     */
    TensorRTEngine(const std::string& onnx_file_path, 
                   const std::string& engine_file_path = "",
                   int max_batch_size = 1, 
                   bool use_fp16 = true);
    
    /**
     * 析构函数
     */
    ~TensorRTEngine();
    
    /**
     * 初始化引擎
     * @return 是否成功初始化
     */
    bool initialize();
    
    /**
     * 执行推理
     * @param input_data 输入数据
     * @param output_data 输出数据
     * @return 是否推理成功
     */
    bool infer(const std::vector<float>& input_data, std::vector<float>& output_data);
    
    /**
     * 获取输入维度
     */
    std::vector<int> getInputDimensions() const { return input_dims_; }
    
    /**
     * 获取输出维度
     */
    std::vector<int> getOutputDimensions() const { return output_dims_; }

private:
    /**
     * 从ONNX文件构建引擎
     */
    bool buildEngineFromOnnx();
    
    /**
     * 从文件加载引擎
     */
    bool loadEngineFromFile();
    
    /**
     * 保存引擎到文件
     */
    bool saveEngineToFile();
    
    /**
     * 创建推理上下文
     */
    bool createInferenceContext();

private:
    std::string onnx_file_path_;
    std::string engine_file_path_;
    int max_batch_size_;
    bool use_fp16_;
    
    // TensorRT相关对象
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // 输入输出信息
    std::vector<int> input_dims_;
    std::vector<int> output_dims_;
    int input_size_;
    int output_size_;
    
    // CUDA内存
    void* gpu_input_buffer_;
    void* gpu_output_buffer_;
    void* cpu_input_buffer_;
    void* cpu_output_buffer_;
    
    // CUDA流
    cudaStream_t stream_;
    
    // Logger类
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            // 只输出警告和错误信息
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };
    
    Logger logger_;
};
