#include "tensorrt_engine.h"
#include "image_processor.h"
#include <iostream>
#include <chrono>
#include <iomanip>

/**
 * EigenPlaces TensorRT推理主程序
 * 用于测试固定尺寸ONNX模型的TensorRT部署
 */

void printUsage(const char* program_name) {
    std::cout << "用法: " << program_name << " <onnx_file> <test_map_image> <test_query_image> [engine_file]" << std::endl;
    std::cout << "参数:" << std::endl;
    std::cout << "  onnx_file        - ONNX模型文件路径" << std::endl;
    std::cout << "  test_map_image   - 测试地图图像路径" << std::endl;
    std::cout << "  test_query_image - 测试查询图像路径" << std::endl;
    std::cout << "  engine_file      - TensorRT引擎文件路径 (可选，用于缓存)" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << program_name << " model.onnx test_map.png test_query.png" << std::endl;
    std::cout << "  " << program_name << " model.onnx test_map.png test_query.png model.engine" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== EigenPlaces TensorRT推理测试 ===" << std::endl;
    
    // 检查命令行参数
    if (argc < 4) {
        printUsage(argv[0]);
        return -1;
    }
    
    std::string onnx_file = argv[1];
    std::string test_map_image = argv[2];
    std::string test_query_image = argv[3];
    std::string engine_file = (argc >= 5) ? argv[4] : "";
    
    std::cout << "ONNX模型文件: " << onnx_file << std::endl;
    std::cout << "测试地图图像: " << test_map_image << std::endl;
    std::cout << "测试查询图像: " << test_query_image << std::endl;
    if (!engine_file.empty()) {
        std::cout << "引擎缓存文件: " << engine_file << std::endl;
    }
    std::cout << std::endl;
    
    try {
        // 1. 初始化TensorRT引擎
        std::cout << "步骤1: 初始化TensorRT引擎" << std::endl;
        TensorRTEngine engine(onnx_file, engine_file, 1, true);  // 批次大小=1, 使用FP16
        
        if (!engine.initialize()) {
            std::cerr << "TensorRT引擎初始化失败" << std::endl;
            return -1;
        }
        std::cout << "✓ TensorRT引擎初始化成功" << std::endl << std::endl;
        
        // 2. 初始化图像处理器
        std::cout << "步骤2: 初始化图像处理器" << std::endl;
        ImageProcessor processor(512, 512);  // 固定尺寸 512x512
        std::cout << "✓ 图像处理器初始化成功" << std::endl << std::endl;
        
        // 3. 处理地图图像
        std::cout << "步骤3: 处理地图图像" << std::endl;
        std::vector<float> map_input_data;
        if (!processor.preprocessImage(test_map_image, map_input_data)) {
            std::cerr << "地图图像预处理失败" << std::endl;
            return -1;
        }
        
        // 推理地图图像
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> map_raw_output;
        if (!engine.infer(map_input_data, map_raw_output)) {
            std::cerr << "地图图像推理失败" << std::endl;
            return -1;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 后处理地图描述符
        std::vector<float> map_descriptor;
        processor.postprocessDescriptor(map_raw_output, map_descriptor);
        
        std::cout << "✓ 地图图像处理完成" << std::endl;
        std::cout << "  - 推理时间: " << inference_time.count() << " ms" << std::endl;
        std::cout << "  - 描述符维度: " << map_descriptor.size() << std::endl;
        std::cout << "  - 描述符范围: [" << std::fixed << std::setprecision(6) 
                  << *std::min_element(map_descriptor.begin(), map_descriptor.end()) << ", "
                  << *std::max_element(map_descriptor.begin(), map_descriptor.end()) << "]" << std::endl;
        std::cout << std::endl;
        
        // 4. 处理查询图像
        std::cout << "步骤4: 处理查询图像" << std::endl;
        std::vector<float> query_input_data;
        if (!processor.preprocessImage(test_query_image, query_input_data)) {
            std::cerr << "查询图像预处理失败" << std::endl;
            return -1;
        }
        
        // 推理查询图像
        start_time = std::chrono::high_resolution_clock::now();
        std::vector<float> query_raw_output;
        if (!engine.infer(query_input_data, query_raw_output)) {
            std::cerr << "查询图像推理失败" << std::endl;
            return -1;
        }
        end_time = std::chrono::high_resolution_clock::now();
        inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // 后处理查询描述符
        std::vector<float> query_descriptor;
        processor.postprocessDescriptor(query_raw_output, query_descriptor);
        
        std::cout << "✓ 查询图像处理完成" << std::endl;
        std::cout << "  - 推理时间: " << inference_time.count() << " ms" << std::endl;
        std::cout << "  - 描述符维度: " << query_descriptor.size() << std::endl;
        std::cout << "  - 描述符范围: [" << std::fixed << std::setprecision(6) 
                  << *std::min_element(query_descriptor.begin(), query_descriptor.end()) << ", "
                  << *std::max_element(query_descriptor.begin(), query_descriptor.end()) << "]" << std::endl;
        std::cout << std::endl;
        
        // 5. 计算相似度
        std::cout << "步骤5: 计算图像相似度" << std::endl;
        
        float cosine_similarity = processor.computeCosineSimilarity(map_descriptor, query_descriptor);
        float euclidean_distance = processor.computeEuclideanDistance(map_descriptor, query_descriptor);
        
        std::cout << "=== 相似度分析结果 ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "余弦相似度: " << cosine_similarity << std::endl;
        std::cout << "欧氏距离:   " << euclidean_distance << std::endl;
        std::cout << std::endl;
        
        // 6. 相似度解释
        std::cout << "=== 相似度解释 ===" << std::endl;
        std::cout << "余弦相似度范围: [-1, 1]" << std::endl;
        std::cout << "  - 1.0  : 完全相同" << std::endl;
        std::cout << "  - 0.0  : 无关" << std::endl;
        std::cout << "  - -1.0 : 完全相反" << std::endl;
        std::cout << std::endl;
        
        if (cosine_similarity > 0.8) {
            std::cout << "🎯 结论: 两张图像非常相似 (余弦相似度 > 0.8)" << std::endl;
        } else if (cosine_similarity > 0.5) {
            std::cout << "✅ 结论: 两张图像比较相似 (余弦相似度 > 0.5)" << std::endl;
        } else if (cosine_similarity > 0.2) {
            std::cout << "⚠️  结论: 两张图像有一定相似性 (余弦相似度 > 0.2)" << std::endl;
        } else {
            std::cout << "❌ 结论: 两张图像相似度较低 (余弦相似度 <= 0.2)" << std::endl;
        }
        std::cout << std::endl;
        
        // 7. 性能统计
        std::cout << "=== 性能统计 ===" << std::endl;
        
        // 进行多次推理测试性能
        const int num_warmup = 5;
        const int num_tests = 20;
        
        std::cout << "进行性能测试 (预热" << num_warmup << "次, 测试" << num_tests << "次)..." << std::endl;
        
        // 预热
        for (int i = 0; i < num_warmup; ++i) {
            std::vector<float> dummy_output;
            engine.infer(map_input_data, dummy_output);
        }
        
        // 性能测试
        std::vector<double> inference_times;
        for (int i = 0; i < num_tests; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<float> dummy_output;
            engine.infer(map_input_data, dummy_output);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            inference_times.push_back(duration.count() / 1000.0);  // 转换为毫秒
        }
        
        // 计算统计信息
        double sum = 0.0;
        for (double t : inference_times) {
            sum += t;
        }
        double avg_time = sum / inference_times.size();
        
        auto min_time = *std::min_element(inference_times.begin(), inference_times.end());
        auto max_time = *std::max_element(inference_times.begin(), inference_times.end());
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "推理时间统计:" << std::endl;
        std::cout << "  - 平均时间: " << avg_time << " ms" << std::endl;
        std::cout << "  - 最小时间: " << min_time << " ms" << std::endl;
        std::cout << "  - 最大时间: " << max_time << " ms" << std::endl;
        std::cout << "  - 吞吐量:   " << std::fixed << std::setprecision(1) << (1000.0 / avg_time) << " FPS" << std::endl;
        std::cout << std::endl;
        
        std::cout << "🎉 TensorRT部署测试完成!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
