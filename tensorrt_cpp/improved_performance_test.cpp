// 改进的性能测试代码示例

void improvedPerformanceTest(TensorRTEngine& engine, const std::vector<float>& input_data) {
    std::cout << "=== 改进的性能测试 ===" << std::endl;
    
    const int num_warmup = 10;    // 增加预热次数
    const int num_tests = 100;    // 增加测试次数
    
    // 1. 充分预热
    std::cout << "预热阶段 (" << num_warmup << "次)..." << std::endl;
    for (int i = 0; i < num_warmup; ++i) {
        std::vector<float> dummy_output;
        engine.infer(input_data, dummy_output);
    }
    
    // 2. 稳定性测试
    std::vector<double> inference_times;
    std::cout << "性能测试 (" << num_tests << "次)..." << std::endl;
    
    for (int i = 0; i < num_tests; ++i) {
        // 使用更高精度的计时
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<float> dummy_output;
        engine.infer(input_data, dummy_output);
        
        // 确保GPU操作完成
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        inference_times.push_back(duration.count() / 1000000.0);  // 转换为毫秒
    }
    
    // 3. 统计分析
    std::sort(inference_times.begin(), inference_times.end());
    
    double avg_time = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
    double median_time = inference_times[inference_times.size() / 2];
    double p95_time = inference_times[static_cast<size_t>(inference_times.size() * 0.95)];
    double p99_time = inference_times[static_cast<size_t>(inference_times.size() * 0.99)];
    
    // 计算标准差
    double variance = 0.0;
    for (double t : inference_times) {
        variance += (t - avg_time) * (t - avg_time);
    }
    double std_dev = std::sqrt(variance / inference_times.size());
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "详细性能统计:" << std::endl;
    std::cout << "  - 平均时间:   " << avg_time << " ms" << std::endl;
    std::cout << "  - 中位数:     " << median_time << " ms" << std::endl;
    std::cout << "  - 最小时间:   " << inference_times.front() << " ms" << std::endl;
    std::cout << "  - 最大时间:   " << inference_times.back() << " ms" << std::endl;
    std::cout << "  - P95延迟:    " << p95_time << " ms" << std::endl;
    std::cout << "  - P99延迟:    " << p99_time << " ms" << std::endl;
    std::cout << "  - 标准差:     " << std_dev << " ms" << std::endl;
    std::cout << "  - 变异系数:   " << (std_dev / avg_time * 100) << "%" << std::endl;
    std::cout << "  - 吞吐量:     " << (1000.0 / avg_time) << " FPS" << std::endl;
}
