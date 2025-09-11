#include "image_processor.h"
#include <iostream>
#include <cmath>
#include <algorithm>

ImageProcessor::ImageProcessor(int input_height, 
                               int input_width,
                               const std::vector<float>& mean,
                               const std::vector<float>& std)
    : input_height_(input_height)
    , input_width_(input_width)
    , mean_(mean)
    , std_(std) {
    
    if (mean_.size() != 3 || std_.size() != 3) {
        throw std::invalid_argument("均值和标准差必须包含3个值 (RGB)");
    }
}

bool ImageProcessor::preprocessImage(const std::string& image_path, std::vector<float>& output_data) {
    // 读取图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "[ImageProcessor] 无法读取图像: " << image_path << std::endl;
        return false;
    }
    
    std::cout << "[ImageProcessor] 加载图像: " << image_path 
              << " [" << image.cols << "x" << image.rows << "]" << std::endl;
    
    return preprocessImage(image, output_data);
}

bool ImageProcessor::preprocessImage(const cv::Mat& image, std::vector<float>& output_data) {
    cv::Mat processed_image;
    
    // 1. 转换颜色空间 BGR -> RGB
    cv::cvtColor(image, processed_image, cv::COLOR_BGR2RGB);
    
    // 2. 调整图像尺寸
    cv::resize(processed_image, processed_image, cv::Size(input_width_, input_height_));
    
    // 3. 转换为float类型并归一化到[0,1]
    processed_image.convertTo(processed_image, CV_32F, 1.0 / 255.0);
    
    // 4. 标准化 (减均值除标准差)
    std::vector<cv::Mat> channels;
    cv::split(processed_image, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }
    
    cv::merge(channels, processed_image);
    
    // 5. 转换为CHW格式
    hwcToChw(processed_image, output_data);
    
    std::cout << "[ImageProcessor] 图像预处理完成, 输出尺寸: " << output_data.size() << std::endl;
    return true;
}

void ImageProcessor::hwcToChw(const cv::Mat& image, std::vector<float>& output_data) {
    int channels = image.channels();
    int height = image.rows;
    int width = image.cols;
    
    output_data.resize(channels * height * width);
    
    std::vector<cv::Mat> channel_mats;
    cv::split(image, channel_mats);
    
    for (int c = 0; c < channels; ++c) {
        float* channel_data = output_data.data() + c * height * width;
        memcpy(channel_data, channel_mats[c].ptr<float>(), height * width * sizeof(float));
    }
}

void ImageProcessor::postprocessDescriptor(const std::vector<float>& raw_output, 
                                          std::vector<float>& normalized_descriptor) {
    // 计算L2范数
    float l2_norm = 0.0f;
    for (float val : raw_output) {
        l2_norm += val * val;
    }
    l2_norm = std::sqrt(l2_norm);
    
    // 避免除零
    if (l2_norm < 1e-12f) {
        l2_norm = 1e-12f;
    }
    
    // L2归一化
    normalized_descriptor.resize(raw_output.size());
    for (size_t i = 0; i < raw_output.size(); ++i) {
        normalized_descriptor[i] = raw_output[i] / l2_norm;
    }
    
    // 验证归一化结果
    float norm_check = 0.0f;
    for (float val : normalized_descriptor) {
        norm_check += val * val;
    }
    norm_check = std::sqrt(norm_check);
    
    std::cout << "[ImageProcessor] L2归一化完成, 范数: " << norm_check 
              << " (应该接近1.0)" << std::endl;
}

float ImageProcessor::computeCosineSimilarity(const std::vector<float>& desc1, 
                                             const std::vector<float>& desc2) {
    if (desc1.size() != desc2.size()) {
        std::cerr << "[ImageProcessor] 描述符维度不匹配" << std::endl;
        return -1.0f;
    }
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    for (size_t i = 0; i < desc1.size(); ++i) {
        dot_product += desc1[i] * desc2[i];
        norm1 += desc1[i] * desc1[i];
        norm2 += desc2[i] * desc2[i];
    }
    
    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    
    if (norm1 < 1e-12f || norm2 < 1e-12f) {
        return 0.0f;
    }
    
    return dot_product / (norm1 * norm2);
}

float ImageProcessor::computeEuclideanDistance(const std::vector<float>& desc1, 
                                              const std::vector<float>& desc2) {
    if (desc1.size() != desc2.size()) {
        std::cerr << "[ImageProcessor] 描述符维度不匹配" << std::endl;
        return -1.0f;
    }
    
    float distance = 0.0f;
    for (size_t i = 0; i < desc1.size(); ++i) {
        float diff = desc1[i] - desc2[i];
        distance += diff * diff;
    }
    
    return std::sqrt(distance);
}
