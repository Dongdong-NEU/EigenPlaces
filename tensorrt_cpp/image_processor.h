#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * 图像处理类
 * 负责EigenPlaces模型的图像预处理和后处理
 */
class ImageProcessor {
public:
    /**
     * 构造函数
     * @param input_height 输入图像高度
     * @param input_width 输入图像宽度
     * @param mean RGB均值 (ImageNet标准)
     * @param std RGB标准差 (ImageNet标准)
     */
    ImageProcessor(int input_height = 512, 
                   int input_width = 512,
                   const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                   const std::vector<float>& std = {0.229f, 0.224f, 0.225f});
    
    /**
     * 预处理图像
     * @param image_path 图像文件路径
     * @param output_data 输出的预处理数据
     * @return 是否成功
     */
    bool preprocessImage(const std::string& image_path, std::vector<float>& output_data);
    
    /**
     * 预处理OpenCV图像
     * @param image OpenCV图像
     * @param output_data 输出的预处理数据
     * @return 是否成功
     */
    bool preprocessImage(const cv::Mat& image, std::vector<float>& output_data);
    
    /**
     * 后处理：L2归一化描述符
     * @param raw_output 原始模型输出
     * @param normalized_descriptor L2归一化后的描述符
     */
    void postprocessDescriptor(const std::vector<float>& raw_output, 
                              std::vector<float>& normalized_descriptor);
    
    /**
     * 计算两个描述符的余弦相似度
     * @param desc1 描述符1
     * @param desc2 描述符2
     * @return 余弦相似度 [-1, 1]
     */
    float computeCosineSimilarity(const std::vector<float>& desc1, 
                                 const std::vector<float>& desc2);
    
    /**
     * 计算两个描述符的欧氏距离
     * @param desc1 描述符1
     * @param desc2 描述符2
     * @return 欧氏距离
     */
    float computeEuclideanDistance(const std::vector<float>& desc1, 
                                  const std::vector<float>& desc2);

private:
    int input_height_;
    int input_width_;
    std::vector<float> mean_;
    std::vector<float> std_;
    
    /**
     * 将HWC格式转换为CHW格式
     * @param image 输入图像 (HWC)
     * @param output_data 输出数据 (CHW)
     */
    void hwcToChw(const cv::Mat& image, std::vector<float>& output_data);
};
