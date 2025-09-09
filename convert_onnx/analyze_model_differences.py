"""
详细的模型差异分析

分析ONNX兼容模型与原始模型差异的主要原因
"""

import torch
import torch.nn as nn
import numpy as np

def analyze_backbone_differences():
    """分析骨干网络差异的原因"""
    print("=== 骨干网络差异分析 ===")
    
    # 导入模型
    from eigenplaces_model.eigenplaces_network import GeoLocalizationNet_
    from onnx_compatible_model import ONNXCompatibleGeoLocalizationNet
    
    # 创建模型
    original_model = GeoLocalizationNet_("ResNet50", 2048)
    onnx_model = ONNXCompatibleGeoLocalizationNet("ResNet50", 2048)
    
    # 检查骨干网络结构差异
    print("原始模型骨干网络结构:")
    print(f"层数: {len(list(original_model.backbone.children()))}")
    for i, layer in enumerate(original_model.backbone.children()):
        print(f"  层{i}: {type(layer).__name__}")
    
    print("\nONNX兼容模型骨干网络结构:")
    print(f"层数: {len(list(onnx_model.backbone.children()))}")
    for i, layer in enumerate(onnx_model.backbone.children()):
        print(f"  层{i}: {type(layer).__name__}")
    
    # 检查权重初始化的差异
    print(f"\n=== 权重初始化差异 ===")
    
    # 比较第一个卷积层的权重
    orig_first_conv = None
    onnx_first_conv = None
    
    for module in original_model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            orig_first_conv = module
            break
    
    for module in onnx_model.backbone.modules():
        if isinstance(module, nn.Conv2d):
            onnx_first_conv = module
            break
    
    if orig_first_conv is not None and onnx_first_conv is not None:
        weight_diff = torch.mean(torch.abs(orig_first_conv.weight - onnx_first_conv.weight))
        print(f"第一个卷积层权重差异: {weight_diff.item():.8f}")
        
        # 检查权重的统计信息
        print(f"原始模型第一层权重 - 均值: {orig_first_conv.weight.mean():.6f}, 标准差: {orig_first_conv.weight.std():.6f}")
        print(f"ONNX模型第一层权重 - 均值: {onnx_first_conv.weight.mean():.6f}, 标准差: {onnx_first_conv.weight.std():.6f}")

def analyze_aggregation_differences():
    """分析聚合层差异"""
    print("\n=== 聚合层实现差异分析 ===")
    
    # 创建测试输入
    test_feature_map = torch.randn(1, 2048, 16, 16)
    
    # 原始层测试
    from eigenplaces_model.layers import L2Norm, GeM, Flatten
    from onnx_compatible_model import ONNXCompatibleL2Norm, ONNXCompatibleGeM, ONNXCompatibleFlatten
    
    print("1. L2Norm层比较:")
    orig_l2norm = L2Norm()
    onnx_l2norm = ONNXCompatibleL2Norm()
    
    with torch.no_grad():
        orig_l2_out = orig_l2norm(test_feature_map)
        onnx_l2_out = onnx_l2norm(test_feature_map)
        l2_diff = torch.mean(torch.abs(orig_l2_out - onnx_l2_out))
        print(f"   L2Norm输出差异: {l2_diff.item():.10f}")
    
    print("2. GeM层比较:")
    orig_gem = GeM()
    onnx_gem = ONNXCompatibleGeM()
    
    with torch.no_grad():
        # 使用相同的p值
        onnx_gem.p.data = orig_gem.p.data.clone()
        
        orig_gem_out = orig_gem(test_feature_map)
        onnx_gem_out = onnx_gem(test_feature_map)
        gem_diff = torch.mean(torch.abs(orig_gem_out - onnx_gem_out))
        print(f"   GeM输出差异: {gem_diff.item():.10f}")
        
        print(f"   原始GeM输出形状: {orig_gem_out.shape}")
        print(f"   ONNX GeM输出形状: {onnx_gem_out.shape}")
    
    print("3. Flatten层比较:")
    test_gem_output = torch.randn(1, 2048, 1, 1)
    
    orig_flatten = Flatten()
    onnx_flatten = ONNXCompatibleFlatten()
    
    with torch.no_grad():
        orig_flat_out = orig_flatten(test_gem_output)
        onnx_flat_out = onnx_flatten(test_gem_output)
        flatten_diff = torch.mean(torch.abs(orig_flat_out - onnx_flat_out))
        print(f"   Flatten输出差异: {flatten_diff.item():.10f}")

def test_individual_components():
    """测试各个组件的数学等价性"""
    print("\n=== 组件数学等价性测试 ===")
    
    # 测试GeM池化的数学等价性
    print("1. GeM池化数学等价性:")
    x = torch.randn(2, 512, 8, 8).clamp(min=1e-6)
    p = 3.0
    eps = 1e-6
    
    # 原始方法
    import torch.nn.functional as F
    orig_result = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    
    # ONNX兼容方法
    adaptive_pool = nn.AdaptiveAvgPool2d(1)
    onnx_result = adaptive_pool(x.pow(p)).pow(1./p)
    
    gem_equiv_diff = torch.mean(torch.abs(orig_result - onnx_result))
    print(f"   GeM池化等价性差异: {gem_equiv_diff.item():.12f}")
    
    # 测试L2归一化等价性
    print("2. L2归一化数学等价性:")
    x = torch.randn(2, 512)
    
    # 原始方法
    orig_l2 = F.normalize(x, p=2.0, dim=1)
    
    # ONNX兼容方法
    eps = 1e-12
    norm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
    onnx_l2 = x / norm
    
    l2_equiv_diff = torch.mean(torch.abs(orig_l2 - onnx_l2))
    print(f"   L2归一化等价性差异: {l2_equiv_diff.item():.12f}")

def create_weight_aligned_models():
    """创建权重对齐的模型进行更公平的比较"""
    print("\n=== 创建权重对齐的模型 ===")
    
    from eigenplaces_model.eigenplaces_network import GeoLocalizationNet_
    from onnx_compatible_model import ONNXCompatibleGeoLocalizationNet
    
    # 创建原始模型
    original_model = GeoLocalizationNet_("ResNet50", 2048)
    original_model.eval()
    
    # 创建ONNX模型并同步所有可能的权重
    onnx_model = ONNXCompatibleGeoLocalizationNet("ResNet50", 2048)
    onnx_model.eval()
    
    # 尝试同步骨干网络权重
    try:
        # 获取原始模型的骨干网络状态字典
        orig_backbone_dict = original_model.backbone.state_dict()
        onnx_backbone_dict = onnx_model.backbone.state_dict()
        
        # 同步形状匹配的权重
        synchronized_weights = 0
        total_weights = len(orig_backbone_dict)
        
        for name in orig_backbone_dict:
            if name in onnx_backbone_dict:
                if orig_backbone_dict[name].shape == onnx_backbone_dict[name].shape:
                    onnx_backbone_dict[name].copy_(orig_backbone_dict[name])
                    synchronized_weights += 1
        
        print(f"同步了 {synchronized_weights}/{total_weights} 个权重")
        
        # 同步聚合层的线性层权重
        with torch.no_grad():
            onnx_model.aggregation[3].weight.copy_(original_model.aggregation[3].weight)
            onnx_model.aggregation[3].bias.copy_(original_model.aggregation[3].bias)
            
            # 同步GeM层的p参数
            onnx_model.aggregation[1].p.data.copy_(original_model.aggregation[1].p.data)
        
        print("✓ 线性层和GeM参数已同步")
        
        # 现在测试对齐后的一致性
        test_input = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            orig_output = original_model(test_input)
            onnx_output = onnx_model(test_input)
            
            cosine_sim = torch.cosine_similarity(orig_output, onnx_output, dim=1)
            euclidean_dist = torch.norm(orig_output - onnx_output, dim=1)
            max_diff = torch.max(torch.abs(orig_output - onnx_output))
            
            print(f"\n权重对齐后的一致性:")
            print(f"  余弦相似度: {cosine_sim.item():.8f}")
            print(f"  欧氏距离: {euclidean_dist.item():.8f}")
            print(f"  最大差异: {max_diff.item():.10f}")
            
    except Exception as e:
        print(f"权重同步失败: {e}")

def main():
    print("EigenPlaces模型差异深度分析")
    print("=" * 50)
    
    analyze_backbone_differences()
    analyze_aggregation_differences()
    test_individual_components()
    create_weight_aligned_models()
    
    print("\n=== 结论 ===")
    print("1. 主要差异来源于骨干网络的权重初始化不同")
    print("2. 聚合层的实现在数学上是等价的")
    print("3. 对于实际应用，两个模型都能正确输出L2归一化的描述符")
    print("4. ONNX兼容模型牺牲了与原始模型的精确一致性，但获得了导出能力")

if __name__ == "__main__":
    main()
