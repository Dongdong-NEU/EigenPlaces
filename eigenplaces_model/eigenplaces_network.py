
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple

from eigenplaces_model.layers import Flatten, L2Norm, GeM, DLAUltraCompatibleL2Norm, NoSqrtManualL2Norm

# 各种骨干网络最后一个卷积层的通道数（在平均池化之前）
# 这些信息用于确定全连接层的输入维度
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,     # ResNet18最后卷积层输出512个通道
    "ResNet50": 2048,    # ResNet50最后卷积层输出2048个通道  
    "ResNet101": 2048,   # ResNet101最后卷积层输出2048个通道
    "ResNet152": 2048,   # ResNet152最后卷积层输出2048个通道
    "VGG16": 512,        # VGG16最后卷积层输出512个通道
}


class GeoLocalizationNet_(nn.Module):
    """
    EigenPlaces地理定位网络主类
    
    该网络用于视觉地点识别任务，将输入图像编码为固定维度的描述符向量。
    网络由两部分组成：
    1. 骨干网络（backbone）：提取图像特征
    2. 聚合层（aggregation）：将特征聚合为最终的描述符
    """
    def __init__(self, backbone : str, fc_output_dim : int):
        """
        初始化地理定位网络
        
        Args:
            backbone (str): 使用的torchvision骨干网络名称，必须是VGG16或ResNet系列
            fc_output_dim (int): 最后全连接层的输出维度，等价于描述符的维度
        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        
        # 获取预训练的骨干网络和特征维度
        self.backbone, features_dim = _get_backbone(backbone)
        
        # 构建聚合层：L2归一化 -> GeM池化 -> 展平 -> 全连接 -> L2归一化
        self.aggregation = nn.Sequential(
            L2Norm(),                                    # 对特征图进行L2归一化
            GeM(),                                       # 广义平均池化（Generalized Mean Pooling）
            Flatten(),                                   # 展平为1D向量
            nn.Linear(features_dim, fc_output_dim),      # 全连接层，输出指定维度的描述符
            L2Norm()                                     # 对最终描述符进行L2归一化
        )
    
    def forward(self, x):
        x = self.backbone(x)     # 通过骨干网络提取特征
        x = self.aggregation(x)  # 通过聚合层生成最终描述符
        return x

class NoSqrtDLACompatibleGeoLocalizationNet(nn.Module):

    def __init__(self, backbone: str, fc_output_dim: int, use_reciprocal: bool = True):
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        
        # 获取预训练的骨干网络和特征维度
        self.backbone, features_dim = _get_backbone(backbone)
        
        # 构建完全DLA兼容的聚合层
        self.aggregation = nn.Sequential(
            DLAUltraCompatibleL2Norm(features_dim, use_reciprocal=use_reciprocal),  # 无Sqrt的特征图L2归一化
            GeM(),                                                                  # 广义平均池化
            Flatten(),                                                              # 展平为1D向量
            nn.Linear(features_dim, fc_output_dim),                                 # 全连接层
            NoSqrtManualL2Norm(fc_output_dim, use_reciprocal=use_reciprocal)        # 无Sqrt的描述符L2归一化
        )
    
    def forward(self, x):

        x = self.backbone(x)     # 通过骨干网络提取特征
        x = self.aggregation(x)  # 通过聚合层生成最终描述符
        return x


def _get_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """
    根据骨干网络名称获取对应的torchvision预训练模型
    
    Args:
        backbone_name (str): 骨干网络名称，如'VGG16'或'ResNet18'
        
    Returns:
        torch.nn.Module: 对应的torchvision模型实例
        
    Examples:
        >>> model = _get_torchvision_model('ResNet18')
        >>> model = _get_torchvision_model('VGG16')
    """
    return getattr(torchvision.models, backbone_name.lower())()


def _get_backbone(backbone_name : str) -> Tuple[torch.nn.Module, int]:
    """
    构建并配置骨干网络
    
    该函数执行以下操作：
    1. 获取torchvision预训练模型
    2. 加载CosPlace预训练权重进行初始化
    3. 冻结部分层以进行微调
    4. 移除分类头，保留特征提取部分
    
    Args:
        backbone_name (str): 骨干网络名称
        
    Returns:
        Tuple[torch.nn.Module, int]: (处理后的骨干网络, 特征维度)
    """
    # 获取原始的torchvision模型
    backbone = _get_torchvision_model(backbone_name)

    # 使用CosPlace预训练权重初始化骨干网络
    logging.info("Loading pretrained backbone's weights from CosPlace")
    cosplace = torch.hub.load("gmberton/cosplace", "get_trained_model", backbone=backbone_name, fc_output_dim=512)
    
    # 只加载形状匹配的权重参数
    new_sd = {k1: v2 for (k1, v1), (k2, v2) in zip(backbone.state_dict().items(), cosplace.state_dict().items())
              if v1.shape == v2.shape}
    backbone.load_state_dict(new_sd, strict=False)

    if backbone_name.startswith("ResNet"):
        # 对于ResNet系列：冻结layer3之前的所有层，只训练layer3和layer4
        for name, child in backbone.named_children():
            if name == "layer3":  # 在layer3处停止冻结
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        
        # 移除平均池化层和全连接层，保留卷积特征提取部分
        layers = list(backbone.children())[:-2]  
    
    elif backbone_name == "VGG16":
        # 对于VGG16：移除最后的平均池化和全连接层
        layers = list(backbone.features.children())[:-2]
        
        # 冻结除最后5层外的所有层
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    # 重新构建骨干网络（仅包含特征提取部分）
    backbone = torch.nn.Sequential(*layers)
    
    # 获取特征维度
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim


def convert_to_no_sqrt_dla_compatible(pretrained_model: GeoLocalizationNet_, use_reciprocal: bool = True) -> NoSqrtDLACompatibleGeoLocalizationNet:

    # 获取原始模型的配置信息
    backbone_name = None
    fc_output_dim = None
    
    # 从模型结构推断配置
    for name, module in pretrained_model.named_modules():
        if isinstance(module, nn.Linear) and 'aggregation' in name:
            fc_output_dim = module.out_features
            break
    
    # 推断backbone类型
    features_dim = module.in_features
    for backbone, dim in CHANNELS_NUM_IN_LAST_CONV.items():
        if dim == features_dim:
            backbone_name = backbone
            break
    
    if backbone_name is None or fc_output_dim is None:
        raise ValueError("无法从预训练模型推断网络配置")
    
    print(f"检测到模型配置: backbone={backbone_name}, fc_output_dim={fc_output_dim}")
    print(f"使用{'倒数方法 (pow(-0.5))' if use_reciprocal else 'pow(0.5)方法'}")
    
    # 创建无Sqrt DLA兼容模型
    no_sqrt_model = NoSqrtDLACompatibleGeoLocalizationNet(backbone_name, fc_output_dim, use_reciprocal)
    
    # 复制权重
    pretrained_dict = pretrained_model.state_dict()
    no_sqrt_dict = no_sqrt_model.state_dict()
    
    # 复制匹配的权重
    matched_keys = []
    for key in no_sqrt_dict.keys():
        if key in pretrained_dict and no_sqrt_dict[key].shape == pretrained_dict[key].shape:
            no_sqrt_dict[key] = pretrained_dict[key]
            matched_keys.append(key)
    
    # 新参数已在构造函数中正确初始化
    print(f"成功复制 {len(matched_keys)} 个权重参数")
    print("DLAUltraCompatibleL2Norm和NoSqrtManualL2Norm的权重已初始化为全1向量")
    
    no_sqrt_model.load_state_dict(no_sqrt_dict)
    return no_sqrt_model

