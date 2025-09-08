
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple

from eigenplaces_model.layers import Flatten, L2Norm, GeM

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
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为(B, C, H, W)
            
        Returns:
            描述符向量，形状为(B, fc_output_dim)，已进行L2归一化
        """
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

