
# 基于 https://github.com/MuggleWang/CosFace_pytorch/blob/master/layer.py
# CosFace损失函数的PyTorch实现，用于深度人脸识别和度量学习

import torch
import torch.nn as nn
from torch.nn import Parameter


def cosine_sim(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    计算两个张量之间的余弦相似度
    
    该函数计算x1中每个向量与x2中每个向量之间的余弦相似度。
    余弦相似度衡量两个向量方向的相似程度，范围为[-1, 1]。
    
    Args:
        x1 (torch.Tensor): 第一个张量，形状为(N, D)
        x2 (torch.Tensor): 第二个张量，形状为(M, D)  
        dim (int): 计算范数的维度
        eps (float): 防止除零的小常数
        
    Returns:
        torch.Tensor: 余弦相似度矩阵，形状为(N, M)
    """
    ip = torch.mm(x1, x2.t())                    # 计算内积矩阵
    w1 = torch.norm(x1, 2, dim)                  # 计算x1每行的L2范数
    w2 = torch.norm(x2, 2, dim)                  # 计算x2每行的L2范数
    return ip / torch.ger(w1, w2).clamp(min=eps) # 内积除以范数乘积，得到余弦相似度


class MarginCosineProduct(nn.Module):
    """
    大间隔余弦乘积层（CosFace损失函数的核心组件）
    
    CosFace是一种用于深度人脸识别的损失函数，通过在余弦空间中
    引入固定的间隔来增强类间分离性和类内紧凑性。
    
    公式：output = s * (cos(θ) - m * y)
    其中：
    - s: 缩放因子，控制特征的范围
    - m: 间隔参数，增加类间分离度
    - y: one-hot标签，决定是否对相应类别施加间隔惩罚
    
    Args:
        in_features (int): 输入特征的维度
        out_features (int): 输出类别数（分类器的权重数量）
        s (float): 缩放因子，默认30.0
        m (float): 间隔参数，默认0.40
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.40):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # 缩放因子
        self.m = m  # 间隔参数
        
        # 可学习的权重矩阵，每一行代表一个类别的原型向量
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier初始化
    
    def forward(self, inputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs (torch.Tensor): 输入特征，形状为(N, in_features)
            label (torch.Tensor): 真实标签，形状为(N,)
            
        Returns:
            torch.Tensor: CosFace损失的logits，形状为(N, out_features)
        """
        # 计算输入特征与所有类别原型之间的余弦相似度
        cosine = cosine_sim(inputs, self.weight)
        
        # 创建one-hot编码的标签矩阵
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # 对真实类别施加间隔惩罚：cos(θ) - m
        # 对其他类别保持原始余弦相似度
        output = self.s * (cosine - one_hot * self.m)
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'
