
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    """
    广义平均池化（Generalized Mean Pooling）函数
    
    GeM池化是一种可学习的池化方法，通过参数p控制池化的性质：
    - p=1: 算术平均池化
    - p=∞: 最大池化
    - p=3: 介于两者之间，常用于图像检索任务
    
    Args:
        x: 输入特征图，形状为(B, C, H, W)
        p: 池化参数，控制池化的激进程度
        eps: 防止除零的小常数
        
    Returns:
        池化后的特征，形状为(B, C, 1, 1)
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    """
    广义平均池化层（Generalized Mean Pooling Layer）
    
    这是一个可学习的全局池化层，特别适用于图像检索任务。
    与传统的全局平均池化相比，GeM通过可学习参数p来调整池化的特性，
    能够更好地关注图像中的重要特征。
    """
    def __init__(self, p=3, eps=1e-6):
        """
        初始化GeM层
        
        Args:
            p (float): 初始池化参数，默认为3
            eps (float): 防止数值不稳定的小常数
        """
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)  # 可学习的池化参数
        self.eps = eps
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图，形状为(B, C, H, W)
            
        Returns:
            池化后的特征，形状为(B, C, 1, 1)
        """
        return gem(x, p=self.p, eps=self.eps)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(torch.nn.Module):
    """
    展平层
    
    将形状为(B, C, 1, 1)的张量展平为(B, C)的形状。
    专门用于处理全局池化后的特征，确保输入的空间维度为1x1。
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，期望形状为(B, C, 1, 1)
            
        Returns:
            展平后的张量，形状为(B, C)
            
        Raises:
            AssertionError: 如果输入的空间维度不是1x1
        """
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    """
    L2归一化层
    
    对输入张量进行L2归一化，使得每个向量的L2范数为1。
    这在度量学习和图像检索任务中非常重要，因为它确保了
    不同描述符之间的比较基于方向而非幅度。
    """
    def __init__(self, dim=1):
        """
        初始化L2归一化层
        
        Args:
            dim (int): 进行归一化的维度，默认为1（特征维度）
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            L2归一化后的张量，与输入形状相同但每个向量的L2范数为1
        """
        return F.normalize(x, p=2.0, dim=self.dim)
