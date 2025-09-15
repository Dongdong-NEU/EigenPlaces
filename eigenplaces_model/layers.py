
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):
    """
    广义平均池化(Generalized Mean Pooling)函数
    
    GeM池化是一种可学习的池化方法,通过参数p控制池化的性质：
    - p=1: 算术平均池化
    - p=∞: 最大池化
    - p=3: 介于两者之间,常用于图像检索任务
    
    Args:
        x: 输入特征图,形状为(B, C, H, W)
        p: 池化参数,控制池化的激进程度
        eps: 防止除零的小常数
        
    Returns:
        池化后的特征,形状为(B, C, 1, 1)
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    """
    广义平均池化层(Generalized Mean Pooling Layer)
    
    这是一个可学习的全局池化层,特别适用于图像检索任务。
    与传统的全局平均池化相比,GeM通过可学习参数p来调整池化的特性,
    能够更好地关注图像中的重要特征。
    """
    def __init__(self, p=3, eps=1e-6):
        """
        初始化GeM层
        
        Args:
            p (float): 初始池化参数,默认为3
            eps (float): 防止数值不稳定的小常数
        """
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)  # 可学习的池化参数
        self.eps = eps
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 原始版本
        # return gem(x, p=self.p, eps=self.eps)

        # 针对有clamp的版本（GPU）
        powered = x.clamp(min=self.eps).pow(self.p)   # 先做 clamp 与 p 次方

        # 针对无clamp的版本（DLA）
        # powered = torch.max(x, torch.tensor(self.eps, device=x.device, dtype=x.dtype)).pow(self.p)

        pooled  = self.adaptive_pool(powered)         # 全局池化：输出 [B,C,1,1]
        return pooled.pow(1.0 / self.p)               # 再开 p 次方根
    
    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class Flatten(torch.nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"

        # DLA版本
        # return x.permute(2,3,0,1)
        # return x.view(x.size(0), -1)
        # 原始版本（GPU）
        return x[:, :, 0, 0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)

class DLAUltraCompatibleL2Norm(nn.Module):

    def __init__(self, channels, eps=1e-12, use_reciprocal=True, newton_iterations=3):
        super().__init__()
        self.eps = eps
        self.use_reciprocal = use_reciprocal
        self.newton_iterations = newton_iterations
        # 预定义1x1卷积层实现通道求和
        self.channel_sum = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_sum.weight.fill_(1.0)  # 权重设为全1

    def newton_sqrt(self, a):

        x = torch.where(a > 1.0, a * 0.5, torch.clamp(a, min=0.1))
        
        # 牛顿迭代 - 使用更高效的形式
        for _ in range(self.newton_iterations):
            x = 0.5 * (x + a / x)
            
        return x
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        x_squared = x * x  # (B, C, H, W)
        norm_squared = self.channel_sum(x_squared) + self.eps  # (B, 1, H, W)

        norm = self.newton_sqrt(norm_squared)  # 等价于sqrt,但用Pow算子
        return x / norm
class NoSqrtManualL2Norm(nn.Module):

    def __init__(self, feature_dim, eps=1e-12, use_reciprocal=True, newton_iterations=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.use_reciprocal = use_reciprocal
        self.newton_iterations = newton_iterations
        # 预定义求和权重
        self.register_buffer('sum_weights', torch.ones(feature_dim, 1))
    def newton_sqrt(self, a):

        x = torch.where(a > 1.0, a * 0.5, torch.clamp(a, min=0.1))
        
        # 牛顿迭代 - 使用更高效的形式
        for _ in range(self.newton_iterations):
            x = 0.5 * (x + a / x)
            
        return x
    
    def forward(self, x):
        # x shape: (B, C)
        assert x.size(1) == self.feature_dim, f"Expected {self.feature_dim} features, got {x.size(1)}"
        
        x_squared = x * x  # (B, C)
        norm_squared = torch.mm(x_squared, self.sum_weights).squeeze(-1)  # (B,)
        norm_squared = norm_squared + self.eps  # (B,)
        
        # if self.use_reciprocal:
        #     # 方法1: x / sqrt(norm_squared) = x * pow(norm_squared, -0.5)
        #     inv_norm = torch.pow(norm_squared.unsqueeze(-1), -0.5)  # (B, 1)
        #     return x * inv_norm  # 广播乘法
        # else:
            # 方法2: 使用pow(0.5)替代sqrt
            # norm = torch.pow(norm_squared.unsqueeze(-1), 0.5)  # (B, 1)
            # return x / norm
        norm = self.newton_sqrt(norm_squared)  # 使用牛顿迭代法代替torch.sqrt
        return x / norm
