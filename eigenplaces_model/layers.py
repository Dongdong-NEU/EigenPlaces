
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gem(x, p=torch.ones(1)*3, eps: float = 1e-6):

    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):

        super().__init__()
        self.p = Parameter(torch.ones(1)*p)  
        self.eps = eps
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 原始版本
        # return gem(x, p=self.p, eps=self.eps)

        p_val = float(self.p.detach().clamp_min(1e-6))
        print(f"p_val: {p_val}")

        # 针对有clamp的版本（GPU）, 在trace的时候pow中的参数中不能有动态参数
        powered = x.clamp(min=self.eps).pow(p_val)  

        # 针对无clamp的版本（DLA）
        # powered = torch.max(x, torch.tensor(self.eps, device=x.device, dtype=x.dtype)).pow(self.p)

        pooled  = self.adaptive_pool(powered)        
        return pooled.pow(1.0 / p_val)          
    
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

        norm = self.newton_sqrt(norm_squared)
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
        # norm = self.newton_sqrt(norm_squared)  # 使用牛顿迭代法代替torch.sqrt (B,)
        # return x / norm.unsqueeze(-1)  # (B, C) / (B, 1) -> (B, C)
