import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Component 1: 径向基特征提取层 (RBF Layer)
# ==========================================
class RBFLayer(nn.Module):
    def __init__(self, in_features, num_kernels):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.num_kernels = num_kernels
        self.mu = nn.Parameter(torch.Tensor(num_kernels, in_features))
        nn.init.uniform_(self.mu, a=0.0, b=1.0)
        self.sigma = nn.Parameter(torch.ones(num_kernels) * 0.3)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_expanded = x.unsqueeze(1)
        diff = x_expanded - self.mu
        dist_sq = torch.sum(diff ** 2, dim=2)
        epsilon = 1e-8
        sigma_sq = (self.sigma ** 2) + epsilon
        h = torch.exp(-dist_sq / (2.0 * sigma_sq))
        return h

# ==========================================
# Component 2: 物理边界映射层 (Physical Boundary Layer)
# ==========================================
class PhysicalBoundaryLayer(nn.Module):
    def __init__(self, in_features, beta_h_max, beta_e_max):
        super(PhysicalBoundaryLayer, self).__init__()
        self.beta_H_max = beta_h_max
        self.beta_E_max = beta_e_max
        self.fc_F = nn.Linear(in_features, 1)
        self.fc_beta_H = nn.Linear(in_features, 1)
        self.fc_beta_E = nn.Linear(in_features, 1)

    def forward(self, h):
        F_theta = torch.sigmoid(self.fc_F(h)) * 0.5
        beta_H = self.beta_H_max * torch.sigmoid(self.fc_beta_H(h))
        beta_E = self.beta_E_max * torch.sigmoid(self.fc_beta_E(h))
        return beta_H.squeeze(), beta_E.squeeze(), F_theta.squeeze()

# ==========================================
# Component 3: 升级版数据驱动残差网络 (Residual NN - RBF Version)
# ==========================================
class ResidualSpikeLayer(nn.Module):
    """
    接收 11维 高阶周度特征，先经过与阶段一相同的 RBF 核函数进行非线性激活，再预测周度残差。
    """
    def __init__(self, input_dim=11, num_kernels=32): # 这里的核数量可以根据实验情况调整
        super(ResidualSpikeLayer, self).__init__()
        
        # 🌟 核心修改：第一层使用与物理层完全相同的 RBF 核函数层
        # 全随机初始化，让网络自己去寻找 11 维特征空间中的“暴发适宜中心”
        self.rbf = RBFLayer(in_features=input_dim, num_kernels=num_kernels)
        
        # 接收 RBF 激活后的输出 (维度 = num_kernels) 进行降维和补偿计算
        self.network = nn.Sequential(
            nn.Linear(num_kernels, 64),
            nn.ReLU(),  # 这里的 ReLU 仅做隐层特征过滤，不再直接处理原始环境数据
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 🌟 关键初始化：将最后一层初始化为 0。
        # 确保模型在阶段二初始阶段输出为 0，让 ODE 基底平滑过渡。
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        h = self.rbf(x)            # 1. 经过径向基核函数映射
        return self.network(h)     # 2. 映射为具体的病例暴发补偿量

# ==========================================
# Final Assembly: 云端大模型 (Cloud Teacher Model)
# ==========================================
class CloudTeacherModel(nn.Module):
    # 注意：这里我们明确拆分了物理输入维度 (5) 和 残差输入维度 (6)
    def __init__(self, input_dim_phy=5, input_dim_res=6, num_kernels=16, beta_h_max=1.5, beta_e_max=1.0):
        super(CloudTeacherModel, self).__init__()
        self.rbf_layer = RBFLayer(in_features=input_dim_phy, num_kernels=num_kernels)
        self.boundary_layer = PhysicalBoundaryLayer(in_features=num_kernels, beta_h_max=beta_h_max, beta_e_max=beta_e_max)
        
        # 实例化残差层
        self.residual_layer = ResidualSpikeLayer(input_dim=input_dim_res)

    def forward_physics(self, x_daily):
        """专供 ODE 引擎按天调用的物理分支"""
        h_t = self.rbf_layer(x_daily)
        beta_H, beta_E, F_theta = self.boundary_layer(h_t)
        return beta_H, beta_E, F_theta
        
    def forward_residual(self, x_weekly):
        """阶段二专用的残差分支，按周调用"""
        return self.residual_layer(x_weekly)