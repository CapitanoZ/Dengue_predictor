import torch
import torch.nn as nn
from torchdiffeq import odeint
import threading  
import os
import pandas as pd
import numpy as np

from CloudModel import CloudTeacherModel
from ODE import SEIREnv_ODE_Engine
from data_loader import load_and_preprocess_sj_data

# ==============================================================================
# ⚙️ 全局超参数配置区
# ==============================================================================
PRETRAINED_WEIGHTS_PATH = "physics_base_only.pth"  

PHYSICS_EPOCHS = 400   
RESIDUAL_EPOCHS = 300  

PHYSICS_LR = 0.003    
RESIDUAL_LR = 0.002   

POPULATION_SCALE = 400000.0

ODE_SIGMA = 0.2
ODE_GAMMA = 0.15
ODE_OMEGA = 0.1
ODE_TAU = 0.1
ODE_XI = 0.005

INPUT_DIM_PHY = 6  # 🌟 修改：从 5 改为 6，增加了前一周（7天前）的真实感染人数作为特征
INPUT_DIM_RES = 11 # 🌟 修改：残差网络吃 11 个高阶特征
NUM_KERNELS = 16
BETA_H_MAX = 1.5  
BETA_E_MAX = 1.0

INITIAL_STATE_VALUES = [0.99998, 0.00001, 0.00001, 0.0, 0.0, 0.0]

FEATURES_PATH = 'dengue_features_train.csv'
LABELS_PATH = 'dengue_labels_train.csv'
SAVE_MODEL_PATH = 'res_with_RBF.pth'

# ==============================================================================
# 🔌 加载专门给残差网络用的周度特征 (高阶滑动窗口 + 季节性 + Lag3)
# ==============================================================================
def load_weekly_residual_features(features_path, labels_path, total_weeks):
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    df_features_sj = df_features[df_features['city'] == 'sj'].copy()
    df_labels_sj = df_labels[df_labels['city'] == 'sj'].copy()
    
    df = pd.merge(df_features_sj, df_labels_sj, on=['city', 'year', 'weekofyear'])
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.sort_values('week_start_date').reset_index(drop=True)
    
    base_weather = ['station_avg_temp_c', 'precipitation_amt_mm', 
                    'reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'ndvi_se']
    df[base_weather] = df[base_weather].ffill().bfill()
    
    # 🌟 策略 1：滑动窗口 (过去几周的积累)
    df['temp_roll_4'] = df['station_avg_temp_c'].rolling(window=4, min_periods=1).mean()
    df['humid_roll_4'] = df['reanalysis_specific_humidity_g_per_kg'].rolling(window=4, min_periods=1).mean()
    df['precip_roll_4'] = df['precipitation_amt_mm'].rolling(window=4, min_periods=1).sum()
    df['precip_roll_8'] = df['precipitation_amt_mm'].rolling(window=8, min_periods=1).sum()
    
    # 🌟 策略 2：客观气象整体滞后 3 周 (严格防止滞后1周带来的模型偷懒)
    env_features = base_weather + ['temp_roll_4', 'humid_roll_4', 'precip_roll_4', 'precip_roll_8']
    df[env_features] = df[env_features].shift(3).bfill()
    
    # 🌟 策略 3：日历与季节连续编码
    df['sin_week'] = np.sin(2 * np.pi * df['weekofyear'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['weekofyear'] / 52.0)
    
    all_features = env_features + ['sin_week', 'cos_week']
    
    # 🌟 裁剪并归一化：只使用被允许的 total_weeks 范围
    df = df.iloc[:total_weeks].copy()
    
    X_min = df[all_features].min()
    X_max = df[all_features].max()
    df[all_features] = (df[all_features] - X_min) / (X_max - X_min)
    
    X_weekly = torch.tensor(df[all_features].values, dtype=torch.float32)
    return X_weekly

# ==============================================================================
# ⌨️ 后台监听线程
# ==============================================================================
def keyboard_listener(opt_physics, opt_residual):
    while True:
        try:
            cmd = input().strip().lower()
            if cmd == 'increase':
                for opt in [opt_physics, opt_residual]:
                    for g in opt.param_groups: g['lr'] *= 2.0
                print(f"\n🚀 [动态干预] 全局学习率翻倍! Physics LR: {opt_physics.param_groups[0]['lr']:.6f} | Residual LR: {opt_residual.param_groups[0]['lr']:.6f}\n")
            elif cmd == 'decrease':
                for opt in [opt_physics, opt_residual]:
                    for g in opt.param_groups: g['lr'] *= 0.5
                print(f"\n🐢 [动态干预] 全局学习率减半! Physics LR: {opt_physics.param_groups[0]['lr']:.6f} | Residual LR: {opt_residual.param_groups[0]['lr']:.6f}\n")
            elif cmd == 'inc_res':
                for g in opt_residual.param_groups: g['lr'] *= 2.0
                print(f"\n⚡ [动态干预] 残差层学习率翻倍! Residual LR: {opt_residual.param_groups[0]['lr']:.6f}\n")
            elif cmd == 'dec_res':
                for g in opt_residual.param_groups: g['lr'] *= 0.5
                print(f"\n📉 [动态干预] 残差层学习率减半! Residual LR: {opt_residual.param_groups[0]['lr']:.6f}\n")
            elif cmd == 'inc_phy':
                for g in opt_physics.param_groups: g['lr'] *= 2.0
                print(f"\n🔥 [动态干预] 物理层学习率翻倍! Physics LR: {opt_physics.param_groups[0]['lr']:.6f}\n")
            elif cmd == 'dec_phy':
                for g in opt_physics.param_groups: g['lr'] *= 0.5
                print(f"\n🌊 [动态干预] 物理层学习率减半! Physics LR: {opt_physics.param_groups[0]['lr']:.6f}\n")
        except EOFError:
            break

# ==============================================================================
# 🔄 动态 ODE 包装器 (已引入 Teacher Forcing 特征)
# ==============================================================================
class DynamicODEWrapper(nn.Module):
    def __init__(self, teacher_model, ode_engine, daily_features, daily_true_cases):
        super(DynamicODEWrapper, self).__init__()
        self.teacher_model = teacher_model
        self.ode_engine = ode_engine
        self.daily_features = daily_features
        self.daily_true_cases = daily_true_cases # 🌟 每日真实病例张量
        self.total_days = daily_features.shape[0]

    def forward(self, t, state):
        day_idx = int(t.item())
        day_idx = min(day_idx, self.total_days - 1)
        day_idx = max(day_idx, 0)

        # 获取当日气象环境特征 [5]
        current_env = self.daily_features[day_idx]
        
        # 🌟 核心修改点：将前一天 (-1) 改为前一周 (-7)
        # 获取前一周（7天前）的真实感染人数。如果 day_idx < 7 则用第 0 天的来代替兜底
        prev_week_day_idx = max(0, day_idx - 7)
        prev_week_cases = self.daily_true_cases[prev_week_day_idx].unsqueeze(0) 
        
        # 将环境特征与前一周的病例数拼接成 [1, 6] 的向量
        combined_features = torch.cat([current_env, prev_week_cases], dim=-1).unsqueeze(0)

        # 仅调用物理分支，保证 ODE 干净运行
        beta_H, beta_E, F_theta = self.teacher_model.forward_physics(combined_features)
        self.ode_engine.cloud_model = lambda time_dummy: (beta_H.squeeze(), beta_E.squeeze(), F_theta.squeeze())

        return self.ode_engine(t, state)

def train_dengue_model():
    global PRETRAINED_WEIGHTS_PATH

    # 1. 加载 ODE 需要的日度数据和标签
    daily_features, weekly_labels = load_and_preprocess_sj_data(FEATURES_PATH, LABELS_PATH)

    CURRICULUM_WEEKS = 600
    CURRICULUM_DAYS = CURRICULUM_WEEKS * 7

    daily_features = daily_features[:CURRICULUM_DAYS]
    weekly_labels = weekly_labels[:CURRICULUM_WEEKS]
    
    total_days = daily_features.shape[0]
    total_weeks = weekly_labels.shape[0]

    # 🌟 处理真实的日度数据（为 Teacher Forcing 准备）
    # 由于原始标签 weekly_labels 是周度的，做一个简单的平摊插值
    daily_true_cases = torch.zeros(total_days, dtype=torch.float32)
    for w in range(total_weeks):
        start_day = w * 7
        end_day = min(start_day + 7, total_days)
        weekly_case = weekly_labels[w].item()
        daily_true_cases[start_day:end_day] = weekly_case / 7.0
        
    # 归一化处理以适应 RBF 层
    case_max = daily_true_cases.max()
    if case_max > 0:
        daily_true_cases = daily_true_cases / case_max

    # 2. 加载残差网络专属的高阶周度特征
    weekly_res_features = load_weekly_residual_features(FEATURES_PATH, LABELS_PATH, total_weeks)

    print("Initializing models...")
    teacher_model = CloudTeacherModel(
        input_dim_phy=INPUT_DIM_PHY, 
        input_dim_res=INPUT_DIM_RES, 
        num_kernels=NUM_KERNELS, 
        beta_h_max=BETA_H_MAX, 
        beta_e_max=BETA_E_MAX
    )
    ode_engine = SEIREnv_ODE_Engine(sigma=ODE_SIGMA, gamma=ODE_GAMMA, omega=ODE_OMEGA, tau=ODE_TAU, xi=ODE_XI, cloud_model=None)

    if PRETRAINED_WEIGHTS_PATH is not None and os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"📥 检测到预训练权重参数，正在加载: '{PRETRAINED_WEIGHTS_PATH}' ...")
        teacher_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH), strict=False)
        print("✅ 物理网络基底权重已成功加载！")
    elif PRETRAINED_WEIGHTS_PATH is not None and not os.path.exists(PRETRAINED_WEIGHTS_PATH):
        print(f"⚠️ 警告: 找不到权重文件 '{PRETRAINED_WEIGHTS_PATH}'，将从头开始随机初始化训练！")
        PRETRAINED_WEIGHTS_PATH = None

    optimizer_physics = torch.optim.Adam([
        {'params': teacher_model.rbf_layer.parameters()},
        {'params': teacher_model.boundary_layer.parameters()}
    ], lr=PHYSICS_LR)
    
    optimizer_residual = torch.optim.Adam(teacher_model.residual_layer.parameters(), lr=RESIDUAL_LR)

    listener_thread = threading.Thread(
        target=keyboard_listener, 
        args=(optimizer_physics, optimizer_residual), 
        daemon=True
    )
    listener_thread.start()
    print("💡 提示：高级指令已启用！输入 'inc_res'/'dec_res' 仅控制残差学习率，'inc_phy'/'dec_phy' 仅控制物理学习率。")

    criterion_physics = torch.nn.MSELoss() 
    criterion_residual = torch.nn.MSELoss() 

    t_eval = torch.linspace(0, total_days - 1, total_days)

    # ==============================================================================
    # 🚀 阶段一：纯物理驱动训练
    # ==============================================================================
    if PRETRAINED_WEIGHTS_PATH is None:
        print(f"\n--- 1. 开始阶段一：纯物理驱动训练 (使用 PHYSICS_LR={PHYSICS_LR}) ---")
        for epoch in range(1, PHYSICS_EPOCHS + 1):
            optimizer_physics.zero_grad()
            
            initial_state = torch.tensor(INITIAL_STATE_VALUES, requires_grad=True)
            # 传入 daily_true_cases
            wrapper = DynamicODEWrapper(teacher_model, ode_engine, daily_features, daily_true_cases)

            solution = odeint(wrapper, initial_state, t_eval, method='rk4', options={'step_size': 0.25})
            C_trajectory = solution[:, 5]
            weekly_C = C_trajectory[6::7][:total_weeks]

            C_shifted = torch.cat([torch.tensor([0.0]), weekly_C[:-1]])
            pred_weekly_proportion = weekly_C - C_shifted
            pred_weekly_cases = pred_weekly_proportion * POPULATION_SCALE

            loss = criterion_physics(pred_weekly_cases.squeeze(), weekly_labels.squeeze())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(teacher_model.rbf_layer.parameters()) + list(teacher_model.boundary_layer.parameters()), max_norm=5.0)
            optimizer_physics.step()

            if epoch % 10 == 0 or epoch == 1:
                print(f"Stage 1 (Physics)  | Epoch {epoch:03d}/{PHYSICS_EPOCHS} | Loss: {loss.detach().item():.4f} | LR: {optimizer_physics.param_groups[0]['lr']:.6f}")
    else:
        print(f"\n--- 1. 阶段一：纯物理驱动训练已跳过 (使用预训练权重锁定物理基底) ---")

    # ==============================================================================
    # 🚀 阶段二：残差补偿网络训练
    # ==============================================================================
    print(f"\n--- 2. 开始阶段二：残差补偿网络训练 (使用 RESIDUAL_LR={RESIDUAL_LR}) ---")
    for epoch in range(1, RESIDUAL_EPOCHS + 1):
        optimizer_residual.zero_grad()
        
        # 1. 运行 ODE 获取物理基底预测 (无需计算梯度)
        with torch.no_grad():
            initial_state = torch.tensor(INITIAL_STATE_VALUES)
            # 传入 daily_true_cases
            wrapper = DynamicODEWrapper(teacher_model, ode_engine, daily_features, daily_true_cases)
            solution = odeint(wrapper, initial_state, t_eval, method='rk4', options={'step_size': 0.25})
            C_trajectory = solution[:, 5]
            weekly_C = C_trajectory[6::7][:total_weeks]
            C_shifted = torch.cat([torch.tensor([0.0]), weekly_C[:-1]])
            pred_weekly_proportion = weekly_C - C_shifted
            pred_weekly_cases_base = pred_weekly_proportion * POPULATION_SCALE

        # 2. 🛑 使用 11维 高阶周度特征直接预测周度残差！
        weekly_residuals = teacher_model.forward_residual(weekly_res_features)
        
        # 基底 + 神经网络的残差补偿
        pred_weekly_cases_final = pred_weekly_cases_base + weekly_residuals.squeeze()

        loss = criterion_residual(pred_weekly_cases_final.squeeze(), weekly_labels.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher_model.residual_layer.parameters(), max_norm=10.0)
        optimizer_residual.step()

        if epoch % 2 == 0 or epoch == 1:
            print(f"Stage 2 (Residual) | Epoch {epoch:03d}/{RESIDUAL_EPOCHS} | Total Loss: {loss.detach().item():.4f} | LR: {optimizer_residual.param_groups[0]['lr']:.6f}")

    torch.save(teacher_model.state_dict(), SAVE_MODEL_PATH)
    print(f"\n🎉 训练圆满完成！联合模型权重已成功保存至 '{SAVE_MODEL_PATH}'!")

if __name__ == "__main__":
    train_dengue_model()