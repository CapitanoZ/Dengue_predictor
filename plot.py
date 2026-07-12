import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Import custom modules
from CloudModel import CloudTeacherModel
from ODE import SEIREnv_ODE_Engine
from data_loader import load_and_preprocess_sj_data
from torchdiffeq import odeint

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 🚀 Global Configuration Parameters
# ==========================================
TRAIN_WEEKS = 600              
TOTAL_PLOT_WEEKS = 936         
POPULATION_SCALE = 400000.0    
WEIGHT_PATH = 'res_with_RBF.pth' 

PLOT_ODE_ONLY = True       
PLOT_ODE_RESIDUAL = True   

def load_weekly_residual_features(features_path, labels_path, total_weeks):
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    df_features_sj = df_features[df_features['city'] == 'sj'].copy()
    df_labels_sj = df_labels[df_labels['city'] == 'sj'].copy()
    
    df = pd.merge(df_features_sj, df_labels_sj, on=['city', 'year', 'weekofyear'])
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.sort_values('week_start_date').reset_index(drop=True)
    df = df.iloc[:total_weeks].copy()
    
    base_weather = [
        'station_avg_temp_c', 'precipitation_amt_mm', 
        'reanalysis_specific_humidity_g_per_kg', 'reanalysis_dew_point_temp_k', 'ndvi_se'
    ]
    df[base_weather] = df[base_weather].ffill().bfill()
    
    df['temp_roll_4'] = df['station_avg_temp_c'].rolling(window=4, min_periods=1).mean()
    df['humid_roll_4'] = df['reanalysis_specific_humidity_g_per_kg'].rolling(window=4, min_periods=1).mean()
    df['precip_roll_4'] = df['precipitation_amt_mm'].rolling(window=4, min_periods=1).sum()
    df['precip_roll_8'] = df['precipitation_amt_mm'].rolling(window=8, min_periods=1).sum()
    
    env_features = base_weather + ['temp_roll_4', 'humid_roll_4', 'precip_roll_4', 'precip_roll_8']
    df[env_features] = df[env_features].shift(3).bfill()
    
    df['sin_week'] = np.sin(2 * np.pi * df['weekofyear'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['weekofyear'] / 52.0)
    
    all_features = env_features + ['sin_week', 'cos_week'] 
    
    train_df = df.iloc[:TRAIN_WEEKS]
    X_min = train_df[all_features].min()
    X_max = train_df[all_features].max()
    
    df[all_features] = (df[all_features] - X_min) / (X_max - X_min)
    
    X_weekly = torch.tensor(df[all_features].values, dtype=torch.float32)
    true_cases = df['total_cases'].values
    
    return X_weekly, true_cases

class DynamicODEWrapper(nn.Module):
    def __init__(self, teacher_model, ode_engine, daily_features, daily_true_cases):
        super(DynamicODEWrapper, self).__init__()
        self.teacher_model = teacher_model
        self.ode_engine = ode_engine
        self.daily_features = daily_features
        self.daily_true_cases = daily_true_cases
        self.total_days = daily_features.shape[0]

    def forward(self, t, state):
        day_idx = int(t.item())
        day_idx = min(day_idx, self.total_days - 1)
        day_idx = max(day_idx, 0)

        current_env = self.daily_features[day_idx]
        prev_week_day_idx = max(0, day_idx - 7)
        prev_week_cases = self.daily_true_cases[prev_week_day_idx].unsqueeze(0) 
        
        combined_features = torch.cat([current_env, prev_week_cases], dim=-1).unsqueeze(0)

        beta_H, beta_E, F_theta = self.teacher_model.forward_physics(combined_features)
        self.ode_engine.cloud_model = lambda time_dummy: (beta_H.squeeze(), beta_E.squeeze(), F_theta.squeeze())
        return self.ode_engine(t, state)

def plot_model_predictions():
    features_path = 'dengue_features_train.csv'
    labels_path = 'dengue_labels_train.csv'
    
    print(f"Loading data up to {TOTAL_PLOT_WEEKS} weeks...")
    
    daily_features_5d, _ = load_and_preprocess_sj_data(features_path, labels_path)
    daily_features_5d = daily_features_5d[:TOTAL_PLOT_WEEKS * 7]
    total_days = daily_features_5d.shape[0]
    
    X_weekly_res, true_cases = load_weekly_residual_features(features_path, labels_path, TOTAL_PLOT_WEEKS)
    
    daily_true_cases = torch.zeros(total_days, dtype=torch.float32)
    for w in range(TOTAL_PLOT_WEEKS):
        start_day = w * 7
        end_day = min(start_day + 7, total_days)
        weekly_case = true_cases[w]
        daily_true_cases[start_day:end_day] = weekly_case / 7.0
        
    train_days = TRAIN_WEEKS * 7
    case_max = daily_true_cases[:train_days].max()
    if case_max > 0:
        daily_true_cases = daily_true_cases / case_max

    print(f"Loading model weights from '{WEIGHT_PATH}'...")
    
    teacher_model = CloudTeacherModel(
        input_dim_phy=6, 
        input_dim_res=11, 
        num_kernels=16, 
        beta_h_max=1.5, 
        beta_e_max=1.0
    )
    
    teacher_model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
    print("✅ Weights loaded successfully!")
        
    teacher_model.eval() 
    ode_engine = SEIREnv_ODE_Engine(sigma=0.2, gamma=0.15, omega=0.1, tau=0.1, xi=0.005, cloud_model=None)
    
    print("Forward simulating the ODE dynamic progression...")
    
    wrapper = DynamicODEWrapper(teacher_model, ode_engine, daily_features_5d, daily_true_cases)
    
    current_state = torch.tensor([0.99998, 0.00001, 0.00001, 0.0, 0.0, 0.0])
    t_eval = torch.linspace(0, total_days - 1, total_days)
    
    with torch.no_grad():
        solution = odeint(wrapper, current_state, t_eval, method='rk4', options={'step_size': 0.25})
            
    C_trajectory = solution[:, 5]
    weekly_C = C_trajectory[6::7][:TOTAL_PLOT_WEEKS]
    C_shifted = torch.cat([torch.tensor([0.0]), weekly_C[:-1]])
    pred_weekly_proportion = weekly_C - C_shifted 
    pred_weekly_cases_base = pred_weekly_proportion * POPULATION_SCALE

    print("Executing One-shot Residual Compensation...")
    with torch.no_grad():
        weekly_residuals = teacher_model.forward_residual(X_weekly_res).squeeze()
        pred_weekly_cases_final = pred_weekly_cases_base + weekly_residuals
        pred_weekly_cases_final = torch.clamp(pred_weekly_cases_final, min=0.0)

    # =========================================================
    # 📊 新增功能：计算训练集与预测集的 MSE
    # =========================================================
    pred_final_np = pred_weekly_cases_final.detach().cpu().numpy()
    
    # 严格按照 TRAIN_WEEKS 切片计算 MSE
    train_mse = np.mean((true_cases[:TRAIN_WEEKS] - pred_final_np[:TRAIN_WEEKS]) ** 2)
    test_mse = np.mean((true_cases[TRAIN_WEEKS:] - pred_final_np[TRAIN_WEEKS:]) ** 2)
    print(f"📊 Evaluation -> Train MSE: {train_mse:.2f} | Test MSE: {test_mse:.2f}")

    true_cumulative_cases = np.cumsum(true_cases)
    pred_cumulative_cases_base = torch.cumsum(pred_weekly_cases_base, dim=0)
    pred_cumulative_cases_final = torch.cumsum(pred_weekly_cases_final, dim=0)
    
    print("Preparing to plot comparison graphs...")
    # 🌟 将画布高度从 12 调整为 16，以容纳 3 个子图
    plt.figure(figsize=(16, 16), dpi=100) 
    weeks_axis = np.arange(TOTAL_PLOT_WEEKS)
    
    # ==========================================
    # ====== Subplot 1: Weekly New Cases ======
    # ==========================================
    plt.subplot(3, 1, 1)
    plt.plot(weeks_axis, true_cases, label='True Weekly Cases (Ground Truth)', color='#333333', linewidth=2)
    
    if PLOT_ODE_ONLY:
        plt.plot(weeks_axis, pred_weekly_cases_base.detach().cpu().numpy(), label='Predicted (Physics ODE Only)', color='#cc6600', linestyle='--', linewidth=1.5, alpha=0.8)
    if PLOT_ODE_RESIDUAL:
        plt.plot(weeks_axis, pred_final_np, label='Predicted (Physics + 11D RBF Residual)', color='#cc0000', linestyle='-', linewidth=2, alpha=0.9)
    
    plt.axvline(x=TRAIN_WEEKS, color='green', linestyle='-.', linewidth=2, label='Test Phase Start')
    
    plt.text(TRAIN_WEEKS + 5, np.max(true_cases)*0.9, 'Evaluation / Live Inference', color='green', fontsize=12, fontweight='bold')
    plt.text(TRAIN_WEEKS - 150, np.max(true_cases)*0.9, 'Training Phase', color='green', fontsize=12, fontweight='bold')

    # 📌 将计算好的 MSE 以文本框形式添加到图中
    mse_text = f"Final Model Performance\nTrain MSE: {train_mse:.2f}\nTest MSE: {test_mse:.2f}"
    plt.gca().text(0.02, 0.85, mse_text, transform=plt.gca().transAxes, 
                   fontsize=12, fontweight='bold', color='#333333',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc', boxstyle='round,pad=0.5'))

    plt.title(f'A: Weekly New Cases Comparison - {TRAIN_WEEKS} Train + {TOTAL_PLOT_WEEKS - TRAIN_WEEKS} Test', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Time (Weeks)', fontsize=12)
    plt.ylabel('New Cases', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(color='#e0e0e0', linestyle=':', linewidth=0.6)
    
    # ==========================================
    # ====== Subplot 2: Cumulative Total Cases =
    # ==========================================
    plt.subplot(3, 1, 2)
    plt.plot(weeks_axis, true_cumulative_cases, label='True Cumulative Cases', color='#333333', linewidth=2)
    
    if PLOT_ODE_ONLY:
        plt.plot(weeks_axis, pred_cumulative_cases_base.detach().cpu().numpy(), label='Predicted (Physics ODE Only)', color='#66b3ff', linestyle='--', linewidth=1.5, alpha=0.8)
    if PLOT_ODE_RESIDUAL:
        plt.plot(weeks_axis, pred_cumulative_cases_final.detach().cpu().numpy(), label='Predicted (Physics + 11D RBF Residual)', color='#0044cc', linestyle='-', linewidth=2, alpha=0.9)
    
    plt.axvline(x=TRAIN_WEEKS, color='green', linestyle='-.', linewidth=2, label='Test Phase Start')

    plt.title(f'B: Cumulative Total Cases Comparison - {TRAIN_WEEKS} Train + {TOTAL_PLOT_WEEKS - TRAIN_WEEKS} Test', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Time (Weeks)', fontsize=12)
    plt.ylabel('Cumulative Cases', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(color='#e0e0e0', linestyle=':', linewidth=0.6)

    # ==========================================
    # ====== Subplot 3: Residual Compensation ==
    # ==========================================
    plt.subplot(3, 1, 3)
    # 画出 11 维高阶残差网络给出的“病例加减量”
    plt.plot(weeks_axis, weekly_residuals.detach().cpu().numpy(), label='Neural Network Residual Output', color='#800080', linewidth=1.5)
    # 画出 y=0 的基准参考线
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(x=TRAIN_WEEKS, color='green', linestyle='-.', linewidth=2, label='Test Phase Start')
    
    # 使用条形填充（fill_between）让正负残差看起来更明显
    plt.fill_between(weeks_axis, 0, weekly_residuals.detach().cpu().numpy(), where=(weekly_residuals.detach().cpu().numpy() > 0), color='red', alpha=0.3, label='Positive Spike Compensation')
    plt.fill_between(weeks_axis, 0, weekly_residuals.detach().cpu().numpy(), where=(weekly_residuals.detach().cpu().numpy() < 0), color='blue', alpha=0.3, label='Negative Suppression')

    plt.title('C: Neural Network Residual Compensation (Physics-Informed Correction)', fontsize=14, fontweight='bold', pad=10)
    plt.xlabel('Time (Weeks)', fontsize=12)
    plt.ylabel('Cases Compensation (+/-)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(color='#e0e0e0', linestyle=':', linewidth=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_model_predictions()