# data_loader.py
import pandas as pd
import torch

def load_and_preprocess_sj_data(features_path, labels_path):
    """
    加载并预处理 DengAI 圣胡安 (sj) 的全量数据
    """
    df_features = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    df_features_sj = df_features[df_features['city'] == 'sj'].copy()
    df_labels_sj = df_labels[df_labels['city'] == 'sj'].copy()
    
    df = pd.merge(df_features_sj, df_labels_sj, on=['city', 'year', 'weekofyear'])
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.set_index('week_start_date')
    
    # 选取5个环境变量
    feature_cols = [
        'station_avg_temp_c', 
        'precipitation_amt_mm', 
        'reanalysis_specific_humidity_g_per_kg',
        'reanalysis_dew_point_temp_k',
        'ndvi_se'
    ]
    
    df[feature_cols] = df[feature_cols].ffill().bfill()
    
    # 归一化
    df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())
    
    # 将周特征插值为日特征
    df_daily = df[feature_cols].resample('D').interpolate(method='linear')
    
    daily_features_tensor = torch.tensor(df_daily.values, dtype=torch.float32)
    weekly_labels_tensor = torch.tensor(df['total_cases'].values, dtype=torch.float32)
    
    print(f"✅ 全量数据加载完成: {daily_features_tensor.shape[0]} 天特征, {weekly_labels_tensor.shape[0]} 周标签")
    return daily_features_tensor, weekly_labels_tensor