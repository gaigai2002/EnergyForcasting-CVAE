import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib


def load_and_preprocess_data(data_path, continuous_features, categorical_features, target_features, train_test_split_ratio):
    """
    加载数据，进行时间特征工程，并对连续特征和目标特征进行归一化。
    """
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # 提取时间特征 (作为类别特征)
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['Month'] = df['Datetime'].dt.month - 1 # Month 1-12 -> 0-11

    # 划分训练集和测试集
    train_size = int(len(df) * train_test_split_ratio)
    train_df, test_df = df.iloc[:train_size].copy(), df.iloc[train_size:].copy() # Use .copy() to avoid SettingWithCopyWarning

    # 对连续特征和目标特征进行归一化
    scaler_cond_continuous = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # 训练集缩放
    train_df[continuous_features] = scaler_cond_continuous.fit_transform(train_df[continuous_features])
    train_df[target_features] = scaler_target.fit_transform(train_df[target_features])

    # 测试集缩放
    test_df[continuous_features] = scaler_cond_continuous.transform(test_df[continuous_features])
    test_df[target_features] = scaler_target.transform(test_df[target_features])

    print("--- 数据维度检查 ---")
    print(f"原始数据总行数: {len(df)}")
    print(f"训练集行数: {len(train_df)}")
    print(f"测试集行数: {len(test_df)}")
    print(f"连续条件特征维度: {len(continuous_features)}")
    print(f"类别条件特征维度: {len(categorical_features)}")
    print(f"目标特征维度: {len(target_features)}")

    return df, train_df, test_df, scaler_cond_continuous, scaler_target

def create_dataloaders(train_df, test_df, continuous_features, categorical_features, target_features, batch_size):
    """
    根据处理后的DataFrame创建PyTorch DataLoader。
    """
    # 训练数据
    X_train_cont = torch.tensor(train_df[continuous_features].values, dtype=torch.float32)
    X_train_cat = torch.tensor(train_df[categorical_features].values, dtype=torch.long)
    y_train = torch.tensor(train_df[target_features].values, dtype=torch.float32)

    # 测试数据
    X_test_cont = torch.tensor(test_df[continuous_features].values, dtype=torch.float32)
    X_test_cat = torch.tensor(test_df[categorical_features].values, dtype=torch.long)
    y_test = torch.tensor(test_df[target_features].values, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_cont, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test_cont, X_test_cat, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test_cont, X_test_cat, y_test

def save_scalers(scaler_cond_continuous, scaler_target, model_dir, scaler_cond_path, scaler_target_path):
    """保存MinMaxScaler实例。"""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler_cond_continuous, scaler_cond_path)
    joblib.dump(scaler_target, scaler_target_path)
    print(f"MinMax scalers saved to {scaler_cond_path} and {scaler_target_path}")

def load_scalers(scaler_cond_path, scaler_target_path):
    """加载MinMaxScaler实例。"""
    scaler_cond_continuous = joblib.load(scaler_cond_path)
    scaler_target = joblib.load(scaler_target_path)
    print(f"MinMax scalers loaded from {scaler_cond_path} and {scaler_target_path}")
    return scaler_cond_continuous, scaler_target
