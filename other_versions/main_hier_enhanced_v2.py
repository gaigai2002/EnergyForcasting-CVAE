"""
HierCVAE+UniLSTM+Attention 优化版训练脚本 (v2)

核心改进：
1. Lag 精简：只保留 [1, 2, 3, 6, 12]，去掉 24
2. 滚动统计：只保留 rolling mean，去掉 rolling std
3. 差分特征：只保留 diff1（一阶差分），去掉 diff2
4. 残差形式：pred = decoder_out + w1 * (lag1 - rolling_mean)
5. Attention：维度随 LSTM 自然降低，不加 projection
6. LSTM：改为单向 LSTM（不是双向）
7. CVAE：空间释放（global_latent_dim=32，latent 不被 deterministic 吃光）

目标: 提高 CVAE 参与度，让潜在空间真正发挥作用
"""

import torch
import torch.optim as optim
import torch.nn as nn
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import hier_config as config
from src.enhanced_data_utils import load_and_preprocess_enhanced
from src.enhanced_model import EnhancedLoss


# ============= 精简版数据集 =============
class EnhancedTimeSeriesDataset(Dataset):
    """
    精简版时序数据集
    
    核心思路：
    1. 时间窗口：使用前N步预测下一步
    2. 滞后特征：只保留 [1, 2, 3, 6, 12]，去掉 24
    3. 差分特征：只保留一阶差分（diff1），去掉二阶差分
    4. 滚动统计：只保留滚动均值（rolling_mean），去掉滚动标准差
    
    优势：减少特征冗余，让模型更专注于关键信息
    """
    def __init__(self, df, continuous_features, categorical_features,
                 target_col, lookback_steps, lag_features=[1, 2, 3, 6, 12]):
        self.df = df.reset_index(drop=True)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.lookback_steps = lookback_steps
        self.lag_features = lag_features
        self.max_lag = max(lag_features) if lag_features else 0

        # 预计算数据
        self.cont_data = self.df[continuous_features].values.astype(np.float32)
        self.cat_data = self.df[categorical_features].values.astype(np.float32)
        self.target_data = self.df[target_col].values.astype(np.float32)

        # 计算滞后特征 (在归一化后的数据上)
        self.lag_data = np.zeros((len(df), len(lag_features)), dtype=np.float32)
        for i, lag in enumerate(lag_features):
            self.lag_data[lag:, i] = self.target_data[:-lag]

        # 只计算一阶差分 (diff1 = y_t-1 - y_t-2)
        self.diff1 = np.zeros((len(df), 1), dtype=np.float32)
        self.diff1[2:, 0] = self.target_data[1:-1] - self.target_data[:-2]

        # 只计算滚动均值 (使用lookback窗口)
        self.rolling_mean = np.zeros((len(df), 1), dtype=np.float32)
        for i in range(lookback_steps, len(df)):
            window = self.target_data[i-lookback_steps:i]
            self.rolling_mean[i, 0] = window.mean()

        # 有效索引 (需要足够的历史数据)
        min_history = max(lookback_steps, self.max_lag)
        self.valid_indices = list(range(min_history, len(self.df)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        window_start = actual_idx - self.lookback_steps
        window_end = actual_idx

        # 窗口特征
        window_cont = self.cont_data[window_start:window_end]
        window_cat = self.cat_data[window_start:window_end]
        window_target = self.target_data[window_start:window_end].reshape(-1, 1)

        # 归一化类别特征
        window_cat_norm = window_cat.copy()
        window_cat_norm[:, 0] = window_cat_norm[:, 0] / 23.0  # Hour
        window_cat_norm[:, 1] = window_cat_norm[:, 1] / 6.0   # DayOfWeek

        # 拼接窗口特征
        window_features = np.concatenate([window_cont, window_cat_norm, window_target], axis=1)
        window_features = window_features.flatten().astype(np.float32)

        # 当前时刻条件特征
        c_cont = self.cont_data[actual_idx]
        c_cat = self.df[self.categorical_features].iloc[actual_idx].values.astype(np.int64)

        # 滞后特征
        lag_feats = self.lag_data[actual_idx]

        # 一阶差分
        diff1 = self.diff1[actual_idx]

        # 滚动均值（只保留均值）
        rolling_mean = self.rolling_mean[actual_idx]

        # 目标值
        y = np.array([self.target_data[actual_idx]], dtype=np.float32)

        return (
            torch.tensor(window_features),
            torch.tensor(c_cont),
            torch.tensor(c_cat),
            torch.tensor(lag_feats),
            torch.tensor(diff1),
            torch.tensor(rolling_mean),
            torch.tensor(y)
        )


# ============= 精简版时间注意力机制 =============
class TemporalAttention(nn.Module):
    """
    精简版时间注意力机制
    
    核心改进：
    1. 去掉 output_proj，让维度随 LSTM 自然降低
    2. 保持 MultiheadAttention 和 LayerNorm
    3. 直接返回注意力输出，不做额外投影
    """
    def __init__(self, hidden_dim, num_heads=2, dropout=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        out, w = self.attn(q, x, x)
        out = self.norm(out + x[:, -1:, :])
        return out.squeeze(1), w


# ============= 优化版模型 =============
class EnhancedHierCVAE(nn.Module):
    """
    优化版 HierCVAE 模型
    
    架构改进：
    1. 单向 LSTM：改为单向，减少参数和计算量
    2. 精简特征：lag + diff1 + rolling_mean（去掉 diff2 和 rolling_std）
    3. 精简残差：pred = decoder_out + w1 * (lag1 - rolling_mean)
    4. 潜在空间释放：global_latent_dim=32，让 CVAE 真正参与建模
    5. Attention 精简：去掉 projection，维度自然降低
    
    设计目标：让 CVAE 的潜在空间真正发挥作用，而不是被确定性模块完全覆盖
    """
    def __init__(self, cont_cond_dim, num_lag_features=5, 
                 global_latent_dim=32,
                 cat_dims=None, embedding_dims=None, lookback_steps=24,
                 hidden_dim=128, lstm_layers=2, dropout=0.2, num_attention_heads=2):
        super().__init__()

        self.cont_cond_dim = cont_cond_dim
        self.num_lag_features = num_lag_features
        self.global_latent_dim = global_latent_dim
        self.lookback_steps = lookback_steps
        self.hidden_dim = hidden_dim

        cat_dims = cat_dims or {'Hour': 24, 'DayOfWeek': 7}
        embedding_dims = embedding_dims or {'Hour': 16, 'DayOfWeek': 8}

        # 类别嵌入
        self.hour_embed = nn.Embedding(cat_dims["Hour"], embedding_dims["Hour"])
        self.day_embed = nn.Embedding(cat_dims["DayOfWeek"], embedding_dims["DayOfWeek"])
        total_emb_dim = embedding_dims["Hour"] + embedding_dims["DayOfWeek"]
        self.cond_dim = cont_cond_dim + total_emb_dim

        # 滞后特征+diff1+rolling_mean编码器
        # 输入：lag_features (5) + diff1 (1) + rolling_mean (1) = 7
        lag_input_dim = num_lag_features + 1 + 1
        self.lag_encoder = nn.Sequential(
            nn.Linear(lag_input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU()
        )
        lag_feat_dim = hidden_dim // 2  # 64

        # 单向LSTM
        temporal_input_dim = cont_cond_dim + 2 + 1
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False  # 改为单向
        )
        lstm_output_dim = hidden_dim  # 单向，输出维度 = hidden_dim

        # 时间注意力（维度自然降低，不加 projection）
        self.temporal_attention = TemporalAttention(
            hidden_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # 特征融合: LSTM输出 + 滞后特征
        temporal_feat_dim = lstm_output_dim + lag_feat_dim  # 128 + 64 = 192

        # 全局编码器
        global_enc_input = temporal_feat_dim + self.cond_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_enc_input, hidden_dim),
            nn.GELU()
        )
        self.global_mu = nn.Linear(hidden_dim, global_latent_dim)
        self.global_logvar = nn.Linear(hidden_dim, global_latent_dim)

        # 全局先验
        self.global_prior = nn.Sequential(
            nn.Linear(self.cond_dim + lag_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, global_latent_dim * 2)
        )

        # 解码器
        decoder_input = global_latent_dim + temporal_feat_dim + self.cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 精简残差连接：只用 lag1 和 rolling_mean
        self.res_weight = nn.Parameter(torch.tensor(0.5))

        self.temporal_feat_dim = temporal_feat_dim

    def process_conditions(self, c_cont, c_cat):
        c_cat = c_cat.long()
        h = self.hour_embed(c_cat[:, 0])
        d = self.day_embed(c_cat[:, 1])
        return torch.cat([c_cont, h, d], dim=1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, targets=None):
        batch_size = x_window.shape[0]
        c = self.process_conditions(c_cont, c_cat)

        # 编码滞后特征 + diff1 + rolling_mean
        lag_input = torch.cat([lag_feats, diff1, rolling_mean], dim=1)
        lag_encoded = self.lag_encoder(lag_input)

        # LSTM处理窗口（单向）
        features_per_step = self.cont_cond_dim + 2 + 1
        x_window = x_window.view(batch_size, self.lookback_steps, features_per_step)
        lstm_out, _ = self.lstm(x_window)

        # 时间注意力（维度自然降低）
        attn_out, attn_weights = self.temporal_attention(lstm_out)

        # 融合特征
        temporal_feat = torch.cat([attn_out, lag_encoded], dim=1)

        # 全局编码
        global_enc_input = torch.cat([temporal_feat, c], dim=1)
        global_h = self.global_encoder(global_enc_input)
        global_mu_q = self.global_mu(global_h)
        global_logvar_q = self.global_logvar(global_h)

        # 全局先验
        prior_input = torch.cat([c, lag_encoded], dim=1)
        global_prior_out = self.global_prior(prior_input)
        global_mu_p, global_logvar_p = torch.chunk(global_prior_out, 2, dim=1)

        z_g = self.reparameterize(global_mu_q, global_logvar_q)

        # 解码
        decoder_input = torch.cat([z_g, temporal_feat, c], dim=1)
        decoder_out = self.decoder(decoder_input)

        # 精简残差连接：pred = decoder_out + w1 * (lag1 - rolling_mean)
        lag1 = lag_feats[:, 0:1]
        pred = decoder_out + torch.sigmoid(self.res_weight) * (lag1 - rolling_mean)

        return {
            'pred': pred,
            'global_mu_q': global_mu_q,
            'global_logvar_q': global_logvar_q,
            'global_mu_p': global_mu_p,
            'global_logvar_p': global_logvar_p,
            'attention_weights': attn_weights
        }


# ============= 数据加载 =============
def create_dataloaders_enhanced(train_df, val_df, test_df, continuous_features,
                                 categorical_features, target_col, batch_size,
                                 lookback_steps, lag_features):
    train_ds = EnhancedTimeSeriesDataset(train_df, continuous_features, categorical_features,
                                          target_col, lookback_steps, lag_features)
    val_ds = EnhancedTimeSeriesDataset(val_df, continuous_features, categorical_features,
                                        target_col, lookback_steps, lag_features)
    test_ds = EnhancedTimeSeriesDataset(test_df, continuous_features, categorical_features,
                                         target_col, lookback_steps, lag_features)

    # 创建DataLoader：训练集shuffle，验证和测试集不shuffle
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader


# ============= 训练和预测 =============
def train_and_predict_single_apartment(apt_name, train_df, val_df, test_df, device, cfg):
    """
    训练单个公寓的模型并预测
    思路：
    1. 创建数据加载器（包含滞后特征、差分特征等）
    2. 初始化模型和优化器
    3. 训练循环（KL warmup、早停、学习率调度）
    4. 加载最佳模型进行预测
    5. 计算评估指标
    """
    print(f"\n{'='*60}")
    print(f"Processing: {apt_name}")
    print(f"{'='*60}")

    # 配置
    lookback_steps = getattr(cfg, 'LOOKBACK_STEPS_ENHANCED', 24)
    lag_features = getattr(cfg, 'LAG_FEATURES', [1, 2, 3, 6, 12])  # 精简后的 lag features

    train_loader, val_loader, test_loader = create_dataloaders_enhanced(
        train_df, val_df, test_df,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name, cfg.BATCH_SIZE, lookback_steps, lag_features
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Lookback: {lookback_steps}, Lag features: {lag_features}")

    # 模型
    model = EnhancedHierCVAE(
        cont_cond_dim=len(cfg.CONTINUOUS_CONDITIONAL_FEATURES),
        num_lag_features=len(lag_features),
        global_latent_dim=getattr(cfg, 'GLOBAL_LATENT_DIM_V2', 32),  # 使用新的配置
        cat_dims=cfg.CAT_DIMS,
        embedding_dims=cfg.EMBEDDING_DIMS,
        lookback_steps=lookback_steps,
        hidden_dim=cfg.HIDDEN_DIM,
        lstm_layers=2,
        dropout=getattr(cfg, 'DROPOUT_V2', 0.2),  # 使用新的配置
        num_attention_heads=getattr(cfg, 'NUM_ATTENTION_HEADS_V2', 2)  # 使用新的配置
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    loss_fn = EnhancedLoss(beta_global=cfg.BETA_GLOBAL)

    # 训练
    print(f"  Training...")
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = getattr(cfg, 'EARLY_STOP_PATIENCE', 30)

    epochs = getattr(cfg, 'EPOCHS_ENHANCED', cfg.EPOCHS)

    for epoch in range(epochs):
        model.train()
        warmup_epochs = getattr(cfg, 'KL_WARMUP_EPOCHS', 30)
        beta_factor = min(1.0, (epoch + 1) / warmup_epochs)

        for batch in train_loader:
            x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, y = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, targets=y)
            losses = loss_fn(outputs, y, beta_factor=beta_factor)
            losses['total'].backward()

            if getattr(cfg, 'GRADIENT_CLIP', None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, y = [b.to(device) for b in batch]
                outputs = model(x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, targets=y)
                val_loss += loss_fn(outputs, y, beta_factor=1.0)['total'].item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.to(device)

    # 预测
    print(f"  Predicting...")
    model.eval()
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for batch in test_loader:
            x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, y = [b.to(device) for b in batch]
            outputs = model(x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, targets=None)
            all_preds.append(outputs['pred'].cpu())
            all_actuals.append(y.cpu())

    predictions = torch.cat(all_preds, dim=0).numpy()
    actuals = torch.cat(all_actuals, dim=0).numpy()

    # 指标（在归一化空间中计算；在 main 中会使用反归一化后的值重新计算一次）
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    try:
        r2 = r2_score(actuals, predictions)
    except:
        r2 = np.nan

    metrics = {'Apartment': apt_name, 'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mae, 'R2': r2}
    print(f"  Done - MSE: {mse:.4f}, RMSE: {np.sqrt(mse):.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 这里只返回归一化空间中的预测和真实值；反归一化在 main 中统一处理
    return metrics, predictions, actuals, model


# ============= 独立预测函数 =============
def predict_new_data(apt_name, new_df, model_dir, device, cfg, 
                     historical_buffer=None, lookback_steps=24, lag_features=[1, 2, 3, 6, 12]):
    """
    使用训练好的模型预测新数据
    
    参数:
        apt_name: 公寓名称，如 'Apt_1'
        new_df: 新数据 DataFrame，必须包含所有需要的特征列
        model_dir: 模型保存目录
        device: 设备
        cfg: 配置对象
        historical_buffer: 历史数据缓冲区（用于生成滞后特征），如果为None则从new_df开头提取
        lookback_steps: 回看步数
        lag_features: 滞后特征列表（精简后）
    
    返回:
        predictions: 预测结果（反归一化后的值）
    """
    import os
    import joblib
    
    # 加载模型
    model_path = os.path.join(model_dir, f"{apt_name}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 加载 scaler
    scaler_cond_path = os.path.join(model_dir, "scaler_cond.pkl")
    scaler_target_path = os.path.join(model_dir, "scaler_target.pkl")
    
    if not os.path.exists(scaler_cond_path) or not os.path.exists(scaler_target_path):
        raise FileNotFoundError(f"Scalers not found in {model_dir}")
    
    scaler_cond = joblib.load(scaler_cond_path)
    scaler_target = joblib.load(scaler_target_path)
    
    # 准备数据
    df = new_df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    
    # 填充缺失值
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].fillna(
        df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].mean()
    )
    if apt_name in df.columns:
        df[apt_name] = df[apt_name].fillna(0)
    
    # 归一化
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = scaler_cond.transform(
        df[cfg.CONTINUOUS_CONDITIONAL_FEATURES]
    )
    
    # 如果有历史缓冲区，合并用于生成滞后特征
    if historical_buffer is not None:
        # 准备历史缓冲区数据
        hist_df = historical_buffer.copy()
        hist_df['Datetime'] = pd.to_datetime(hist_df['Datetime'])
        hist_df['Hour'] = hist_df['Datetime'].dt.hour
        hist_df['DayOfWeek'] = hist_df['Datetime'].dt.dayofweek
        
        # 填充缺失值
        hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].fillna(
            hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].mean()
        )
        
        # 归一化历史缓冲区的连续特征
        hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = scaler_cond.transform(
            hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES]
        )
        
        # 归一化历史缓冲区中的目标列（如果存在）
        if apt_name in hist_df.columns:
            apt_idx = cfg.TARGET_FEATURES.index(apt_name) if apt_name in cfg.TARGET_FEATURES else 0
            temp_scaler = MinMaxScaler()
            temp_scaler.min_ = scaler_target.min_[apt_idx:apt_idx+1]
            temp_scaler.scale_ = scaler_target.scale_[apt_idx:apt_idx+1]
            temp_scaler.data_min_ = scaler_target.data_min_[apt_idx:apt_idx+1]
            temp_scaler.data_max_ = scaler_target.data_max_[apt_idx:apt_idx+1]
            temp_scaler.data_range_ = scaler_target.data_range_[apt_idx:apt_idx+1]
            temp_scaler.n_features_in_ = 1
            
            # 归一化历史缓冲区中的目标值
            hist_target = hist_df[[apt_name]].values
            hist_target_scaled = temp_scaler.transform(hist_target).flatten()
            hist_df[apt_name] = hist_target_scaled
        
        # 合并历史数据和新区数据
        combined_df = pd.concat([hist_df, df], ignore_index=True)
    else:
        combined_df = df.copy()
        # 如果没有历史数据，使用0填充目标列
        if apt_name not in combined_df.columns:
            combined_df[apt_name] = 0.0
        else:
            # 归一化目标列
            apt_idx = cfg.TARGET_FEATURES.index(apt_name) if apt_name in cfg.TARGET_FEATURES else 0
            temp_scaler = MinMaxScaler()
            temp_scaler.min_ = scaler_target.min_[apt_idx:apt_idx+1]
            temp_scaler.scale_ = scaler_target.scale_[apt_idx:apt_idx+1]
            temp_scaler.data_min_ = scaler_target.data_min_[apt_idx:apt_idx+1]
            temp_scaler.data_max_ = scaler_target.data_max_[apt_idx:apt_idx+1]
            temp_scaler.data_range_ = scaler_target.data_range_[apt_idx:apt_idx+1]
            temp_scaler.n_features_in_ = 1
            combined_df[apt_name] = temp_scaler.transform(combined_df[[apt_name]]).flatten()
    
    # 创建数据集
    dataset = EnhancedTimeSeriesDataset(
        combined_df,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name,
        lookback_steps,
        lag_features
    )
    
    # 只预测新数据部分
    if historical_buffer is not None:
        hist_len = len(historical_buffer)
        # 只取新数据部分的索引
        new_indices = [i for i in range(len(dataset)) if dataset.valid_indices[i] >= hist_len]
        if len(new_indices) == 0:
            print(f"Warning: No valid indices for new data prediction")
            return np.array([])
    else:
        new_indices = list(range(len(dataset)))
    
    # 创建 DataLoader（只包含新数据）
    new_dataset = torch.utils.data.Subset(dataset, new_indices)
    dataloader = DataLoader(new_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True)
    
    # 加载模型
    model = EnhancedHierCVAE(
        cont_cond_dim=len(cfg.CONTINUOUS_CONDITIONAL_FEATURES),
        num_lag_features=len(lag_features),
        global_latent_dim=getattr(cfg, 'GLOBAL_LATENT_DIM_V2', 32),
        cat_dims=cfg.CAT_DIMS,
        embedding_dims=cfg.EMBEDDING_DIMS,
        lookback_steps=lookback_steps,
        hidden_dim=cfg.HIDDEN_DIM,
        lstm_layers=2,
        dropout=getattr(cfg, 'DROPOUT_V2', 0.2),
        num_attention_heads=getattr(cfg, 'NUM_ATTENTION_HEADS_V2', 2)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预测
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, _ = [b.to(device) for b in batch]
            outputs = model(x_window, c_cont, c_cat, lag_feats, diff1, rolling_mean, targets=None)
            all_preds.append(outputs['pred'].cpu())
    
    predictions_scaled = torch.cat(all_preds, dim=0).numpy()
    
    # 反归一化
    apt_idx = cfg.TARGET_FEATURES.index(apt_name) if apt_name in cfg.TARGET_FEATURES else 0
    temp_scaler = MinMaxScaler()
    temp_scaler.min_ = scaler_target.min_[apt_idx:apt_idx+1]
    temp_scaler.scale_ = scaler_target.scale_[apt_idx:apt_idx+1]
    temp_scaler.data_min_ = scaler_target.data_min_[apt_idx:apt_idx+1]
    temp_scaler.data_max_ = scaler_target.data_max_[apt_idx:apt_idx+1]
    temp_scaler.data_range_ = scaler_target.data_range_[apt_idx:apt_idx+1]
    temp_scaler.n_features_in_ = 1
    
    predictions = temp_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    return predictions


def main():
    """
    主函数：训练所有公寓的模型
    
    流程：
    1. 加载和预处理数据
    2. 循环训练每个公寓（单独模型）
    3. 保存模型和评估结果
    4. 输出统计信息
    """
    print("=" * 60)
    print("Enhanced HierCVAE+UniLSTM+Attention v2 (Optimized)")
    print("Key improvements:")
    print("  - Lag features [1,2,3,6,12] (removed 24)")
    print("  - Only diff1 (removed diff2)")
    print("  - Only rolling_mean (removed rolling_std)")
    print("  - Simplified residual: w1*(lag1 - rolling_mean)")
    print("  - Unidirectional LSTM")
    print("  - Attention without projection")
    print("  - Reduced latent dim (32) for better CVAE participation")
    print("=" * 60)

    # 优化配置
    config.LOOKBACK_STEPS_ENHANCED = 24
    config.LAG_FEATURES = [1, 2, 3, 6, 12]  # 精简后的滞后特征（去掉24）
    config.EPOCHS_ENHANCED = 100
    config.EARLY_STOP_PATIENCE = 30
    
    # v2 特定配置
    config.GLOBAL_LATENT_DIM_V2 = 32  # 降低潜在维度，让 CVAE 真正参与
    config.NUM_ATTENTION_HEADS_V2 = 2  # 减少注意力头数
    config.DROPOUT_V2 = 0.2  # 稍微增加 dropout

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    per_apt_model_dir = os.path.join(config.MODEL_DIR, "enhanced_hier_lstm_v2")
    os.makedirs(per_apt_model_dir, exist_ok=True)

    # 加载数据
    print("\n[1/3] Loading data...")
    train_df, val_df, test_df, scaler_cond, scaler_target = load_and_preprocess_enhanced(
        config.DATA_PATH,
        config.CONTINUOUS_CONDITIONAL_FEATURES,
        config.CATEGORICAL_CONDITIONAL_FEATURES,
        config.TARGET_FEATURES,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 保存 scaler（用于后续预测）
    scaler_cond_path = os.path.join(per_apt_model_dir, "scaler_cond.pkl")
    scaler_target_path = os.path.join(per_apt_model_dir, "scaler_target.pkl")
    joblib.dump(scaler_cond, scaler_cond_path)
    joblib.dump(scaler_target, scaler_target_path)
    print(f"  Scalers saved to {per_apt_model_dir}")

    # 训练
    print("\n[2/3] Training apartments...")
    all_metrics = []
    all_predictions = {}
    all_actuals = {}

    for apt_idx, apt_name in enumerate(tqdm(config.TARGET_FEATURES, desc="Progress")):
        try:
            metrics, predictions, actuals, model = train_and_predict_single_apartment(
                apt_name, train_df, val_df, test_df, config.DEVICE, config
            )

            model_path = os.path.join(per_apt_model_dir, f"{apt_name}_model.pth")
            torch.save(model.state_dict(), model_path)

            # 反归一化预测结果（使用全局 scaler 中对应公寓的部分）
            apt_idx_in_targets = config.TARGET_FEATURES.index(apt_name)
            if scaler_target is not None:
                # 创建单公寓的临时 scaler
                temp_scaler = MinMaxScaler()
                temp_scaler.min_ = scaler_target.min_[apt_idx_in_targets:apt_idx_in_targets+1]
                temp_scaler.scale_ = scaler_target.scale_[apt_idx_in_targets:apt_idx_in_targets+1]
                temp_scaler.data_min_ = scaler_target.data_min_[apt_idx_in_targets:apt_idx_in_targets+1]
                temp_scaler.data_max_ = scaler_target.data_max_[apt_idx_in_targets:apt_idx_in_targets+1]
                temp_scaler.data_range_ = scaler_target.data_range_[apt_idx_in_targets:apt_idx_in_targets+1]
                temp_scaler.n_features_in_ = 1
                
                # 反归一化
                predictions_unscaled = temp_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                actuals_unscaled = temp_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
                
                # 重新计算反归一化后的指标
                mse_unscaled = mean_squared_error(actuals_unscaled, predictions_unscaled)
                mae_unscaled = mean_absolute_error(actuals_unscaled, predictions_unscaled)
                try:
                    r2_unscaled = r2_score(actuals_unscaled, predictions_unscaled)
                except:
                    r2_unscaled = np.nan
                
                # 更新指标（使用反归一化后的值）
                metrics['MSE'] = mse_unscaled
                metrics['RMSE'] = np.sqrt(mse_unscaled)
                metrics['MAE'] = mae_unscaled
                metrics['R2'] = r2_unscaled
            else:
                predictions_unscaled = predictions.flatten()
                actuals_unscaled = actuals.flatten()

            all_metrics.append(metrics)
            all_predictions[apt_name] = predictions_unscaled
            all_actuals[apt_name] = actuals_unscaled

        except Exception as e:
            print(f"\nWarning: {apt_name} failed - {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({'Apartment': apt_name, 'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan})

    # 保存结果
    print("\n[3/3] Saving results...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join("outputs", "enhanced_per_apartment_metrics_v2.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if all_predictions:
        max_len = max(len(v) for v in all_predictions.values())
        pred_df = pd.DataFrame({
            apt: np.pad(all_predictions[apt], (0, max_len - len(all_predictions[apt])), constant_values=np.nan)
            for apt in all_predictions
        }).add_suffix("_Predict")

        actual_df = pd.DataFrame({
            apt: np.pad(all_actuals[apt], (0, max_len - len(all_actuals[apt])), constant_values=np.nan)
            for apt in all_actuals
        })

        result_df = pd.concat([actual_df, pred_df], axis=1)
        pred_path = os.path.join("outputs", "enhanced_per_apartment_predictions_v2.csv")
        result_df.to_csv(pred_path, index=False)

    # 统计
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    valid = metrics_df.dropna(subset=['R2'])
    if len(valid) > 0:
        print(f"\nValid apartments: {len(valid)} / {len(metrics_df)}")
        print(f"Mean R2: {valid['R2'].mean():.4f} (+/- {valid['R2'].std():.4f})")
        print(f"Median R2: {valid['R2'].median():.4f}")
        print(f"Mean RMSE: {valid['RMSE'].mean():.4f}")
        print(f"\nR2 > 0.85: {(valid['R2'] > 0.85).sum()} apartments")
        print(f"R2 > 0.7: {(valid['R2'] > 0.7).sum()} apartments")
        print(f"R2 < 0: {(valid['R2'] < 0).sum()} apartments")

        print(f"\nBest: {valid.loc[valid['R2'].idxmax(), 'Apartment']} (R2={valid['R2'].max():.4f})")
        print(f"Worst: {valid.loc[valid['R2'].idxmin(), 'Apartment']} (R2={valid['R2'].min():.4f})")

    print(f"\nOutputs:")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Predictions: {pred_path}")
    print(f"  - Models: {per_apt_model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

