"""
HierCVAE+UniLSTM+Attention 精简优化版训练脚本 (v3)

核心精简思路（信息路径正交化）：
1. 删除 diff 特征（对噪声敏感，MAE 变差）
2. 精简 lag：只保留 [1, 2]（短期确定性）
3. 删除 rolling std，只保留 rolling mean（作为 residual bias）
4. lookback: 24 → 12（MAE 更关心最近点）
5. rolling mean 不作为 encoder 输入，只作为 residual 的 bias
6. Conditioned Residual Gate：用 z_g 和 attention 输出控制 lag 权重
7. Local Attention：只关注最后 k=4 步（增强近期关注）

目标: 降低 MAE，提升模型对"错误信息过度自信"的克制能力
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
    精简版时序数据集（v3）
    
    核心改进：
    1. 删除 diff 特征（对噪声敏感）
    2. 只保留 lag[1,2]（短期确定性）
    3. 只保留 rolling mean（作为 residual bias，不送 encoder）
    4. lookback 缩短到 12（MAE 更关心最近点）
    
    信息路径正交化：
    - 短期确定性：lag[1,2] + rolling_mean（residual）
    - 中期模式：LSTM (lookback=12)
    - 长期/不确定性：global latent z_g
    - 日历条件：Hour/DoW embeddings
    """
    def __init__(self, df, continuous_features, categorical_features,
                 target_col, lookback_steps, lag_features=[1, 2]):
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

        # 计算滞后特征（只保留 lag[1,2]）
        self.lag_data = np.zeros((len(df), len(lag_features)), dtype=np.float32)
        for i, lag in enumerate(lag_features):
            self.lag_data[lag:, i] = self.target_data[:-lag]

        # 只计算 rolling mean（不计算 rolling std）
        self.rolling_mean = np.zeros((len(df), 1), dtype=np.float32)
        for i in range(lookback_steps, len(df)):
            self.rolling_mean[i, 0] = self.target_data[i-lookback_steps:i].mean()

        # 有效索引
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

        # 滞后特征（只 lag[1,2]）
        lag_feats = self.lag_data[actual_idx]

        # 滚动均值（只作为 residual bias，不送 encoder）
        rolling_mean = self.rolling_mean[actual_idx]

        # 目标值
        y = np.array([self.target_data[actual_idx]], dtype=np.float32)

        return (
            torch.tensor(window_features),
            torch.tensor(c_cont),
            torch.tensor(c_cat),
            torch.tensor(lag_feats),
            torch.tensor(rolling_mean),
            torch.tensor(y)
        )


# ============= Local Attention（只关注最后 k 步）=============
class LocalTemporalAttention(nn.Module):
    """
    Local Temporal Attention（v3）
    
    核心改进：
    - 只关注最后 k=4 步（MAE 更关心最近点）
    - 移除 output projection（维度自然降低）
    """
    def __init__(self, hidden_dim, num_heads=2, dropout=0.2, local_window=4):
        super().__init__()
        self.local_window = local_window
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (B, T, H) - LSTM 输出
        
        Returns:
            output: (B, H) - 聚合后的特征
            attn_weights: attention 权重
        """
        batch_size = lstm_output.shape[0]
        
        # 只取最后 local_window 步
        local_output = lstm_output[:, -self.local_window:, :]
        
        # Query
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention（只关注局部窗口）
        attn_output, attn_weights = self.multihead_attn(
            query, local_output, local_output
        )
        
        # Residual connection（与最后一步）
        last_step = lstm_output[:, -1:, :]
        attn_output = self.layer_norm(attn_output + last_step)
        
        return attn_output.squeeze(1), attn_weights


# ============= Conditioned Residual Gate =============
class ConditionedResidualGate(nn.Module):
    """
    Conditioned Residual Gate（v3 核心创新）
    
    核心思路：
    - 用 z_g（长期/不确定性）和 attention 输出（中期模式）控制 lag 的权重
    - 让模型学会"什么时候信 lag，什么时候不信"
    - 降低 MAE（对 regime change 敏感）
    
    公式：
    gate = σ(MLP([z_g, h_attn]))
    r_t = gate ⊙ lag_vector
    ŷ_t = decoder_out + r_t + β·rolling_mean
    """
    def __init__(self, latent_dim, hidden_dim, num_lag_features=2):
        super().__init__()
        self.num_lag_features = num_lag_features
        
        # Gate network: 用 z_g 和 attention 输出控制 lag 权重
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_lag_features),
            nn.Sigmoid()  # 输出 [0, 1] 的权重
        )
        
        # Rolling mean 的固定权重（作为 bias）
        self.rolling_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, z_g, attn_out, lag_feats, rolling_mean):
        """
        Args:
            z_g: (B, latent_dim) - 全局潜在变量
            attn_out: (B, hidden_dim) - attention 输出
            lag_feats: (B, num_lag_features) - 滞后特征
            rolling_mean: (B, 1) - 滚动均值
        
        Returns:
            residual: (B, 1) - 残差项
        """
        # Gate: 根据 z_g 和 attention 输出决定 lag 的权重
        gate_input = torch.cat([z_g, attn_out], dim=1)
        gate_weights = self.gate_net(gate_input)  # (B, num_lag_features)
        
        # 加权 lag features
        lag_residual = (gate_weights * lag_feats).sum(dim=1, keepdim=True)  # (B, 1)
        
        # Rolling mean 作为 bias（固定权重）
        rolling_bias = torch.sigmoid(self.rolling_weight) * rolling_mean
        
        # 总残差
        residual = lag_residual + rolling_bias
        
        return residual


# ============= 精简版模型 =============
class EnhancedHierCVAE(nn.Module):
    """
    精简版 HierCVAE（v3）
    
    架构改进：
    1. 删除 diff 特征编码器
    2. 简化 lag encoder（只有 lag[1,2]）
    3. rolling mean 不作为 encoder 输入，只作为 residual bias
    4. Local Attention（只关注最后 k=4 步）
    5. Conditioned Residual Gate（核心创新）
    6. lookback=12（缩短窗口）
    """
    def __init__(self, cont_cond_dim, num_lag_features=2,
                 global_latent_dim=32,
                 cat_dims=None, embedding_dims=None, lookback_steps=12,
                 hidden_dim=128, dropout=0.2, num_attention_heads=2,
                 local_window=4):
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

        # 简化 lag encoder（只有 lag[1,2]，不包含 rolling mean）
        self.lag_encoder = nn.Sequential(
            nn.Linear(num_lag_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU()
        )
        lag_feat_dim = hidden_dim // 2  # 64

        # 单向 LSTM（lookback=12）
        temporal_input_dim = cont_cond_dim + 2 + 1  # cont + cat_norm + target
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
            bidirectional=False  # 单向 LSTM
        )
        lstm_output_dim = hidden_dim  # 单向

        # Local Temporal Attention（只关注最后 k=4 步）
        self.temporal_attention = LocalTemporalAttention(
            hidden_dim=lstm_output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            local_window=local_window
        )

        # 特征融合: LSTM输出 + lag编码
        temporal_feat_dim = lstm_output_dim + lag_feat_dim  # 128 + 64 = 192

        # 全局编码器（不包含 rolling mean）
        global_enc_input = temporal_feat_dim + self.cond_dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_enc_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.global_mu = nn.Linear(hidden_dim // 2, global_latent_dim)
        self.global_logvar = nn.Linear(hidden_dim // 2, global_latent_dim)

        # 全局先验（不包含 rolling mean）
        self.global_prior = nn.Sequential(
            nn.Linear(self.cond_dim + lag_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, global_latent_dim * 2)
        )

        # 解码器
        decoder_input = global_latent_dim + temporal_feat_dim + self.cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Conditioned Residual Gate（核心创新）
        self.residual_gate = ConditionedResidualGate(
            latent_dim=global_latent_dim,
            hidden_dim=lstm_output_dim,
            num_lag_features=num_lag_features
        )

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

    def forward(self, x_window, c_cont, c_cat, lag_feats, rolling_mean, targets=None):
        """
        Args:
            x_window: (B, lookback_steps * features_per_step) - 展平的窗口特征
            c_cont: (B, cont_cond_dim) - 当前时刻连续特征
            c_cat: (B, 2) - 当前时刻类别特征
            lag_feats: (B, num_lag_features) - 滞后特征 [lag1, lag2]
            rolling_mean: (B, 1) - 滚动均值（只作为 residual bias）
            targets: (B, 1) - 目标值（训练时使用）
        
        Returns:
            dict: 包含预测和潜在变量信息
        """
        batch_size = x_window.shape[0]
        c = self.process_conditions(c_cont, c_cat)

        # 编码滞后特征（不包含 rolling mean）
        lag_encoded = self.lag_encoder(lag_feats)

        # LSTM处理窗口
        features_per_step = self.cont_cond_dim + 2 + 1
        x_window = x_window.view(batch_size, self.lookback_steps, features_per_step)
        lstm_out, _ = self.lstm(x_window)

        # Local Temporal Attention（只关注最后 k=4 步）
        attn_out, attn_weights = self.temporal_attention(lstm_out)

        # 融合特征
        temporal_feat = torch.cat([attn_out, lag_encoded], dim=1)

        # 全局编码（不包含 rolling mean）
        global_enc_input = torch.cat([temporal_feat, c], dim=1)
        global_h = self.global_encoder(global_enc_input)
        global_mu_q = self.global_mu(global_h)
        global_logvar_q = self.global_logvar(global_h)

        # 全局先验（不包含 rolling mean）
        prior_input = torch.cat([c, lag_encoded], dim=1)
        global_prior_out = self.global_prior(prior_input)
        global_mu_p, global_logvar_p = torch.chunk(global_prior_out, 2, dim=1)

        z_g = self.reparameterize(global_mu_q, global_logvar_q)

        # 解码
        decoder_input = torch.cat([z_g, temporal_feat, c], dim=1)
        decoder_out = self.decoder(decoder_input)

        # Conditioned Residual Gate（核心创新）
        # 用 z_g 和 attention 输出控制 lag 权重，rolling_mean 作为 bias
        residual = self.residual_gate(z_g, attn_out, lag_feats, rolling_mean)
        
        # 最终预测
        pred = decoder_out + residual

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
    训练单个公寓的模型并预测（v3）
    思路：
    1. 创建数据加载器（只包含 lag[1,2] 和 rolling_mean）
    2. 初始化模型和优化器
    3. 训练循环（KL warmup、早停、学习率调度）
    4. 加载最佳模型进行预测
    5. 计算评估指标
    """
    print(f"\n{'='*60}")
    print(f"Processing: {apt_name}")
    print(f"{'='*60}")

    # 配置
    lookback_steps = getattr(cfg, 'LOOKBACK_STEPS_V3', 12)  # v3: 12
    lag_features = getattr(cfg, 'LAG_FEATURES_V3', [1, 2])  # v3: 只保留 [1,2]

    train_loader, val_loader, test_loader = create_dataloaders_enhanced(
        train_df, val_df, test_df,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name, cfg.BATCH_SIZE, lookback_steps, lag_features
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Lookback: {lookback_steps}, Lag features: {lag_features}")

    # 模型（v3：删除 num_diff_features）
    model = EnhancedHierCVAE(
        cont_cond_dim=len(cfg.CONTINUOUS_CONDITIONAL_FEATURES),
        num_lag_features=len(lag_features),
        global_latent_dim=getattr(cfg, 'GLOBAL_LATENT_DIM_V3', 32),
        cat_dims=cfg.CAT_DIMS,
        embedding_dims=cfg.EMBEDDING_DIMS,
        lookback_steps=lookback_steps,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=getattr(cfg, 'DROPOUT_V3', 0.2),
        num_attention_heads=getattr(cfg, 'NUM_ATTENTION_HEADS_V3', 2),
        local_window=getattr(cfg, 'LOCAL_WINDOW', 4)
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

    epochs = getattr(cfg, 'EPOCHS_V3', cfg.EPOCHS)

    for epoch in range(epochs):
        model.train()
        warmup_epochs = getattr(cfg, 'KL_WARMUP_EPOCHS', 30)
        beta_factor = min(1.0, (epoch + 1) / warmup_epochs)

        for batch in train_loader:
            # v3: 删除 diff_feats，rolling_mean 是单独的 tensor
            x_window, c_cont, c_cat, lag_feats, rolling_mean, y = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(x_window, c_cont, c_cat, lag_feats, rolling_mean, targets=y)
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
                x_window, c_cont, c_cat, lag_feats, rolling_mean, y = [b.to(device) for b in batch]
                outputs = model(x_window, c_cont, c_cat, lag_feats, rolling_mean, targets=y)
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
            x_window, c_cont, c_cat, lag_feats, rolling_mean, y = [b.to(device) for b in batch]
            outputs = model(x_window, c_cont, c_cat, lag_feats, rolling_mean, targets=None)
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
                     historical_buffer=None, lookback_steps=12, lag_features=[1, 2]):
    """
    对新数据进行预测（v3）
    
    Args:
        apt_name: 公寓名称
        new_df: 新数据 DataFrame（需要包含所有特征）
        model_dir: 模型目录
        device: 设备
        cfg: 配置
        historical_buffer: 历史数据缓冲区（用于计算 lag 和 rolling_mean）
        lookback_steps: 回看步数（v3: 12）
        lag_features: 滞后特征（v3: [1,2]）
    
    Returns:
        predictions: 预测值数组（反归一化后）
    """
    # 加载 scaler
    scaler_cond_path = os.path.join(model_dir, "scaler_cond.pkl")
    scaler_target_path = os.path.join(model_dir, "scaler_target.pkl")
    
    if not os.path.exists(scaler_cond_path) or not os.path.exists(scaler_target_path):
        raise FileNotFoundError("Scaler files not found")
    
    scaler_cond = joblib.load(scaler_cond_path)
    scaler_target = joblib.load(scaler_target_path)
    
    # 准备数据
    df = new_df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    
    # 填充缺失值
    target_features = cfg.TARGET_FEATURES
    df[target_features] = df[target_features].fillna(0)
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].fillna(
        df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].mean()
    )
    
    # 如果有历史缓冲区，合并数据
    if historical_buffer is not None:
        df = pd.concat([historical_buffer, df], ignore_index=True)
    
    # 归一化
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = scaler_cond.transform(df[cfg.CONTINUOUS_CONDITIONAL_FEATURES])
    df[target_features] = scaler_target.transform(df[target_features])
    
    # Clip
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].clip(0, 1)
    df[target_features] = df[target_features].clip(0, 1)
    
    # 创建数据集
    dataset = EnhancedTimeSeriesDataset(
        df, cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name, lookback_steps, lag_features
    )
    
    # 加载模型
    model_path = os.path.join(model_dir, f"{apt_name}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 只取新数据部分
    if historical_buffer is not None:
        hist_len = len(historical_buffer)
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
        global_latent_dim=getattr(cfg, 'GLOBAL_LATENT_DIM_V3', 32),
        cat_dims=cfg.CAT_DIMS,
        embedding_dims=cfg.EMBEDDING_DIMS,
        lookback_steps=lookback_steps,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=getattr(cfg, 'DROPOUT_V3', 0.2),
        num_attention_heads=getattr(cfg, 'NUM_ATTENTION_HEADS_V3', 2),
        local_window=getattr(cfg, 'LOCAL_WINDOW', 4)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预测
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            x_window, c_cont, c_cat, lag_feats, rolling_mean, _ = [b.to(device) for b in batch]
            outputs = model(x_window, c_cont, c_cat, lag_feats, rolling_mean, targets=None)
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
    主函数：训练所有公寓的模型（v3）
    
    流程：
    1. 加载和预处理数据
    2. 循环训练每个公寓（单独模型）
    3. 保存模型和评估结果
    4. 输出统计信息
    
    v3 核心改进：
    - 删除 diff 特征
    - 只保留 lag[1,2]
    - lookback=12
    - Conditioned Residual Gate
    - Local Attention (k=4)
    """
    print("=" * 60)
    print("Enhanced HierCVAE+UniLSTM+LocalAttention v3")
    print("Key improvements:")
    print("  - Removed diff features (noise-sensitive)")
    print("  - Lag features: [1,2] only (short-term deterministic)")
    print("  - Lookback: 12 (MAE-focused)")
    print("  - Conditioned Residual Gate (regime-aware)")
    print("  - Local Attention (k=4)")
    print("=" * 60)

    # v3 配置
    config.LOOKBACK_STEPS_V3 = 12  # v3: 缩短到 12
    config.LAG_FEATURES_V3 = [1, 2]  # v3: 只保留 [1,2]
    config.GLOBAL_LATENT_DIM_V3 = 32
    config.NUM_ATTENTION_HEADS_V3 = 2
    config.DROPOUT_V3 = 0.2
    config.LOCAL_WINDOW = 4  # Local attention window
    config.EPOCHS_V3 = 100
    config.EARLY_STOP_PATIENCE = 30

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    per_apt_model_dir = os.path.join(config.MODEL_DIR, "enhanced_hier_lstm_v3")
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
    metrics_path = os.path.join("outputs", "enhanced_per_apartment_metrics_v3.csv")
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
        pred_path = os.path.join("outputs", "enhanced_per_apartment_predictions_v3.csv")
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
        print(f"Mean MAE: {valid['MAE'].mean():.4f}")
        print(f"\nR2 > 0.85: {(valid['R2'] > 0.85).sum()} apartments")
        print(f"R2 > 0.7: {(valid['R2'] > 0.7).sum()} apartments")
        print(f"R2 < 0: {(valid['R2'] < 0).sum()} apartments")

        print(f"\nBest: {valid.loc[valid['R2'].idxmax(), 'Apartment']} (R2={valid['R2'].max():.4f})")
        print(f"Worst: {valid.loc[valid['R2'].idxmin(), 'Apartment']} (R2={valid['R2'].min():.4f})")

    print(f"\nOutputs:")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Models: {per_apt_model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

