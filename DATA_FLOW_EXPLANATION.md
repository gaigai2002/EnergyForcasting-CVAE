# 数据流程详解：双向LSTM + CVAE 的数据处理和使用

## 📋 目录
1. [数据预处理流程](#1-数据预处理流程)
2. [特征工程](#2-特征工程)
3. [数据输入格式](#3-数据输入格式)
4. [双向LSTM的数据使用](#4-双向lstm的数据使用)
5. [CVAE的数据使用](#5-cvae的数据使用)
6. [数据融合和预测](#6-数据融合和预测)
7. [完整数据流图](#7-完整数据流图)

---

## 1. 数据预处理流程

### 1.1 原始数据加载

**代码位置**：`src/enhanced_data_utils.py` 的 `load_and_preprocess_enhanced()`

```python
df = pd.read_csv(data_path)
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
```

**数据格式**：
- 时间序列数据，每行代表一个时间点（例如每小时）
- 包含：
  - **连续特征**：`temperature`, `humidity`, `visibility`, `apparentTemperature`, `pressure`, `windSpeed`, `windBearing`, `precipIntensity`, `dewPoint`, `precipProbability` (11维)
  - **类别特征**：`Hour` (0-23), `DayOfWeek` (0-6)
  - **目标特征**：`Apt_1`, `Apt_2`, ..., `Apt_114` (114个公寓的用电量)

### 1.2 时间特征提取

**代码位置**：第389-390行

```python
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
```

**作用**：提取时间特征，用于捕捉周期性模式（一天24小时、一周7天）

### 1.3 数据划分

**代码位置**：第396-403行

```python
n = len(df)
train_end = int(n * train_ratio)      # 60%
val_end = int(n * (train_ratio + val_ratio))  # 80%

train_df = df.iloc[:train_end].copy()      # 前60%
val_df = df.iloc[train_end:val_end].copy() # 中间20%
test_df = df.iloc[val_end:].copy()         # 最后20%
```

**划分方式**：按时间顺序划分，确保训练时看不到未来数据

### 1.4 归一化

**代码位置**：第405-416行

```python
scaler_cond = MinMaxScaler()
scaler_target = MinMaxScaler()

# ✅ 只在训练集上fit
train_df[continuous_features] = scaler_cond.fit_transform(train_df[continuous_features])
train_df[target_features] = scaler_target.fit_transform(train_df[target_features])

# ✅ 验证集和测试集只transform
val_df[continuous_features] = scaler_cond.transform(val_df[continuous_features])
val_df[target_features] = scaler_target.transform(val_df[target_features])
test_df[continuous_features] = scaler_cond.transform(test_df[continuous_features])
test_df[target_features] = scaler_target.transform(test_df[target_features])
```

**归一化范围**：所有特征和目标值都被归一化到 [0, 1]

---

## 2. 特征工程

**代码位置**：`EnhancedTimeSeriesDataset.__init__` 第61-86行

### 2.1 滞后特征 (Lag Features)

```python
# 计算滞后特征
self.lag_data = np.zeros((len(df), len(lag_features)), dtype=np.float32)
for i, lag in enumerate(lag_features):  # lag_features = [1, 2, 3, 6, 12, 24]
    self.lag_data[lag:, i] = self.target_data[:-lag]
```

**含义**：
- `lag_data[t, 0] = target[t-1]`  (1步前)
- `lag_data[t, 1] = target[t-2]`  (2步前)
- `lag_data[t, 2] = target[t-3]`  (3步前)
- `lag_data[t, 3] = target[t-6]`  (6步前)
- `lag_data[t, 4] = target[t-12]` (12步前)
- `lag_data[t, 5] = target[t-24]` (24步前)

**作用**：显式提供历史值，帮助模型捕捉短期和中期依赖

### 2.2 差分特征 (Difference Features)

```python
# 一阶差分: target[t-1] - target[t-2]
self.diff_data[2:, 0] = self.target_data[1:-1] - self.target_data[:-2]
# 二阶差分: diff[t-1] - diff[t-2]
self.diff_data[3:, 1] = self.diff_data[2:-1, 0] - self.diff_data[1:-2, 0]
```

**含义**：
- `diff_1[t] = target[t-1] - target[t-2]`  (一阶差分，捕捉变化率)
- `diff_2[t] = diff_1[t-1] - diff_1[t-2]` (二阶差分，捕捉加速度)

**作用**：捕捉变化趋势和加速度，帮助模型理解数据的动态变化

### 2.3 滚动统计特征 (Rolling Statistics)

```python
# 计算滚动统计 (使用lookback窗口)
self.rolling_mean = np.zeros(len(df), dtype=np.float32)
self.rolling_std = np.zeros(len(df), dtype=np.float32)
for i in range(lookback_steps, len(df)):
    window = self.target_data[i-lookback_steps:i]  # 窗口：[i-24, i)
    self.rolling_mean[i] = window.mean()
    self.rolling_std[i] = window.std() if window.std() > 0 else 0.01
```

**含义**：
- `rolling_mean[t]` = 过去24步的平均值
- `rolling_std[t]` = 过去24步的标准差

**作用**：提供局部统计信息，帮助模型理解数据的局部模式

---

## 3. 数据输入格式

**代码位置**：`EnhancedTimeSeriesDataset.__getitem__` 第95-138行

### 3.1 单个样本的数据结构

当预测时刻 `t` 时，数据集返回以下7个组件：

```python
return (
    x_window,      # 时间窗口数据 → 输入双向LSTM
    c_cont,        # 当前时刻连续条件特征 → 输入CVAE
    c_cat,         # 当前时刻类别条件特征 → 输入CVAE
    lag_feats,     # 滞后特征 → 输入lag_encoder
    diff_feats,    # 差分特征 → 输入lag_encoder
    roll_stats,    # 滚动统计 → 输入lag_encoder
    y              # 目标值（用于训练）
)
```

### 3.2 各组件详细说明

#### (1) `x_window` - 时间窗口数据

**形状**：`(lookback_steps * features_per_step,)` → reshape为 `(lookback_steps, features_per_step)`

**内容**：
```python
window_start = actual_idx - lookback_steps  # t-24
window_end = actual_idx                      # t (不包含)

# 每个时间步包含：
# [连续特征(11维) + 类别特征(2维) + 目标值(1维)] = 14维
window_features = np.concatenate([
    window_cont,      # (24, 11) - 连续特征
    window_cat_norm,  # (24, 2)  - 类别特征（归一化）
    window_target     # (24, 1)  - 目标值
], axis=1)  # → (24, 14)

# 展平为 (24*14,) = (336,)
window_features = window_features.flatten()
```

**用途**：输入到**双向LSTM**，捕捉时序依赖

#### (2) `c_cont` - 当前时刻连续条件特征

**形状**：`(11,)`

**内容**：当前时刻 `t` 的连续特征（温度、湿度等）

**用途**：输入到**CVAE的条件编码器**

#### (3) `c_cat` - 当前时刻类别条件特征

**形状**：`(2,)`

**内容**：当前时刻 `t` 的类别特征（Hour, DayOfWeek）

**用途**：经过Embedding层后，输入到**CVAE的条件编码器**

#### (4) `lag_feats` - 滞后特征

**形状**：`(6,)`

**内容**：`[target[t-1], target[t-2], target[t-3], target[t-6], target[t-12], target[t-24]]`

**用途**：与差分特征、滚动统计一起输入到**lag_encoder**

#### (5) `diff_feats` - 差分特征

**形状**：`(2,)`

**内容**：`[diff_1[t], diff_2[t]]`

**用途**：与滞后特征、滚动统计一起输入到**lag_encoder**

#### (6) `roll_stats` - 滚动统计

**形状**：`(2,)`

**内容**：`[rolling_mean[t], rolling_std[t]]`

**用途**：与滞后特征、差分特征一起输入到**lag_encoder**

#### (7) `y` - 目标值

**形状**：`(1,)`

**内容**：当前时刻 `t` 的真实目标值（用于计算损失）

---

## 4. 双向LSTM的数据使用

**代码位置**：`EnhancedHierCVAE.forward` 第302-308行

### 4.1 数据输入

```python
# LSTM处理窗口
features_per_step = self.cont_cond_dim + 2 + 1  # 11 + 2 + 1 = 14
x_window = x_window.view(batch_size, self.lookback_steps, features_per_step)
# x_window形状: (batch_size, 24, 14)

lstm_out, _ = self.lstm(x_window)
# lstm_out形状: (batch_size, 24, hidden_dim*2) = (batch_size, 24, 256)
```

**输入数据**：
- **形状**：`(batch_size, 24, 14)`
- **内容**：过去24个时间步的特征
  - 每个时间步包含：连续特征(11) + 类别特征(2) + 目标值(1) = 14维

**双向LSTM配置**：
```python
self.lstm = nn.LSTM(
    input_size=14,           # 每个时间步的特征数
    hidden_size=128,         # 隐藏层维度
    num_layers=2,            # 2层LSTM
    bidirectional=True      # ✅ 双向
)
```

**输出**：
- **形状**：`(batch_size, 24, 256)`
- **含义**：每个时间步的LSTM输出（前向+后向，各128维）

### 4.2 时间注意力机制

```python
# 时间注意力
attn_out, attn_weights = self.temporal_attention(lstm_out)
# attn_out形状: (batch_size, 256)
```

**作用**：
- 使用多头注意力机制聚合24个时间步的信息
- 自动学习哪些时间步更重要
- 输出：聚合后的时序特征（256维）

**注意力机制**：
```python
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
```

**输出**：`attn_out` - 聚合后的时序特征（256维）

---

## 5. CVAE的数据使用

**代码位置**：`EnhancedHierCVAE.forward` 第296-333行

### 5.1 条件特征处理

#### (1) 类别特征嵌入

```python
def process_conditions(self, c_cont, c_cat):
    c_cat = c_cat.long()
    h = self.hour_embed(c_cat[:, 0])      # Hour → 16维
    d = self.day_embed(c_cat[:, 1])        # DayOfWeek → 8维
    return torch.cat([c_cont, h, d], dim=1)  # → (11+16+8=35维)
```

**输入**：
- `c_cont`: 当前时刻连续特征（11维）
- `c_cat`: 当前时刻类别特征（2维：Hour, DayOfWeek）

**输出**：
- `c`: 条件特征（35维）

#### (2) 滞后/差分/滚动统计特征编码

```python
# 编码滞后特征 + 差分特征 + 滚动统计
lag_diff_input = torch.cat([lag_feats, diff_feats, roll_stats], dim=1)
# lag_diff_input形状: (batch_size, 6+2+2=10)

lag_encoded = self.lag_encoder(lag_diff_input)
# lag_encoded形状: (batch_size, 64)
```

**输入**：
- `lag_feats`: (6,) - 滞后特征
- `diff_feats`: (2,) - 差分特征
- `roll_stats`: (2,) - 滚动统计
- **总计**：10维

**编码器结构**：
```python
self.lag_encoder = nn.Sequential(
    nn.Linear(10, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 64),
    nn.LayerNorm(64),
    nn.GELU(),
)
```

**输出**：`lag_encoded` - 编码后的特征（64维）

### 5.2 CVAE编码器（后验分布 q(z|x,c)）

```python
# 融合特征
temporal_feat = torch.cat([attn_out, lag_encoded], dim=1)
# temporal_feat形状: (batch_size, 256+64=320)

# 全局编码
global_enc_input = torch.cat([temporal_feat, c], dim=1)
# global_enc_input形状: (batch_size, 320+35=355)

global_h = self.global_encoder(global_enc_input)
# global_h形状: (batch_size, 128)

global_mu_q = self.global_mu(global_h)      # 均值
global_logvar_q = self.global_logvar(global_h)  # 对数方差
# 形状: (batch_size, 64)
```

**输入数据**：
- `temporal_feat`: LSTM+注意力输出（256维）+ lag_encoded（64维）= 320维
- `c`: 条件特征（35维）
- **总计**：355维

**编码器结构**：
```python
self.global_encoder = nn.Sequential(
    nn.Linear(355, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.LayerNorm(128),
    nn.GELU(),
)
```

**输出**：
- `global_mu_q`: 潜在变量的均值（64维）
- `global_logvar_q`: 潜在变量的对数方差（64维）

### 5.3 CVAE先验分布（p(z|c)）

```python
# 全局先验
prior_input = torch.cat([c, lag_encoded], dim=1)
# prior_input形状: (batch_size, 35+64=99)

global_prior_out = self.global_prior(prior_input)
# global_prior_out形状: (batch_size, 128)

global_mu_p, global_logvar_p = torch.chunk(global_prior_out, 2, dim=1)
# 形状: (batch_size, 64)
```

**输入数据**：
- `c`: 条件特征（35维）
- `lag_encoded`: 编码后的滞后特征（64维）
- **总计**：99维

**先验网络结构**：
```python
self.global_prior = nn.Sequential(
    nn.Linear(99, 128),
    nn.LayerNorm(128),
    nn.GELU(),
    nn.Linear(128, 128)  # 输出128维，然后chunk为mu和logvar
)
```

**输出**：
- `global_mu_p`: 先验均值（64维）
- `global_logvar_p`: 先验对数方差（64维）

### 5.4 重参数化采样

```python
z_g = self.reparameterize(global_mu_q, global_logvar_q)
# z_g形状: (batch_size, 64)

def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std  # z = μ + ε * σ
```

**作用**：从后验分布采样潜在变量 `z_g`（64维）

### 5.5 CVAE解码器

```python
# 解码
decoder_input = torch.cat([z_g, temporal_feat, c], dim=1)
# decoder_input形状: (batch_size, 64+320+35=419)

decoder_out = self.decoder(decoder_input)
# decoder_out形状: (batch_size, 1)
```

**输入数据**：
- `z_g`: 潜在变量（64维）
- `temporal_feat`: 时序特征（320维）
- `c`: 条件特征（35维）
- **总计**：419维

**解码器结构**：
```python
self.decoder = nn.Sequential(
    nn.Linear(419, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 128),
    nn.LayerNorm(128),
    nn.GELU(),
    nn.Dropout(0.05),
    nn.Linear(128, 64),
    nn.LayerNorm(64),
    nn.GELU(),
    nn.Linear(64, 1)  # 输出预测值
)
```

**输出**：`decoder_out` - 初步预测值（1维）

---

## 6. 数据融合和预测

**代码位置**：第330-333行

### 6.1 增强残差连接

```python
# 增强残差连接
lag1 = lag_feats[:, 0:1]  # target[t-1]
lag2 = lag_feats[:, 1:2]  # target[t-2]

pred = decoder_out + torch.sigmoid(self.residual_weight_1) * lag1 + torch.sigmoid(self.residual_weight_2) * lag2
```

**作用**：
- 将解码器输出与滞后特征（lag1, lag2）结合
- 可学习的权重自动平衡解码器输出和滞后特征的重要性
- 增强短期依赖的捕捉

**最终输出**：`pred` - 最终预测值（1维）

---

## 7. 完整数据流图

```
原始数据 (CSV)
    ↓
[数据预处理]
    ├─ 时间特征提取 (Hour, DayOfWeek)
    ├─ 数据划分 (train/val/test)
    └─ 归一化 (MinMaxScaler)
    ↓
[特征工程]
    ├─ 滞后特征: [target[t-1], target[t-2], ..., target[t-24]]
    ├─ 差分特征: [diff_1[t], diff_2[t]]
    └─ 滚动统计: [rolling_mean[t], rolling_std[t]]
    ↓
[数据输入] (时刻t)
    ├─ x_window: (24, 14) → 双向LSTM
    ├─ c_cont: (11,) → CVAE条件
    ├─ c_cat: (2,) → CVAE条件（嵌入后）
    ├─ lag_feats: (6,) → lag_encoder
    ├─ diff_feats: (2,) → lag_encoder
    └─ roll_stats: (2,) → lag_encoder
    ↓
[双向LSTM分支]
    x_window (24, 14)
        ↓
    双向LSTM (hidden_dim=128, bidirectional=True)
        ↓
    lstm_out (24, 256)
        ↓
    时间注意力机制 (MultiheadAttention)
        ↓
    attn_out (256,)
    ↓
[lag_encoder分支]
    [lag_feats, diff_feats, roll_stats] (10,)
        ↓
    lag_encoder (Linear layers)
        ↓
    lag_encoded (64,)
    ↓
[特征融合]
    temporal_feat = concat([attn_out, lag_encoded])  (320,)
    ↓
[CVAE编码器]
    global_enc_input = concat([temporal_feat, c])  (355,)
        ↓
    global_encoder
        ↓
    [global_mu_q, global_logvar_q]  (64,)
        ↓
    重参数化采样
        ↓
    z_g (64,)
    ↓
[CVAE解码器]
    decoder_input = concat([z_g, temporal_feat, c])  (419,)
        ↓
    decoder
        ↓
    decoder_out (1,)
    ↓
[残差连接]
    pred = decoder_out + w1*lag1 + w2*lag2
    ↓
最终预测值 (1,)
```

---

## 📊 数据维度总结

| 组件 | 输入维度 | 输出维度 | 用途 |
|------|---------|---------|------|
| **x_window** | (batch, 24, 14) | - | 输入双向LSTM |
| **双向LSTM** | (batch, 24, 14) | (batch, 24, 256) | 捕捉时序依赖 |
| **时间注意力** | (batch, 24, 256) | (batch, 256) | 聚合时序信息 |
| **lag_encoder** | (batch, 10) | (batch, 64) | 编码滞后/差分/统计特征 |
| **条件特征c** | (batch, 35) | (batch, 35) | CVAE条件输入 |
| **CVAE编码器** | (batch, 355) | (batch, 64) | 编码到潜在空间 |
| **CVAE解码器** | (batch, 419) | (batch, 1) | 生成预测 |
| **最终预测** | - | (batch, 1) | 预测值 |

---

## 🔑 关键设计思路

1. **双向LSTM**：
   - 处理时间窗口数据（24步历史）
   - 捕捉前向和后向的时序依赖
   - 输出聚合后的时序特征

2. **CVAE**：
   - 使用条件特征（当前时刻的天气、时间等）
   - 结合时序特征（LSTM输出）和滞后特征
   - 通过潜在变量捕捉不确定性

3. **特征工程**：
   - 滞后特征：显式提供历史值
   - 差分特征：捕捉变化趋势
   - 滚动统计：提供局部统计信息

4. **残差连接**：
   - 结合解码器输出和滞后特征
   - 增强短期依赖的捕捉

---

**文档生成时间**：2024年
**代码版本**：`src/enhanced_data_utils.py`、`src/enhanced_model.py`、`src/enhanced_trainer.py`（`main_hier_enhanced.py` 为兼容入口）

