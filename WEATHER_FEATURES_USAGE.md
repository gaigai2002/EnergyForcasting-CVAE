# 天气特征使用详解：历史天气 vs 当前时刻天气

## 📋 目录
1. [天气特征定义](#1-天气特征定义)
2. [历史天气特征的使用](#2-历史天气特征的使用)
3. [当前时刻天气特征的使用](#3-当前时刻天气特征的使用)
4. [两者的区别和作用](#4-两者的区别和作用)
5. [完整数据流](#5-完整数据流)

---

## 1. 天气特征定义

**代码位置**：`src/hier_config.py` 第34-39行

```python
CONTINUOUS_CONDITIONAL_FEATURES = [
    "temperature",           # 温度
    "humidity",             # 湿度
    "visibility",          # 能见度
    "apparentTemperature",  # 体感温度
    "pressure",             # 气压
    "windSpeed",            # 风速
    "windBearing",         # 风向
    "precipIntensity",      # 降水强度
    "dewPoint",            # 露点
    "precipProbability"   # 降水概率
]
```

**总计**：11个连续天气特征

---

## 2. 历史天气特征的使用

### 2.1 数据提取

**代码位置**：`src/enhanced_data_utils.py` 的 `EnhancedTimeSeriesDataset.__getitem__()`

```python
def __getitem__(self, idx):
    actual_idx = self.valid_indices[idx]
    window_start = actual_idx - self.lookback_steps  # t-24
    window_end = actual_idx                          # t (不包含)
    
    # 窗口特征：提取过去24步的天气特征
    window_cont = self.cont_data[window_start:window_end]
    # window_cont形状: (24, 11) - 24个时间步，每步11个天气特征
```

**数据内容**：
- **时间范围**：`[t-24, t-23, ..., t-2, t-1]`（过去24个时间步）
- **特征维度**：每步11个天气特征
- **总维度**：`(24, 11)`

### 2.2 特征拼接

**代码位置**：第110-112行

```python
# 拼接窗口特征
window_features = np.concatenate([
    window_cont,      # (24, 11) - 历史天气特征
    window_cat_norm,  # (24, 2)  - 历史时间特征（Hour, DayOfWeek）
    window_target     # (24, 1)  - 历史目标值（用电量）
], axis=1)
# window_features形状: (24, 14)

# 展平为向量
window_features = window_features.flatten()
# window_features形状: (336,) = 24 * 14
```

**拼接后的结构**：
```
每个时间步包含：
- 天气特征 (11维): temperature, humidity, visibility, ...
- 时间特征 (2维): Hour, DayOfWeek
- 目标值 (1维): 用电量

24个时间步 × 14维 = 336维向量
```

### 2.3 输入到双向LSTM

**代码位置**：第302-305行

```python
# LSTM处理窗口
features_per_step = self.cont_cond_dim + 2 + 1  # 11 + 2 + 1 = 14
x_window = x_window.view(batch_size, self.lookback_steps, features_per_step)
# x_window形状: (batch_size, 24, 14)

lstm_out, _ = self.lstm(x_window)
# lstm_out形状: (batch_size, 24, 256)
```

**LSTM配置**：
```python
self.lstm = nn.LSTM(
    input_size=14,           # 每个时间步的特征数（11天气 + 2时间 + 1目标）
    hidden_size=128,
    num_layers=2,
    bidirectional=True      # ✅ 双向LSTM
)
```

**作用**：
- **捕捉天气的时序模式**：例如，温度的变化趋势、湿度的累积效应
- **学习天气与用电量的关系**：在不同天气条件下，用电量的变化模式
- **双向处理**：同时考虑前向和后向的天气变化趋势

### 2.4 时间注意力机制

**代码位置**：第307-308行

```python
# 时间注意力
attn_out, attn_weights = self.temporal_attention(lstm_out)
# attn_out形状: (batch_size, 256)
```

**作用**：
- 自动学习哪些时间步的天气信息更重要
- 例如：最近的天气可能比24小时前的天气更重要
- 输出：聚合后的时序特征（256维）

---

## 3. 当前时刻天气特征的使用

### 3.1 数据提取

**代码位置**：第114-116行

```python
# 当前时刻条件特征
c_cont = self.cont_data[actual_idx]
# c_cont形状: (11,) - 当前时刻t的11个天气特征

c_cat = self.df[self.categorical_features].iloc[actual_idx].values
# c_cat形状: (2,) - 当前时刻t的时间特征（Hour, DayOfWeek）
```

**数据内容**：
- **时间点**：当前预测时刻 `t`
- **特征维度**：11个天气特征
- **用途**：作为CVAE的条件输入

### 3.2 条件特征处理

**代码位置**：第283-287行

```python
def process_conditions(self, c_cont, c_cat):
    c_cat = c_cat.long()
    h = self.hour_embed(c_cat[:, 0])      # Hour → 16维嵌入
    d = self.day_embed(c_cat[:, 1])       # DayOfWeek → 8维嵌入
    return torch.cat([c_cont, h, d], dim=1)
    # 输出形状: (11 + 16 + 8 = 35维)
```

**处理流程**：
```
c_cont (11维) - 当前天气特征
    ↓
c_cat (2维) - 当前时间特征
    ↓
嵌入层
    ↓ Hour → 16维
    ↓ DayOfWeek → 8维
    ↓
拼接
    ↓
c (35维) - 条件特征
```

### 3.3 输入到CVAE编码器

**代码位置**：第314-317行

```python
# 全局编码
global_enc_input = torch.cat([temporal_feat, c], dim=1)
# global_enc_input形状: (batch_size, 320 + 35 = 355)

global_h = self.global_encoder(global_enc_input)
global_mu_q = self.global_mu(global_h)
global_logvar_q = self.global_logvar(global_h)
```

**输入组成**：
- `temporal_feat` (320维)：LSTM输出的时序特征
- `c` (35维)：当前时刻的条件特征（天气 + 时间）
- **总计**：355维

**作用**：
- **条件编码**：将当前时刻的天气作为条件，影响潜在变量的分布
- **条件生成**：在给定当前天气的条件下，生成预测值

### 3.4 输入到CVAE先验

**代码位置**：第319-322行

```python
# 全局先验
prior_input = torch.cat([c, lag_encoded], dim=1)
# prior_input形状: (batch_size, 35 + 64 = 99)

global_prior_out = self.global_prior(prior_input)
global_mu_p, global_logvar_p = torch.chunk(global_prior_out, 2, dim=1)
```

**输入组成**：
- `c` (35维)：当前时刻的条件特征（天气 + 时间）
- `lag_encoded` (64维)：编码后的滞后特征

**作用**：
- **先验分布**：在给定当前天气和历史用电量的条件下，定义潜在变量的先验分布
- **KL散度计算**：用于正则化，确保后验分布接近先验分布

### 3.5 输入到CVAE解码器

**代码位置**：第327-328行

```python
# 解码
decoder_input = torch.cat([z_g, temporal_feat, c], dim=1)
# decoder_input形状: (batch_size, 64 + 320 + 35 = 419)

decoder_out = self.decoder(decoder_input)
```

**输入组成**：
- `z_g` (64维)：潜在变量（从后验分布采样）
- `temporal_feat` (320维)：时序特征（LSTM输出）
- `c` (35维)：当前时刻的条件特征（天气 + 时间）

**作用**：
- **条件解码**：在给定当前天气的条件下，从潜在变量解码出预测值
- **天气影响**：当前天气直接影响预测结果

---

## 4. 两者的区别和作用

### 4.1 历史天气特征（window_cont）

| 属性 | 说明 |
|------|------|
| **时间范围** | 过去24个时间步 `[t-24, ..., t-1]` |
| **数据形状** | `(24, 11)` → 展平为 `(336,)` |
| **输入位置** | 双向LSTM |
| **主要作用** | 捕捉天气的时序模式和变化趋势 |
| **学习内容** | - 天气如何随时间变化<br>- 历史天气与用电量的关系<br>- 天气变化的累积效应 |

**示例**：
- 如果过去24小时温度持续上升，模型可以学习到这种趋势
- 如果过去有降雨，模型可以捕捉到湿度累积的影响

### 4.2 当前时刻天气特征（c_cont）

| 属性 | 说明 |
|------|------|
| **时间范围** | 当前预测时刻 `t` |
| **数据形状** | `(11,)` → 处理后为 `(35,)`（加上时间嵌入） |
| **输入位置** | CVAE的条件编码器、先验网络、解码器 |
| **主要作用** | 作为条件，影响预测结果 |
| **学习内容** | - 当前天气如何影响用电量<br>- 在特定天气条件下的预测模式 |

**示例**：
- 如果当前温度很高，模型预测用电量会增加（空调使用）
- 如果当前有降雨，模型预测用电量可能变化（湿度影响）

### 4.3 两者的协同作用

```
历史天气特征 (window_cont)
    ↓
双向LSTM
    ↓ 捕捉时序模式
时序特征 (temporal_feat)
    ↓
    ├─→ CVAE编码器 ←─┐
    │                 │
    └─→ CVAE解码器 ←─┘
                      │
当前天气特征 (c_cont) ┘
    ↓ 作为条件
条件特征 (c)
```

**协同机制**：
1. **历史天气** → LSTM → 学习天气的时序模式
2. **当前天气** → CVAE条件 → 影响预测结果
3. **两者结合** → 既考虑历史趋势，又考虑当前状态

---

## 5. 完整数据流

### 5.1 预测时刻 t 的数据流

```
原始数据（时刻t）
    ↓
[历史天气特征提取]
    ├─ window_cont: (24, 11) - 过去24步的天气
    │   ├─ t-24: [temp, humidity, ...]
    │   ├─ t-23: [temp, humidity, ...]
    │   ├─ ...
    │   └─ t-1:  [temp, humidity, ...]
    │
    └─ [当前天气特征提取]
        └─ c_cont: (11,) - 当前时刻t的天气
            └─ t: [temp, humidity, ...]
    ↓
[特征拼接]
    window_features = concat([
        window_cont (24, 11),
        window_cat (24, 2),
        window_target (24, 1)
    ])
    → (24, 14) → 展平为 (336,)
    ↓
[双向LSTM处理历史天气]
    x_window (batch, 24, 14)
        ↓
    双向LSTM
        ↓
    lstm_out (batch, 24, 256)
        ↓
    时间注意力
        ↓
    attn_out (batch, 256)  ← 历史天气的时序特征
    ↓
[条件特征处理]
    c_cont (batch, 11) + c_cat (batch, 2)
        ↓
    嵌入层
        ↓
    c (batch, 35)  ← 当前天气的条件特征
    ↓
[特征融合]
    temporal_feat = concat([attn_out, lag_encoded])
    → (batch, 320)
    ↓
[CVAE编码器]
    global_enc_input = concat([temporal_feat, c])
    → (batch, 355)
        ↓
    编码器
        ↓
    [mu_q, logvar_q] (batch, 64)
        ↓
    重参数化采样
        ↓
    z_g (batch, 64)
    ↓
[CVAE解码器]
    decoder_input = concat([z_g, temporal_feat, c])
    → (batch, 419)
        ↓
    解码器
        ↓
    decoder_out (batch, 1)
        ↓
    残差连接
        ↓
    最终预测 (batch, 1)
```

### 5.2 关键数据维度变化

| 阶段 | 历史天气特征 | 当前天气特征 | 输出 |
|------|------------|------------|------|
| **数据提取** | `(24, 11)` | `(11,)` | - |
| **特征拼接** | `(336,)` | - | - |
| **LSTM输入** | `(batch, 24, 14)` | - | - |
| **LSTM输出** | - | - | `(batch, 24, 256)` |
| **注意力输出** | - | - | `(batch, 256)` |
| **条件处理** | - | `(batch, 35)` | - |
| **编码器输入** | - | - | `(batch, 355)` |
| **解码器输入** | - | - | `(batch, 419)` |
| **最终预测** | - | - | `(batch, 1)` |

---

## 🔑 关键设计思路

### 1. 历史天气特征的作用

- **时序模式学习**：通过LSTM学习天气如何随时间变化
- **累积效应**：捕捉天气变化的累积影响（例如，连续高温）
- **关系学习**：学习历史天气与用电量的关系

### 2. 当前天气特征的作用

- **条件生成**：作为CVAE的条件，影响预测结果
- **即时影响**：捕捉当前天气对用电量的即时影响
- **条件分布**：在给定当前天气的条件下，生成预测分布

### 3. 两者的结合

- **历史 + 当前**：既考虑历史趋势，又考虑当前状态
- **时序 + 条件**：LSTM处理时序，CVAE处理条件生成
- **互补作用**：历史天气提供上下文，当前天气提供条件

---

## 📊 实际应用示例

### 场景1：预测明天中午的用电量

**历史天气特征（过去24小时）**：
- 温度：20°C → 22°C → 24°C → ... → 28°C（持续上升）
- 湿度：60% → 58% → 55% → ... → 50%（持续下降）

**当前天气特征（预测时刻）**：
- 温度：30°C（高温）
- 湿度：45%（低湿度）

**模型处理**：
1. **LSTM**：学习到温度上升趋势和湿度下降趋势
2. **CVAE条件**：当前高温 + 低湿度 → 预测用电量增加（空调使用）

### 场景2：预测明天凌晨的用电量

**历史天气特征（过去24小时）**：
- 温度：25°C → 23°C → 21°C → ... → 18°C（持续下降）
- 湿度：50% → 55% → 60% → ... → 70%（持续上升）

**当前天气特征（预测时刻）**：
- 温度：15°C（低温）
- 湿度：75%（高湿度）

**模型处理**：
1. **LSTM**：学习到温度下降趋势和湿度上升趋势
2. **CVAE条件**：当前低温 + 高湿度 → 预测用电量可能减少（不需要空调）

---

## 总结

1. **历史天气特征**：
   - 输入到双向LSTM
   - 捕捉天气的时序模式和变化趋势
   - 学习历史天气与用电量的关系

2. **当前天气特征**：
   - 作为CVAE的条件输入
   - 影响潜在变量的分布和预测结果
   - 捕捉当前天气对用电量的即时影响

3. **两者结合**：
   - 历史天气提供上下文和趋势
   - 当前天气提供条件和即时影响
   - 共同作用，提高预测准确性

---

**文档生成时间**：2024年
**代码版本**：`src/enhanced_data_utils.py`、`src/enhanced_model.py`（`main_hier_enhanced.py` 为兼容入口）

