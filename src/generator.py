import pandas as pd
import torch
import numpy as np

def generate_power_consumption(model, input_data, scaler_cond_continuous, scaler_target,
                               continuous_conditional_features, categorical_conditional_features, device):
    """
    根据提供的环境和时间条件，生成 Zone 1、2 和 3 的合成电力消耗数据。

    参数:
        model (nn.Module): 训练好的 CVAE 模型。
        input_data (pd.Series 或 dict): 包含输入条件的一行数据，
                                        键名需与原始数据列名匹配：
                                        'Datetime', 'Temperature', 'Humidity',
                                        'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows'。
        scaler_cond_continuous (MinMaxScaler): 用于连续条件特征的归一化器。
        scaler_target (MinMaxScaler): 用于目标特征的反归一化器。
        continuous_conditional_features (list): 连续条件特征的名称列表。
        categorical_conditional_features (list): 类别条件特征的名称列表。
        device (torch.device): 运行模型的设备 (CPU/GPU)。

    返回:
        dict: 包含生成的 'PowerConsumption_Zone1'、
              'PowerConsumption_Zone2' 和 'PowerConsumption_Zone3' 的字典。
    """
    # 1. 将 input_data 转换为 DataFrame 以进行统一处理
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_df = pd.DataFrame([input_data.to_dict()])
    else:
        raise ValueError("input_data 必须是 pandas Series 或 dictionary 类型。")

    # 确保 'Datetime' 列是日期时间格式
    input_df['Datetime'] = pd.to_datetime(input_df['Datetime'])

    # 2. 提取时间特征 (与训练数据预处理保持一致)
    input_df['Hour'] = input_df['Datetime'].dt.hour
    input_df['DayOfWeek'] = input_df['Datetime'].dt.dayofweek
    input_df['Month'] = input_df['Datetime'].dt.month - 1 # Month 1-12 -> 0-11

    # 3. 准备连续条件特征和类别条件特征
    c_cont_input = input_df[continuous_conditional_features].values.astype(np.float32)
    c_cat_input = input_df[categorical_conditional_features].values.astype(np.int64)

    # 4. 使用预先拟合的 scaler_cond_continuous 对连续输入特征进行归一化
    c_cont_scaled = scaler_cond_continuous.transform(c_cont_input)

    # 将 NumPy 数组转换为 PyTorch 张量并移动到指定设备 (CPU/GPU)
    c_cont_tensor = torch.tensor(c_cont_scaled, dtype=torch.float32).to(device)
    c_cat_tensor = torch.tensor(c_cat_input, dtype=torch.long).to(device)

    # 5. 使用训练好的 CVAE 模型生成合成数据
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        generated_y_scaled = model.generate(c_cont_tensor, c_cat_tensor).cpu().numpy()

    # 6. 将生成的归一化数据反归一化回原始尺度
    generated_y_unscaled = scaler_target.inverse_transform(generated_y_scaled)

    # 7. 格式化输出为字典
    output_dict = {
        'PowerConsumption_Zone1': generated_y_unscaled[0, 0],
        'PowerConsumption_Zone2': generated_y_unscaled[0, 1],
        'PowerConsumption_Zone3': generated_y_unscaled[0, 2]
    }
    return output_dict
