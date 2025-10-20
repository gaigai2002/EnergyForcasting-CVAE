import torch
import torch.optim as optim
import os
import joblib
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# 导入自定义模块
from src import config
from src.data_utils import load_and_preprocess_data, create_dataloaders, save_scalers, load_scalers
from src.models import UpgradedCVAE
from src.losses import combined_loss_function
from src.trainer import train_model
from src.generator import generate_power_consumption
from src.visualization import plot_training_mse, plot_prediction_vs_actual, \
    plot_prediction_with_confidence_interval, plot_generated_vs_actual_density, plot_time_series_comparison

def main():
    # 0. 创建模型 artifacts 目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. 数据加载与预处理
    original_df, train_df, test_df, scaler_cond_continuous, scaler_target = load_and_preprocess_data(
        config.DATA_PATH,
        config.CONTINUOUS_CONDITIONAL_FEATURES,
        config.CATEGORICAL_CONDITIONAL_FEATURES,
        config.TARGET_FEATURES,
        config.TRAIN_TEST_SPLIT_RATIO
    )

    # 根据数据更新TARGET_DIM
    config.TARGET_DIM = len(config.TARGET_FEATURES)

    # 2. 创建 DataLoader
    train_loader, test_loader, X_test_cont_tensor, X_test_cat_tensor, y_test_tensor = create_dataloaders(
        train_df, test_df,
        config.CONTINUOUS_CONDITIONAL_FEATURES,
        config.CATEGORICAL_CONDITIONAL_FEATURES,
        config.TARGET_FEATURES,
        config.BATCH_SIZE
    )

    # 3. 初始化模型、优化器
    model = UpgradedCVAE(
        target_dim=config.TARGET_DIM,
        cont_cond_dim=config.CONT_COND_DIM,
        latent_dim=config.LATENT_DIM,
        cat_dims=config.CAT_DIMS,
        embedding_dims=config.EMBEDDING_DIMS
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. 训练模型
    print("\n--- 开始训练模型 ---")
    trained_model, train_losses, test_mses = train_model(
        model, train_loader, test_loader, optimizer,
        config.DEVICE, config, combined_loss_function
    )
    print("--- 模型训练完成 ---")

    # 5. 保存模型和归一化器
    torch.save(trained_model.state_dict(), config.MODEL_PATH)
    save_scalers(scaler_cond_continuous, scaler_target, config.MODEL_DIR, config.SCALER_COND_PATH, config.SCALER_TARGET_PATH)
    print(f"训练好的模型已保存到: {config.MODEL_PATH}")

    # 6. 评估与可视化
    print("\n--- 开始模型评估与可视化 ---")
    test_mses_np = np.array(test_mses)
    plot_training_mse(test_mses_np, config.TARGET_FEATURES)

    # 计算整体预测值和真实值 (用于 prediction_vs_actual 和 density plots)
    trained_model.eval()
    all_recon_x = []
    all_x = []
    with torch.no_grad():
        for c_cont, c_cat, x in test_loader:
            c_cont, c_cat, x = c_cont.to(config.DEVICE), c_cat.to(config.DEVICE), x.to(config.DEVICE)
            recon_x, _, _ = trained_model(x, c_cont, c_cat)
            all_recon_x.append(recon_x)
            all_x.append(x)

    all_recon_x_scaled = torch.cat(all_recon_x, dim=0).cpu().numpy()
    all_x_scaled = torch.cat(all_x, dim=0).cpu().numpy()

    all_recon_x_unscaled = scaler_target.inverse_transform(all_recon_x_scaled)
    all_x_unscaled = scaler_target.inverse_transform(all_x_scaled)

    plot_prediction_vs_actual(all_x_unscaled, all_recon_x_unscaled, config.TARGET_FEATURES, config.NUM_POINTS_TO_PLOT_TRAINING_PRED)
    plot_generated_vs_actual_density(all_x_unscaled, all_recon_x_unscaled, config.TARGET_FEATURES)
    plot_prediction_with_confidence_interval(
        trained_model, X_test_cont_tensor, X_test_cat_tensor, y_test_tensor,
        scaler_target, config.TARGET_FEATURES, config.DEVICE,
        config.NUM_SAMPLES_FOR_CI, config.NUM_POINTS_TO_PLOT_CI_VIS
    )
    print("--- 模型评估与可视化完成 ---")

    # 7. 示例生成
    print("\n--- 运行生成函数示例 ---")
    example_input_data_1 = {
        'Datetime': '2017-01-01 00:00:00', 'Temperature': 6.559, 'Humidity': 73.8,
        'WindSpeed': 0.083, 'GeneralDiffuseFlows': 0.051, 'DiffuseFlows': 0.119
    }
    generated_power_1 = generate_power_consumption(
        trained_model, example_input_data_1, scaler_cond_continuous, scaler_target,
        config.CONTINUOUS_CONDITIONAL_FEATURES, config.CATEGORICAL_CONDITIONAL_FEATURES, config.DEVICE
    )
    print("\n生成的电力消耗 (示例 1 - 新年午夜):")
    for zone, value in generated_power_1.items():
        print(f"  {zone}: {value:.2f}")

    example_input_data_2 = {
        'Datetime': '2017-07-15 14:30:00', 'Temperature': 35.0, 'Humidity': 60.0,
        'WindSpeed': 2.5, 'GeneralDiffuseFlows': 800.0, 'DiffuseFlows': 250.0
    }
    generated_power_2 = generate_power_consumption(
        trained_model, example_input_data_2, scaler_cond_continuous, scaler_target,
        config.CONTINUOUS_CONDITIONAL_FEATURES, config.CATEGORICAL_CONDITIONAL_FEATURES, config.DEVICE
    )
    print("\n生成的电力消耗 (示例 2 - 夏季下午):")
    for zone, value in generated_power_2.items():
        print(f"  {zone}: {value:.2f}")

    # 8. 批量生成并可视化
    print("\n--- 批量生成并添加到DataFrame ---")
    num_rows_to_predict_batch = 5000
    predictions_zone1, predictions_zone2, predictions_zone3 = [], [], []

    input_cols_for_generation = [
        'Datetime', 'Temperature', 'Humidity', 'WindSpeed',
        'GeneralDiffuseFlows', 'DiffuseFlows'
    ]

    for i in tqdm(range(num_rows_to_predict_batch), desc="批量生成预测"):
        current_input_data = original_df.iloc[i][input_cols_for_generation].to_dict()
        generated_power = generate_power_consumption(
            trained_model, current_input_data, scaler_cond_continuous, scaler_target,
            config.CONTINUOUS_CONDITIONAL_FEATURES, config.CATEGORICAL_CONDITIONAL_FEATURES, config.DEVICE
        )
        predictions_zone1.append(generated_power['PowerConsumption_Zone1'])
        predictions_zone2.append(generated_power['PowerConsumption_Zone2'])
        predictions_zone3.append(generated_power['PowerConsumption_Zone3'])

    df_comparison = original_df.iloc[:num_rows_to_predict_batch].copy()
    df_comparison['PowerConsumption_Zone1_Predict'] = predictions_zone1
    df_comparison['PowerConsumption_Zone2_Predict'] = predictions_zone2
    df_comparison['PowerConsumption_Zone3_Predict'] = predictions_zone3

    print(f"成功将预测结果添加到 df 中。")
    print(df_comparison[['Datetime'] + config.TARGET_FEATURES + [col for col in df_comparison.columns if 'Predict' in col]].head())

    plot_time_series_comparison(df_comparison, 2000, config.TARGET_FEATURES)
    print("--- 所有任务完成 ---")


if __name__ == "__main__":
    main()
