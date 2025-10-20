import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm.notebook import tqdm

def plot_training_mse(test_mses_np, target_features):
    """绘制训练过程中各区域在测试集上的MSE变化图。"""
    plt.figure(figsize=(12, 6))
    plt.plot(test_mses_np[:, 0], label=f'{target_features[0]} MSE')
    plt.plot(test_mses_np[:, 1], label=f'{target_features[1]} MSE')
    plt.plot(test_mses_np[:, 2], label=f'{target_features[2]} MSE (Weighted)', linestyle='--')
    plt.title('各区域在测试集上的MSE随Epoch变化')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1_training_mse.png")  # 保存图片
    plt.close()  # 关闭图形

def plot_prediction_vs_actual(all_x_unscaled, all_recon_x_unscaled, target_features, num_points_to_plot):
    """绘制真实值与预测值的对比图。"""
    plt.figure(figsize=(18, 12))

    for i, zone in enumerate(target_features):
        plt.subplot(len(target_features), 1, i + 1)
        plt.plot(all_x_unscaled[:num_points_to_plot, i], label=f'实际 {zone}', color='blue', alpha=0.7)
        plt.plot(all_recon_x_unscaled[:num_points_to_plot, i], label=f'预测 {zone}', color='red', linestyle='--', alpha=0.7)
        plt.title(f'测试集前{num_points_to_plot}个点：{zone} 预测 vs 实际')
        plt.xlabel('时间点')
        plt.ylabel('电量消耗')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("2_prediction_vs_actual.png")  # 保存图片
    plt.close()  # 关闭图形

def plot_prediction_with_confidence_interval(model, X_test_cont, X_test_cat, y_test, scaler_target,
                                             target_features, device, num_samples_for_ci, num_points_to_plot_vis):
    """绘制预测均值与置信区间对比图。"""
    model.eval()

    # 获取需要可视化的测试集数据
    X_test_cont_plot = X_test_cont[:num_points_to_plot_vis].to(device)
    X_test_cat_plot = X_test_cat[:num_points_to_plot_vis].to(device)
    y_test_plot_np = y_test[:num_points_to_plot_vis].cpu().numpy() # 真实值在CPU上，保持原始缩放状态

    # 存储多次采样生成的预测值
    all_generated_samples_scaled = torch.zeros(
        num_points_to_plot_vis, num_samples_for_ci, model.target_dim
    ).to(device)

    print(f"Generating {num_samples_for_ci} samples for each of the first {num_points_to_plot_vis} test points for CI...")

    with torch.no_grad():
        for s in tqdm(range(num_samples_for_ci), desc="Sampling for CI"):
            generated_s_scaled = model.generate(X_test_cont_plot, X_test_cat_plot)
            all_generated_samples_scaled[:, s, :] = generated_s_scaled

    # 移动到CPU并转换为NumPy
    all_generated_samples_scaled_np = all_generated_samples_scaled.cpu().numpy()

    # 反归一化所有数据到原始尺度
    all_generated_samples_unscaled = np.array([
        scaler_target.inverse_transform(all_generated_samples_scaled_np[i, :, :])
        for i in range(num_points_to_plot_vis)
    ])
    y_test_plot_unscaled = scaler_target.inverse_transform(y_test_plot_np)

    # 计算均值预测和置信区间 (例如，95%)
    mean_predictions = np.mean(all_generated_samples_unscaled, axis=1)
    lower_bound = np.percentile(all_generated_samples_unscaled, 2.5, axis=1)
    upper_bound = np.percentile(all_generated_samples_unscaled, 97.5, axis=1)

    # --- 可视化 ---
    plt.figure(figsize=(18, 12))

    for i, zone in enumerate(target_features):
        plt.subplot(len(target_features), 1, i + 1)
        plt.plot(y_test_plot_unscaled[:, i], label=f'实际 {zone}', color='blue', alpha=0.8)
        plt.plot(mean_predictions[:, i], label=f'预测均值 {zone}', color='red', linestyle='-',
                 alpha=0.8)
        plt.fill_between(
            range(num_points_to_plot_vis),
            lower_bound[:, i],
            upper_bound[:, i],
            color='red', alpha=0.2, label='95% 置信区间'
        )
        plt.title(f'测试集前{num_points_to_plot_vis}个点：{zone} 预测均值与95%置信区间 vs 实际')
        plt.xlabel('时间点')
        plt.ylabel('电量消耗')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("3_prediction_with_ci.png")  # 保存图片
    plt.close()  # 关闭图形

def plot_generated_vs_actual_density(all_x_unscaled, all_recon_x_unscaled, target_features):
    """绘制生成数据与真实数据分布的核密度估计图。"""
    plt.figure(figsize=(15, 4))
    for i, zone in enumerate(target_features):
        plt.subplot(1, len(target_features), i + 1)
        sns.kdeplot(all_x_unscaled[:, i], label=f'实际 {zone}', color='blue', fill=True, alpha=0.6)
        sns.kdeplot(all_recon_x_unscaled[:, i], label=f'预测 {zone}', color='red', linestyle='--', fill=True, alpha=0.4)
        plt.title(f'{zone} 预测 vs 实际分布')
        plt.xlabel('电量消耗')
        plt.ylabel('密度')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("4_density_comparison.png")  # 保存图片
    plt.close()  # 关闭图形

def plot_time_series_comparison(df_comparison, num_points, target_features):
    """绘制真实值与生成值的时序对比图。"""
    # 创建子图
    fig, axs = plt.subplots(len(target_features), 1, figsize=(18, 10), sharex=True)

    for i, zone in enumerate(target_features):
        axs[i].plot(df_comparison['Datetime'][:num_points], df_comparison[zone][:num_points], label=f'{zone} Real')
        axs[i].plot(df_comparison['Datetime'][:num_points], df_comparison[f'{zone}_Predict'][:num_points], label=f'{zone} Predict')
        axs[i].set_title(f'{zone} Power Consumption')
        axs[i].legend()
        axs[i].grid(True)

    # 总体设置
    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.savefig("5_time_series_comparison.png")  # 保存图片
    plt.close()  # 关闭图形
