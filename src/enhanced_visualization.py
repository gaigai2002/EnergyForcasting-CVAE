"""Shared CSV loading and diagnostic plots for per-apartment predictions."""

from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure fonts to be safe on Linux environments
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_prediction_arrays(predictions_file, target_features=None):
    """Read matching actual/predicted columns and discard nonfinite rows."""
    df = pd.read_csv(predictions_file)
    if target_features is None:
        target_features = [
            col for col in df.columns
            if col.startswith("Apt_") and not col.endswith("_Predict")
        ]
    target_features = list(target_features)
    if not target_features:
        raise ValueError("No apartment columns found in predictions file.")

    predicted_columns = [f"{name}_Predict" for name in target_features]
    missing = [
        col for col in target_features + predicted_columns if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing prediction columns: {', '.join(missing)}")

    actuals = df[target_features].to_numpy(dtype=float)
    predictions = df[predicted_columns].to_numpy(dtype=float)
    valid = np.isfinite(actuals).all(axis=1) & np.isfinite(predictions).all(axis=1)
    if not valid.any():
        raise ValueError("No valid prediction samples remain after filtering.")
    return actuals[valid], predictions[valid], target_features


def _save_figure(fig, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


def _apartment_axes(target_features, apartments_to_show, row_height):
    if apartments_to_show is None:
        apartments_to_show = [1, 10, 50, 100]
    indices = list(dict.fromkeys(
        apt - 1 for apt in apartments_to_show if 0 < apt <= len(target_features)
    ))
    if not indices:
        raise ValueError("No selected apartments are present in the predictions.")
    cols = min(2, len(indices))
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, row_height * rows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(indices):]:
        ax.set_visible(False)
    return fig, axes, indices


def plot_prediction_time_series(
    actuals,
    predictions,
    target_features,
    num_points=200,
    save_path="prediction_time_series.png",
):
    """Compare actual and predicted values over the first test time steps."""
    num_points = min(num_points, len(actuals))
    if not target_features or num_points <= 0:
        raise ValueError("Time-series plots require apartments and samples.")
    fig, axes = plt.subplots(
        len(target_features), 1, figsize=(16, 4 * len(target_features)),
        sharex=True, squeeze=False,
    )
    for i, (name, ax) in enumerate(zip(target_features, axes.ravel())):
        ax.plot(actuals[:num_points, i], label="Actual", linewidth=1.5, alpha=0.8)
        ax.plot(
            predictions[:num_points, i], label="Predicted", linestyle="--",
            linewidth=1.5, alpha=0.8,
        )
        ax.set_title(f"{name} Prediction vs Actual")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Time Step")
    _save_figure(fig, save_path)


def plot_prediction_scatter(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_features: List[str],
    apartments_to_show: Optional[List[int]] = None,
    save_path: str = "hier_scatter.png",
) -> None:
    """Scatter plot of predicted vs actual values for selected apartments."""
    fig, axes, apt_indices = _apartment_axes(
        target_features, apartments_to_show, row_height=5
    )

    for i, apt_idx in enumerate(apt_indices):
        ax = axes[i]
        pred = predictions[:, apt_idx]
        actual = actuals[:, apt_idx]

        ax.scatter(actual, pred, alpha=0.3, s=10, c="blue")

        # Diagonal
        min_val = min(actual.min(), pred.min())
        max_val = max(actual.max(), pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        # R²
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{target_features[apt_idx]} (R²={r2:.3f})", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _save_figure(fig, save_path)


def plot_error_distribution(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_features: List[str],
    apartments_to_show: Optional[List[int]] = None,
    save_path: str = "hier_error_dist.png",
) -> None:
    """Histogram of prediction errors for selected apartments."""
    fig, axes, apt_indices = _apartment_axes(
        target_features, apartments_to_show, row_height=4
    )

    for i, apt_idx in enumerate(apt_indices):
        ax = axes[i]
        pred = predictions[:, apt_idx]
        actual = actuals[:, apt_idx]
        errors = pred - actual

        ax.hist(
            errors,
            bins=50,
            density=True,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax.axvline(
            x=errors.mean(),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {errors.mean():.3f}",
        )

        ax.set_xlabel("Prediction Error", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{target_features[apt_idx]} Error Distribution", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _save_figure(fig, save_path)


def plot_metrics_heatmap(
    metrics_df: pd.DataFrame, save_path: str = "hier_metrics_heatmap.png"
) -> None:
    """Heatmap of metrics (MSE/RMSE/MAE/MAPE/R2) over apartments."""
    available_metrics = ["MSE", "RMSE", "MAE", "MAPE", "R2"]
    metrics_cols = [col for col in available_metrics if col in metrics_df.columns]

    if not metrics_cols or metrics_df.empty:
        print("Warning: no usable metric columns found in metrics file.")
        return

    num_metrics = len(metrics_cols)
    if num_metrics <= 2:
        rows, cols = 1, num_metrics
    elif num_metrics <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3

    fig, axes = plt.subplots(rows, cols, figsize=(16, 12), squeeze=False)
    axes = axes.ravel()

    num_apartments = len(metrics_df)
    matrix_rows = 10
    matrix_cols = (num_apartments + matrix_rows - 1) // matrix_rows

    for metric, ax in zip(metrics_cols, axes):
        values = metrics_df[metric].values
        padded = np.full(matrix_rows * matrix_cols, np.nan)
        padded[: len(values)] = values
        matrix = padded.reshape(matrix_rows, matrix_cols)

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_title(f"{metric} by Apartment", fontsize=12)
        ax.set_xlabel("Column", fontsize=10)
        ax.set_ylabel("Row", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    _save_figure(fig, save_path)


def plot_metrics_ranking(
    metrics_df: pd.DataFrame,
    top_k: int = 20,
    save_path: str = "hier_metrics_ranking.png",
) -> None:
    """Bar plots for best and worst apartments ranked by MSE."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    rankings = [
        (metrics_df.nsmallest(top_k, "MSE"), "green", "Best", "Lowest"),
        (metrics_df.nlargest(top_k, "MSE"), "red", "Worst", "Highest"),
    ]
    for ax, (ranked, color, quality, direction) in zip(axes, rankings):
        y_pos = np.arange(len(ranked))
        ax.barh(y_pos, ranked["MSE"], color=color, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranked["Apartment"])
        ax.set_xlabel("MSE", fontsize=11)
        ax.set_title(f"Top {top_k} {quality} Apartments ({direction} MSE)", fontsize=12)
        ax.invert_yaxis()
    _save_figure(fig, save_path)


def visualize_from_predictions_file(
    predictions_file: str = "outputs/enhanced_per_apartment_predictions.csv",
    metrics_file: str = "outputs/enhanced_per_apartment_metrics.csv",
    output_dir: str = "outputs/visualizations",
) -> None:
    """
    High-level entry point:
    Read prediction and metrics CSVs and generate all standard plots.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not Path(predictions_file).exists():
        print(f"Error: predictions file not found: {predictions_file}")
        print("Please run the training script to generate predictions first.")
        return

    if not Path(metrics_file).exists():
        print(f"Warning: metrics file not found: {metrics_file}")
        print("Will generate plots based only on predictions.")
        metrics_df = None
    else:
        metrics_df = pd.read_csv(metrics_file)
        print(f"Loaded metrics file: {metrics_file}")

    print(f"Reading predictions file: {predictions_file}")
    try:
        actuals, predictions, target_features = load_prediction_arrays(predictions_file)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print(f"Effective samples: {len(actuals)}")
    sample_apts = [1, 10, min(50, len(target_features)), min(100, len(target_features))]
    for plot, filename in [
        (plot_prediction_scatter, "prediction_scatter.png"),
        (plot_error_distribution, "error_distribution.png"),
    ]:
        plot(
            predictions,
            actuals,
            target_features,
            apartments_to_show=sample_apts,
            save_path=str(output_dir_path / filename),
        )

    if metrics_df is not None and not metrics_df.empty and "MSE" in metrics_df.columns:
        plot_metrics_heatmap(
            metrics_df,
            save_path=str(output_dir_path / "metrics_heatmap.png"),
        )
        plot_metrics_ranking(
            metrics_df,
            top_k=20,
            save_path=str(output_dir_path / "metrics_ranking.png"),
        )

    print(f"\nAll visualization figures saved to: {output_dir_path}")


if __name__ == "__main__":
    visualize_from_predictions_file()
