"""Plot predictions, residual bands, densities, and cumulative MSE for Apt_1–4."""

from pathlib import Path

# Load the shared module first so it selects the noninteractive plotting backend.
from src.enhanced_visualization import load_prediction_arrays, plot_prediction_time_series

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

APARTMENTS_TO_PLOT = ['Apt_1', 'Apt_2', 'Apt_3', 'Apt_4']
NUM_POINTS_TO_PLOT = 200
OUTPUT_DIR = Path("outputs/visualizations")


def _save_plot(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def load_prediction_data(predictions_file="outputs/enhanced_per_apartment_predictions_v3.csv"):
    """Return actuals, predictions, and apartment names, filtering invalid rows."""
    try:
        actuals, predictions, target_features = load_prediction_arrays(
            predictions_file, APARTMENTS_TO_PLOT
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return None, None, None
    print(f"Data loaded: {len(actuals)} valid samples")
    print(f"Apartments: {target_features}")
    return actuals, predictions, target_features


def plot_prediction_vs_actual(actuals, predictions, target_features, num_points_to_plot=200):
    """Compare actual and predicted values using the shared time-series plot."""
    plot_prediction_time_series(
        actuals, predictions, target_features, num_points_to_plot,
        save_path=OUTPUT_DIR / "1_prediction_vs_actual.png",
    )


def plot_prediction_with_confidence_interval(
    actuals, predictions, target_features, num_points_to_plot=200, num_samples_for_ci=20,
):
    """Plot a descriptive ±2 local residual-standard-deviation band.

    This uses observed errors, not a calibrated predictive confidence interval.
    The legacy name and unused num_samples_for_ci argument are kept for callers.
    """
    num_points = min(num_points_to_plot, len(actuals))
    plot_count = len(target_features)
    errors = predictions[:num_points] - actuals[:num_points]
    window_size = min(10, num_points // 10)
    uncertainties = np.empty_like(errors, dtype=float)
    for t in range(num_points):
        start = max(0, t - window_size // 2)
        end = min(num_points, t + window_size // 2 + 1)
        uncertainties[t] = errors[start:end].std(axis=0)

    mean_predictions = predictions[:num_points]
    lower_bound = mean_predictions - 2 * uncertainties
    upper_bound = mean_predictions + 2 * uncertainties
    plt.figure(figsize=(18, 4 * plot_count))
    for i, zone in enumerate(target_features):
        plt.subplot(plot_count, 1, i + 1)
        plt.plot(
            actuals[:num_points, i], label=f'Actual {zone}', color='blue',
            linewidth=1.5, alpha=0.8,
        )
        plt.plot(
            mean_predictions[:, i], label=f'Predicted {zone}', color='red',
            linewidth=1.5, alpha=0.8,
        )
        plt.fill_between(
            range(num_points), lower_bound[:, i], upper_bound[:, i],
            color='red', alpha=0.2, label='±2 local residual standard deviations',
        )
        plt.title(f'First {num_points} Test Points: {zone} Prediction and Residual Band')
        plt.xlabel('Time Step')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
    _save_plot("2_prediction_with_ci.png")


def plot_generated_vs_actual_density(actuals, predictions, target_features, max_plots=4):
    """Compare actual and predicted distributions using kernel density estimates."""
    plot_count = min(len(target_features), max_plots)
    plt.figure(figsize=(4 * plot_count, 4))
    for i, zone in enumerate(target_features[:plot_count]):
        plt.subplot(1, plot_count, i + 1)
        actual = actuals[:, i]
        predicted = predictions[:, i]
        valid = np.isfinite(actual) & np.isfinite(predicted)
        sns.kdeplot(actual[valid], label=f'Actual {zone}', fill=True, alpha=0.6)
        sns.kdeplot(predicted[valid], label=f'Predicted {zone}', fill=True, alpha=0.5)
        plt.legend()
        plt.title(zone)
        plt.xlabel('Power Consumption')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
    _save_plot("3_density_comparison.png")


def plot_time_series_comparison(actuals, predictions, target_features, num_points=200):
    """Preserve the original time-series output using the shared implementation."""
    plot_prediction_time_series(
        actuals, predictions, target_features, num_points,
        save_path=OUTPUT_DIR / "4_time_series_comparison.png",
    )


def plot_cumulative_prediction_mse(actuals, predictions, target_features):
    """Plot cumulative prediction MSE over test samples, not training epochs."""
    squared_errors = (predictions - actuals) ** 2
    counts = np.arange(1, len(squared_errors) + 1)[:, None]
    cumulative_mse = squared_errors.cumsum(axis=0) / counts
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_mse.mean(axis=1), label='Mean MSE (All Apartments)', linewidth=2)
    for i, zone in enumerate(target_features):
        plt.plot(
            cumulative_mse[:, i], linestyle='--', label=f'{zone} MSE',
            linewidth=1.5, alpha=0.7,
        )
    plt.title('Cumulative Prediction MSE on Test Samples')
    plt.xlabel('Test Time Step')
    plt.ylabel('Cumulative Average MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Keep the historical output filename for consumers of this script.
    _save_plot("5_training_mse_simulation.png")


# Compatibility alias; the curve describes prediction errors, not model training.
plot_training_mse_simulation = plot_cumulative_prediction_mse


def main():
    """Generate all visualization plots using the existing output filenames."""
    print("Visualization Tool for Apt_1 to Apt_4")
    actuals, predictions, target_features = load_prediction_data()
    if actuals is None:
        return
    plot_prediction_vs_actual(actuals, predictions, target_features, NUM_POINTS_TO_PLOT)
    plot_prediction_with_confidence_interval(
        actuals, predictions, target_features, NUM_POINTS_TO_PLOT
    )
    plot_generated_vs_actual_density(actuals, predictions, target_features)
    plot_time_series_comparison(actuals, predictions, target_features, NUM_POINTS_TO_PLOT)
    plot_cumulative_prediction_mse(actuals, predictions, target_features)
    print(f"All visualization plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
