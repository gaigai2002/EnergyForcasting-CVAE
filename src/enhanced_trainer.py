"""
Training and inference utilities for the EnhancedHierCVAE model.

Shared implementations used by both training entry points:
- train_and_predict_single_apartment: train one apartment model and evaluate.
- predict_new_data: use a saved model to predict on new data.
- train_all_apartments: end-to-end training loop over all apartments, with
  metrics and predictions saved to disk (used by the new top-level main).

"""

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import hier_config as config
from .enhanced_data_utils import (
    EnhancedTimeSeriesDataset,
    create_dataloaders_enhanced,
    load_and_preprocess_enhanced,
)
from .enhanced_model import EnhancedHierCVAE, EnhancedLoss


def _build_model(cfg, device, lookback_steps, lag_features):
    """Use the same architecture for training and checkpoint inference."""
    return EnhancedHierCVAE(
        cont_cond_dim=len(cfg.CONTINUOUS_CONDITIONAL_FEATURES),
        num_lag_features=len(lag_features),
        num_diff_features=2,
        global_latent_dim=cfg.GLOBAL_LATENT_DIM,
        cat_dims=cfg.CAT_DIMS,
        embedding_dims=cfg.EMBEDDING_DIMS,
        lookback_steps=lookback_steps,
        hidden_dim=cfg.HIDDEN_DIM,
        lstm_layers=2,
        dropout=cfg.DROPOUT,
        num_attention_heads=getattr(cfg, "NUM_ATTENTION_HEADS", 4),
    ).to(device)


def _apartment_scaler(scaler_target, apt_idx):
    """Select one apartment from the fitted multi-target MinMaxScaler."""
    scaler = MinMaxScaler()
    for attr in ("min_", "scale_", "data_min_", "data_max_", "data_range_"):
        setattr(scaler, attr, getattr(scaler_target, attr)[apt_idx : apt_idx + 1])
    scaler.n_features_in_ = 1
    return scaler


def _regression_metrics(apt_name, actuals, predictions):
    """Calculate the metrics shared by normalized and original-scale reports."""
    if predictions.size == 0:
        return dict(Apartment=apt_name, MSE=np.nan, RMSE=np.nan, MAE=np.nan, R2=np.nan)
    mse = mean_squared_error(actuals, predictions)
    return {
        "Apartment": apt_name,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(actuals, predictions),
        "R2": r2_score(actuals, predictions),
    }


def _predict_batches(model, dataloader, device):
    """Collect predictions and targets in dataset order."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in dataloader:
            *features, targets = [tensor.to(device) for tensor in batch]
            outputs = model(*features, targets=None)
            predictions.append(outputs["pred"].cpu())
            actuals.append(targets.cpu())
    if not predictions:
        return np.array([]), np.array([])
    return torch.cat(predictions).numpy(), torch.cat(actuals).numpy()


def train_and_predict_single_apartment(
    apt_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: torch.device,
    cfg=config,
) -> Tuple[Dict, np.ndarray, np.ndarray, EnhancedHierCVAE]:
    """
    Train a single-apartment model and generate test-set predictions.

    Steps:
    1. Build DataLoaders with lag/diff/rolling features.
    2. Initialize model, optimizer, scheduler, and loss.
    3. Run training loop with KL warmup and early stopping.
    4. Reload the best model and evaluate on the test set.
    5. Return per-apartment metrics and (normalized) predictions / targets.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {apt_name}")
    print(f"{'=' * 60}")

    # Configuration
    lookback_steps = getattr(cfg, "LOOKBACK_STEPS_ENHANCED", 24)
    lag_features = getattr(cfg, "LAG_FEATURES", [1, 2, 3, 6, 12, 24])

    train_loader, val_loader, test_loader = create_dataloaders_enhanced(
        train_df,
        val_df,
        test_df,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name,
        cfg.BATCH_SIZE,
        lookback_steps,
        lag_features,
    )

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Lookback: {lookback_steps}, Lag features: {lag_features}")

    # Model
    model = _build_model(cfg, device, lookback_steps, lag_features)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    loss_fn = EnhancedLoss(beta_global=cfg.BETA_GLOBAL)

    # Training loop
    print("  Training...")
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stop_patience = getattr(cfg, "EARLY_STOP_PATIENCE", 30)

    epochs = getattr(cfg, "EPOCHS_ENHANCED", cfg.EPOCHS)

    for epoch in range(epochs):
        model.train()
        warmup_epochs = getattr(cfg, "KL_WARMUP_EPOCHS", 30)
        beta_factor = min(1.0, (epoch + 1) / warmup_epochs)

        for batch in train_loader:
            (
                x_window,
                c_cont,
                c_cat,
                lag_feats,
                diff_feats,
                roll_stats,
                y,
            ) = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(
                x_window, c_cont, c_cat, lag_feats, diff_feats, roll_stats, targets=y
            )
            losses = loss_fn(outputs, y, beta_factor=beta_factor)
            losses["total"].backward()

            if getattr(cfg, "GRADIENT_CLIP", None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (
                    x_window,
                    c_cont,
                    c_cat,
                    lag_feats,
                    diff_feats,
                    roll_stats,
                    y,
                ) = [b.to(device) for b in batch]
                outputs = model(
                    x_window,
                    c_cont,
                    c_cat,
                    lag_feats,
                    diff_feats,
                    roll_stats,
                    targets=y,
                )
                val_loss += loss_fn(outputs, y, beta_factor=1.0)["total"].item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} | Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= early_stop_patience:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)

    # Prediction on test set
    print("  Predicting...")
    predictions, actuals = _predict_batches(model, test_loader, device)
    metrics = _regression_metrics(apt_name, actuals, predictions)
    print(
        f"  Done - MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, "
        f"MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}"
    )

    # Return normalized predictions/targets; caller is responsible for inverse scaling
    return metrics, predictions, actuals, model


def predict_new_data(
    apt_name: str,
    new_df: pd.DataFrame,
    model_dir: str,
    device: torch.device,
    cfg=config,
    historical_buffer: pd.DataFrame = None,
    lookback_steps: int = 24,
    lag_features: List[int] = None,
) -> np.ndarray:
    """
    Predict on new data using a trained per-apartment model.

    The logic matches the original `predict_new_data` implementation.
    """
    if lag_features is None:
        lag_features = [1, 2, 3, 6, 12, 24]

    # Load model path
    model_path = os.path.join(model_dir, f"{apt_name}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load scalers
    scaler_cond_path = os.path.join(model_dir, "scaler_cond.pkl")
    scaler_target_path = os.path.join(model_dir, "scaler_target.pkl")

    if not (os.path.exists(scaler_cond_path) and os.path.exists(scaler_target_path)):
        raise FileNotFoundError(f"Scalers not found in {model_dir}")

    scaler_cond = joblib.load(scaler_cond_path)
    scaler_target = joblib.load(scaler_target_path)

    # Prepare new data
    df = new_df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek

    # Fill missing values
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = df[
        cfg.CONTINUOUS_CONDITIONAL_FEATURES
    ].fillna(df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].mean())
    if apt_name in df.columns:
        df[apt_name] = df[apt_name].fillna(0)

    # Normalize continuous features
    df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = scaler_cond.transform(
        df[cfg.CONTINUOUS_CONDITIONAL_FEATURES]
    )

    # Optionally prepend historical buffer to build lags
    if historical_buffer is not None:
        hist_df = historical_buffer.copy()
        hist_df["Datetime"] = pd.to_datetime(hist_df["Datetime"])
        hist_df["Hour"] = hist_df["Datetime"].dt.hour
        hist_df["DayOfWeek"] = hist_df["Datetime"].dt.dayofweek

        hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = hist_df[
            cfg.CONTINUOUS_CONDITIONAL_FEATURES
        ].fillna(hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES].mean())

        # Normalize historical buffer continuous features
        hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES] = scaler_cond.transform(
            hist_df[cfg.CONTINUOUS_CONDITIONAL_FEATURES]
        )

        # Normalize historical target if present
        if apt_name in hist_df.columns:
            apt_idx = (
                cfg.TARGET_FEATURES.index(apt_name)
                if apt_name in cfg.TARGET_FEATURES
                else 0
            )
            temp_scaler = _apartment_scaler(scaler_target, apt_idx)

            hist_target = hist_df[[apt_name]].values
            hist_target_scaled = temp_scaler.transform(hist_target).flatten()
            hist_df[apt_name] = hist_target_scaled

        combined_df = pd.concat([hist_df, df], ignore_index=True)
    else:
        combined_df = df.copy()
        # If no history, fill/normalize target column if needed
        if apt_name not in combined_df.columns:
            combined_df[apt_name] = 0.0
        else:
            apt_idx = (
                cfg.TARGET_FEATURES.index(apt_name)
                if apt_name in cfg.TARGET_FEATURES
                else 0
            )
            temp_scaler = _apartment_scaler(scaler_target, apt_idx)
            combined_df[apt_name] = temp_scaler.transform(
                combined_df[[apt_name]]
            ).flatten()

    # Build dataset
    dataset = EnhancedTimeSeriesDataset(
        combined_df,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        apt_name,
        lookback_steps,
        lag_features,
    )

    # Compute indices that correspond to new data only
    if historical_buffer is not None:
        hist_len = len(historical_buffer)
        new_indices = [
            i for i in range(len(dataset)) if dataset.valid_indices[i] >= hist_len
        ]
        if len(new_indices) == 0:
            print("Warning: No valid indices for new data prediction")
            return np.array([])
    else:
        new_indices = list(range(len(dataset)))

    new_dataset = torch.utils.data.Subset(dataset, new_indices)
    dataloader = DataLoader(
        new_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # Load model
    model = _build_model(cfg, device, lookback_steps, lag_features)

    model.load_state_dict(torch.load(model_path, map_location=device))
    predictions_scaled, _ = _predict_batches(model, dataloader, device)
    if predictions_scaled.size == 0:
        return np.array([])

    # Inverse transform to original scale
    apt_idx = (
        cfg.TARGET_FEATURES.index(apt_name)
        if apt_name in cfg.TARGET_FEATURES
        else 0
    )
    temp_scaler = _apartment_scaler(scaler_target, apt_idx)

    predictions = temp_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()
    return predictions


def train_all_apartments(cfg=config) -> None:
    """
    End-to-end training procedure for all apartments.

    This function mirrors `main()` in `main_hier_enhanced.py`:
    - Loads and preprocesses data.
    - Trains one model per apartment.
    - Saves models, scalers, metrics and predictions to disk.
    """
    print("=" * 60)
    print("Enhanced HierCVAE+BiLSTM+Attention v2")
    print("Key improvements:")
    print("  - Lag features [1,2,3,6,12,24]")
    print("  - Diff features (1st & 2nd order)")
    print("  - Enhanced residual: w1*lag1 + w2*lag2")
    print("=" * 60)

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    per_apt_model_dir = os.path.join(cfg.MODEL_DIR, "enhanced_hier_lstm")
    os.makedirs(per_apt_model_dir, exist_ok=True)

    # Load and preprocess data
    print("\n[1/3] Loading data...")
    train_df, val_df, test_df, scaler_cond, scaler_target = load_and_preprocess_enhanced(
        cfg.DATA_PATH,
        cfg.CONTINUOUS_CONDITIONAL_FEATURES,
        cfg.CATEGORICAL_CONDITIONAL_FEATURES,
        cfg.TARGET_FEATURES,
        cfg.TRAIN_RATIO,
        cfg.VAL_RATIO,
        cfg.TEST_RATIO,
    )

    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Save scalers for future inference
    scaler_cond_path = os.path.join(per_apt_model_dir, "scaler_cond.pkl")
    scaler_target_path = os.path.join(per_apt_model_dir, "scaler_target.pkl")
    joblib.dump(scaler_cond, scaler_cond_path)
    joblib.dump(scaler_target, scaler_target_path)
    print(f"  Scalers saved to {per_apt_model_dir}")

    # Training loop over apartments
    print("\n[2/3] Training apartments...")
    all_metrics: List[Dict] = []
    all_predictions: Dict[str, np.ndarray] = {}
    all_actuals: Dict[str, np.ndarray] = {}

    for apt_name in tqdm(cfg.TARGET_FEATURES, desc="Progress"):
        try:
            metrics, predictions, actuals, model = train_and_predict_single_apartment(
                apt_name, train_df, val_df, test_df, cfg.DEVICE, cfg
            )

            # Save model
            model_path = os.path.join(per_apt_model_dir, f"{apt_name}_model.pth")
            torch.save(model.state_dict(), model_path)

            # Inverse scale predictions and actuals using the global target scaler
            apt_idx_in_targets = cfg.TARGET_FEATURES.index(apt_name)
            if scaler_target is not None and predictions.size > 0:
                temp_scaler = _apartment_scaler(scaler_target, apt_idx_in_targets)

                predictions_unscaled = temp_scaler.inverse_transform(
                    predictions.reshape(-1, 1)
                ).flatten()
                actuals_unscaled = temp_scaler.inverse_transform(
                    actuals.reshape(-1, 1)
                ).flatten()

                metrics = _regression_metrics(
                    apt_name, actuals_unscaled, predictions_unscaled
                )
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
            all_metrics.append(
                {
                    "Apartment": apt_name,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                }
            )

    # Save results
    print("\n[3/3] Saving results...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join("outputs", "enhanced_per_apartment_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if all_predictions:
        pred_df = pd.DataFrame(
            {apt: pd.Series(values) for apt, values in all_predictions.items()}
        ).add_suffix("_Predict")
        actual_df = pd.DataFrame(
            {apt: pd.Series(values) for apt, values in all_actuals.items()}
        )

        result_df = pd.concat([actual_df, pred_df], axis=1)
        pred_path = os.path.join("outputs", "enhanced_per_apartment_predictions.csv")
        result_df.to_csv(pred_path, index=False)
    else:
        print("No predictions were generated; metrics file may be empty.")

    # Simple summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    valid = metrics_df.dropna(subset=["R2"])
    if len(valid) > 0:
        print(f"\nValid apartments: {len(valid)} / {len(metrics_df)}")
        print(f"Mean R2: {valid['R2'].mean():.4f} (+/- {valid['R2'].std():.4f})")
        print(f"Median R2: {valid['R2'].median():.4f}")
        print(f"Mean RMSE: {valid['RMSE'].mean():.4f}")
        print(f"\nR2 > 0.85: {(valid['R2'] > 0.85).sum()} apartments")
        print(f"R2 > 0.7: {(valid['R2'] > 0.7).sum()} apartments")
        print(f"R2 < 0: {(valid['R2'] < 0).sum()} apartments")

        print(
            f"\nBest: {valid.loc[valid['R2'].idxmax(), 'Apartment']} "
            f"(R2={valid['R2'].max():.4f})"
        )
        print(
            f"Worst: {valid.loc[valid['R2'].idxmin(), 'Apartment']} "
            f"(R2={valid['R2'].min():.4f})"
        )

    print("\nOutputs:")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Models: {per_apt_model_dir}")
    print("=" * 60)
