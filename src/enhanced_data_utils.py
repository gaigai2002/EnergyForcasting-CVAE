"""
Enhanced time series dataset and data loading utilities for the
per-apartment EnhancedHierCVAE model.

- Builds rich time-window features (continuous, categorical, target history)
- Adds lag features, first/second-order differences, and rolling statistics
- Provides train/val/test DataLoaders.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class EnhancedTimeSeriesDataset(Dataset):
    """
    Enhanced time series dataset for single-apartment forecasting.

    Core ideas:
    1. Time window: use previous N steps to predict the next step.
    2. Lag features: explicit historical values (lags 1, 2, 3, 6, 12, 24).
    3. Difference features: first and second order differences for trends.
    4. Rolling statistics: moving mean and std over the target window.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        continuous_features: List[str],
        categorical_features: List[str],
        target_col: str,
        lookback_steps: int,
        lag_features: List[int] = None,
    ):
        super().__init__()
        if lag_features is None:
            lag_features = [1, 2, 3, 6, 12, 24]

        self.df = df.reset_index(drop=True)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.lookback_steps = lookback_steps
        self.lag_features = lag_features
        self.max_lag = max(lag_features) if lag_features else 0

        # Precompute arrays
        self.cont_data = self.df[continuous_features].values.astype(np.float32)
        self.cat_data = self.df[categorical_features].values.astype(np.int64)
        self.target_data = self.df[target_col].values.astype(np.float32)

        # Lag features (on normalized target)
        self.lag_data = np.zeros((len(df), len(lag_features)), dtype=np.float32)
        for i, lag in enumerate(lag_features):
            self.lag_data[lag:, i] = self.target_data[:-lag]

        # Difference features: first and second order
        self.diff_data = np.zeros((len(df), 2), dtype=np.float32)
        # First order diff: y_{t-1} - y_{t-2}
        self.diff_data[2:, 0] = self.target_data[1:-1] - self.target_data[:-2]
        # Second order diff: diff_{t-1} - diff_{t-2}
        self.diff_data[3:, 1] = self.diff_data[2:-1, 0] - self.diff_data[1:-2, 0]

        # Rolling statistics over the target
        self.rolling_mean = np.zeros(len(df), dtype=np.float32)
        self.rolling_std = np.zeros(len(df), dtype=np.float32)
        for i in range(lookback_steps, len(df)):
            window = self.target_data[i - lookback_steps : i]
            self.rolling_mean[i] = window.mean()
            std = window.std()
            self.rolling_std[i] = std if std > 0 else 0.01

        # Valid indices need enough history for both window and lags
        min_history = max(lookback_steps, self.max_lag)
        self.valid_indices = list(range(min_history, len(self.df)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        actual_idx = self.valid_indices[idx]
        window_start = actual_idx - self.lookback_steps
        window_end = actual_idx

        # Window features
        window_cont = self.cont_data[window_start:window_end]
        window_cat = self.cat_data[window_start:window_end]
        window_target = self.target_data[window_start:window_end].reshape(-1, 1)

        # Normalize categorical features to [0,1] (Hour, DayOfWeek)
        window_cat_norm = window_cat.astype(np.float32)
        # Hour in [0,23]
        window_cat_norm[:, 0] = window_cat_norm[:, 0] / 23.0
        # DayOfWeek in [0,6]
        window_cat_norm[:, 1] = window_cat_norm[:, 1] / 6.0

        # Concatenate: cont + normalized cat + target history
        window_features = np.concatenate(
            [window_cont, window_cat_norm, window_target], axis=1
        )
        window_features = window_features.flatten().astype(np.float32)

        # Current time-step conditional features
        c_cont = self.cont_data[actual_idx]
        c_cat = self.cat_data[actual_idx]

        # Lag features
        lag_feats = self.lag_data[actual_idx]

        # Difference features
        diff_feats = self.diff_data[actual_idx]

        # Rolling statistics
        roll_stats = np.array(
            [self.rolling_mean[actual_idx], self.rolling_std[actual_idx]],
            dtype=np.float32,
        )

        # Target value (single apartment)
        y = np.array([self.target_data[actual_idx]], dtype=np.float32)

        return (
            torch.tensor(window_features),
            torch.tensor(c_cont),
            torch.tensor(c_cat),
            torch.tensor(lag_feats),
            torch.tensor(diff_feats),
            torch.tensor(roll_stats),
            torch.tensor(y),
        )


def load_and_preprocess_enhanced(
    data_path: str,
    continuous_features: List[str],
    categorical_features: List[str],
    target_features: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:
    """
    Load and preprocess raw data for the enhanced per-apartment pipeline.

    Steps:
    1. Read CSV and sort by time.
    2. Extract time features (Hour, DayOfWeek).
    3. Fill missing values.
    4. Split into train/val/test.
    5. Fit MinMaxScaler on train and transform all splits.
    6. Clip values to [0, 1] for numerical stability.
    """
    df = pd.read_csv(data_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek

    # Fill missing values
    df[target_features] = df[target_features].fillna(0)
    df[continuous_features] = df[continuous_features].fillna(
        df[continuous_features].mean()
    )

    # Split into train / val / test
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Fit on training data only, then apply the same transform to each split.
    scaler_cond = MinMaxScaler().fit(train_df[continuous_features])
    scaler_target = MinMaxScaler().fit(train_df[target_features])
    for split in (train_df, val_df, test_df):
        for features, scaler in (
            (continuous_features, scaler_cond), (target_features, scaler_target)
        ):
            split[features] = scaler.transform(split[features]).clip(0, 1)

    return train_df, val_df, test_df, scaler_cond, scaler_target


def create_dataloaders_enhanced(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    continuous_features: List[str],
    categorical_features: List[str],
    target_col: str,
    batch_size: int,
    lookback_steps: int,
    lag_features: List[int],
):
    """Build identical feature datasets; shuffle only the training split."""
    return tuple(
        DataLoader(
            EnhancedTimeSeriesDataset(
                split, continuous_features, categorical_features,
                target_col, lookback_steps, lag_features,
            ),
            batch_size=batch_size,
            shuffle=is_training,
            pin_memory=True,
        )
        for split, is_training in (
            (train_df, True), (val_df, False), (test_df, False)
        )
    )
