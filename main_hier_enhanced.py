"""Compatibility entry point for the enhanced per-apartment training pipeline.

New code should import the implementations from ``src``. The original public
names remain available here for existing scripts and notebooks.
"""

from types import SimpleNamespace

from src import hier_config as config
from src.enhanced_data_utils import (
    EnhancedTimeSeriesDataset,
    create_dataloaders_enhanced,
    load_and_preprocess_enhanced,
)
from src.enhanced_model import EnhancedHierCVAE, EnhancedLoss, TemporalAttention
from src.enhanced_trainer import (
    predict_new_data,
    train_all_apartments,
    train_and_predict_single_apartment,
)

__all__ = [
    "EnhancedTimeSeriesDataset",
    "create_dataloaders_enhanced",
    "load_and_preprocess_enhanced",
    "TemporalAttention",
    "EnhancedHierCVAE",
    "EnhancedLoss",
    "train_and_predict_single_apartment",
    "predict_new_data",
    "main",
]


def main() -> None:
    """Preserve the legacy script's settings without mutating shared config."""
    cfg = SimpleNamespace(**{
        name: value for name, value in vars(config).items() if name.isupper()
    })
    cfg.LOOKBACK_STEPS_ENHANCED = 24
    cfg.LAG_FEATURES = [1, 2, 3, 6, 12, 24]
    cfg.EPOCHS_ENHANCED = 100
    cfg.EARLY_STOP_PATIENCE = 30
    train_all_apartments(cfg)


if __name__ == "__main__":
    main()
