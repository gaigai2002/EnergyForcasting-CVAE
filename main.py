"""
Entry point for the enhanced per-apartment HierCVAE training pipeline.

This script is intentionally minimal: it delegates all heavy lifting to
the refactored modules under `src/`:
- `src.hier_config` for configuration
- `src.enhanced_data_utils` for data preparation
- `src.enhanced_model` for model definitions
- `src.enhanced_trainer` for training and evaluation
- `src.enhanced_visualization` for post-hoc analysis (optional)

`main_hier_enhanced.py` remains a compatibility entry point with its
original early-stopping patience of 30; this entry uses `src.hier_config`.
"""

from src import hier_config as config
from src.enhanced_trainer import train_all_apartments


def main() -> None:
    """Train one enhanced model per apartment and save results."""
    train_all_apartments(config)


if __name__ == "__main__":
    main()
