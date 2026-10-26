"""Data and training settings for the enhanced per-apartment models."""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data and features
DATA_PATH = "./data/slice_power_weather.csv"
CONTINUOUS_CONDITIONAL_FEATURES = [
    "temperature",
    "humidity",
    "visibility",
    "apparentTemperature",
    "pressure",
    "windSpeed",
    "windBearing",
    "precipIntensity",
    "dewPoint",
    "precipProbability",
]
CATEGORICAL_CONDITIONAL_FEATURES = ["Hour", "DayOfWeek"]
TARGET_FEATURES = [f"Apt_{i}" for i in range(1, 115)]
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
LOOKBACK_STEPS_ENHANCED = 24

# Model
GLOBAL_LATENT_DIM = 64
HIDDEN_DIM = 128
NUM_ATTENTION_HEADS = 8
DROPOUT = 0.1
CAT_DIMS = {"Hour": 24, "DayOfWeek": 7}
EMBEDDING_DIMS = {"Hour": 16, "DayOfWeek": 8}

# Training
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
KL_WARMUP_EPOCHS = 30
BETA_GLOBAL = 0.0001
EARLY_STOP_PATIENCE = 20
GRADIENT_CLIP = 1.0
MODEL_DIR = "model_artifacts"
