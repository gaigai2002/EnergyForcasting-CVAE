import torch
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# --- 环境设置 ---
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# --- 数据相关配置 ---
DATA_PATH = "./data/powerconsumption.csv"
CONTINUOUS_CONDITIONAL_FEATURES = ['Temperature', 'Humidity', 'WindSpeed', 'GeneralDiffuseFlows', 'DiffuseFlows']
CATEGORICAL_CONDITIONAL_FEATURES = ['Hour', 'DayOfWeek', 'Month']
TARGET_FEATURES = ['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']
TRAIN_TEST_SPLIT_RATIO = 0.8 # 80% train, 20% test

# --- 模型超参数 ---
TARGET_DIM = len(TARGET_FEATURES) # 会在data_utils.py中自动确定，这里先占位
CONT_COND_DIM = len(CONTINUOUS_CONDITIONAL_FEATURES)
LATENT_DIM = 30
CAT_DIMS = {'Hour': 24, 'DayOfWeek': 7, 'Month': 12} # Month in data is 1-12, but in our code we transform to 0-11
EMBEDDING_DIMS = {'Hour': 8, 'DayOfWeek': 4, 'Month': 4}

# --- 训练超参数 ---
EPOCHS = 50
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
BETA = 0.05  # KLD损失的权重 (可调)
ALPHA = 10.0  # MMD损失的权重 (可调)
ZONE_WEIGHTS = torch.tensor([1.0, 1.0, 0.5]).to(DEVICE) # 为Zone3赋予更高权重

# --- 保存路径 ---
MODEL_DIR = "model_artifacts"
MODEL_PATH = f"{MODEL_DIR}/cvae_model.pth"
SCALER_COND_PATH = f"{MODEL_DIR}/scaler_cond_continuous.pkl"
SCALER_TARGET_PATH = f"{MODEL_DIR}/scaler_target.pkl"

# --- 可视化参数 ---
NUM_POINTS_TO_PLOT_TRAINING_PRED = 24 * 6 * 10 # 用于训练阶段的预测对比
NUM_SAMPLES_FOR_CI = 100 # 用于置信区间采样的样本数
NUM_POINTS_TO_PLOT_CI_VIS = 24 * 6 * 2 # 用于置信区间可视化显示的点数
