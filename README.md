# Energy Forecasting CVAE

按公寓分别训练用电量预测模型，使用天气、时间、历史用电量及滞后统计特征。

## 代码结构

| 文件 | 用途 |
| --- | --- |
| `main.py` | 主训练入口 |
| `src/hier_config.py` | 数据路径、特征和训练参数 |
| `src/enhanced_data_utils.py` | 预处理、时间窗口和 DataLoader |
| `src/enhanced_model.py` | 模型、注意力和损失函数 |
| `src/enhanced_trainer.py` | 训练、预测、指标和模型保存 |
| `src/enhanced_visualization.py` | 共享结果读取与绘图 |
| `data_process.py` | 合并原始用电量与天气数据 |
| `slice_power_weather.py` | 按时间范围提取数据 |
| `visualize_apartments_1_4.py` | 公寓 1–4 的详细图表，默认读取 v3 结果 |
| `main_hier_enhanced.py` | 旧入口兼容层，复用 `src/` 实现 |
| `other_versions/` | 保留网络结构不同的 v2、v3 实验 |

## 运行

在项目根目录、已安装依赖的 Python 环境中执行：

```bash
git clone https://github.com/gaigai2002/EnergyForcasting-CVAE.git
cd EnergyForcasting-CVAE
python -m pip install -r requirements.txt
python main.py
```

训练默认读取 `data/slice_power_weather.csv`，模型和缩放器写入
`model_artifacts/enhanced_hier_lstm/`，指标和预测写入
`outputs/enhanced_per_apartment_metrics.csv` 和
`outputs/enhanced_per_apartment_predictions.csv`。

需要从原始数据重新生成输入时执行：

```bash
python data_process.py
python slice_power_weather.py
```

前者读取 `data/2016/Apt*_2016.csv` 和 `data/apartment2016.csv`，生成
`data/power_weather.csv`；后者默认提取 2016-07-01 至 2016-07-07。
仓库包含训练用的周数据；完整合并数据 `data/power_weather.csv` 在本地生成，
不上传到 GitHub。

生成主模型诊断图：

```bash
python -m src.enhanced_visualization
```

## 实验版本

```bash
python -m other_versions.main_hier_enhanced_v2
python -m other_versions.main_hier_enhanced_v3
```

v2 使用单向 LSTM 和精简特征；v3 使用短窗口、局部注意力及条件残差门控。
它们保留各自的模型和结果目录，共用相同的预处理及损失函数。

`python main_hier_enhanced.py` 保留旧脚本的 100 轮上限和 30 轮早停耐心；
`python main.py` 使用 `src/hier_config.py`，默认早停耐心为 20 轮。
旧脚本中的模型、数据和预测函数仍可导入。

图表中的残差带由观察到的预测误差计算，累计 MSE 图描述测试样本误差，
不代表训练曲线。缓存、模型和输出目录由 `.gitignore` 排除；现有数据和模型文件保留。
