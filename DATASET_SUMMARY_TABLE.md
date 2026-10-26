# Dataset Summary Table

## Dataset Overview

| **Attribute** | **Value** |
|--------------|-----------|
| **Dataset Name** | Multi-Apartment Energy Consumption Dataset (2016) |
| **File Path** | `./data/slice_power_weather.csv` |
| **Time Resolution** | 1 minute |
| **Time Period** | July 1, 2016 to December 15, 2016, 19:59:00 |
| **Duration** | ~5.5 months |
| **Total Time Points** | ~237,600 minutes |
| **Total Columns** | 127 |
| **Total Rows** | ~237,600 |
| **Data Size** | ~200-300 MB (estimated) |

---

## Data Structure

| **Category** | **Count** | **Column Names** | **Description** |
|-------------|-----------|------------------|-----------------|
| **Timestamp** | 1 | `Datetime` | Timestamp in YYYY-MM-DD HH:MM:SS format |
| **Target Variables** | 114 | `Apt_1` to `Apt_114` | Energy consumption (kWh) for each apartment |
| **Weather Features** | 10 | `temperature`, `humidity`, `visibility`, `apparentTemperature`, `pressure`, `windSpeed`, `windBearing`, `precipIntensity`, `dewPoint`, `precipProbability` | Meteorological conditions |
| **Temporal Features** | 2 | `Hour`, `DayOfWeek` | Extracted from timestamp (Hour: 0-23, DayOfWeek: 0-6) |

---

## Weather Features Details

| **Feature** | **Unit** | **Description** |
|------------|----------|-----------------|
| `temperature` | °C | Air temperature |
| `humidity` | % | Relative humidity |
| `visibility` | km | Visibility distance |
| `apparentTemperature` | °C | Apparent/feels-like temperature |
| `pressure` | hPa | Atmospheric pressure |
| `windSpeed` | m/s | Wind speed |
| `windBearing` | degrees | Wind direction (0-360) |
| `precipIntensity` | mm/h | Precipitation intensity |
| `dewPoint` | °C | Dew point temperature |
| `precipProbability` | 0-1 | Probability of precipitation |

---

## Data Splitting

| **Split** | **Ratio** | **Time Period** | **Purpose** |
|-----------|-----------|-----------------|-------------|
| **Training Set** | 60% | Earliest 60% of time points | Model training |
| **Validation Set** | 20% | Next 20% of time points | Hyperparameter tuning, early stopping |
| **Test Set** | 20% | Latest 20% of time points | Final evaluation |

**Note**: Split is chronological (time-ordered) to prevent data leakage.

---

## Feature Engineering

| **Feature Type** | **Count** | **Lags/Window** | **Description** |
|-----------------|-----------|-----------------|-----------------|
| **Lag Features** | 6 | [1, 2, 3, 6, 12, 24] minutes | Historical consumption values at specific time lags |
| **Difference Features** | 2 | 1st and 2nd order | Rate of change and acceleration |
| **Rolling Statistics** | 2 | 24-minute window | Mean and standard deviation over rolling window |
| **Temporal Window** | 24 steps | Past 24 minutes | Sliding window containing historical features and targets |

---

## Data Preprocessing

| **Step** | **Method** | **Applied To** | **Details** |
|----------|------------|-----------------|-------------|
| **Normalization** | MinMaxScaler | All features and targets | Scale to [0, 1] range |
| **Clipping** | Clip to [0, 1] | All normalized values | Handle outliers |
| **Missing Values** | Fill with 0 or mean | Target and continuous features | Target: 0, Features: mean (computed on training set) |
| **Categorical Normalization** | Manual division | Hour, DayOfWeek | Hour/23.0, DayOfWeek/6.0 |

---

## Model Input Format

| **Input Name** | **Shape** | **Content** |
|---------------|-----------|-------------|
| `x_window` | (batch_size, 336) | Flattened temporal window: 24 steps × [11 weather + 2 categorical + 1 target] |
| `c_cont` | (batch_size, 11) | Current weather features |
| `c_cat` | (batch_size, 2) | Current Hour and DayOfWeek |
| `lag_feats` | (batch_size, 6) | [lag1, lag2, lag3, lag6, lag12, lag24] |
| `diff_feats` | (batch_size, 2) | [diff1, diff2] |
| `roll_stats` | (batch_size, 2) | [rolling_mean, rolling_std] |
| `y` (target) | (batch_size, 1) | Energy consumption at current time point |

---

## Dataset Characteristics

| **Characteristic** | **Description** |
|-------------------|-----------------|
| **Temporal Patterns** | Strong daily cycles (higher during day, lower at night), weekly patterns (weekdays vs. weekends), seasonal trends (summer to early winter) |
| **Apartment Heterogeneity** | High variability across apartments (orders of magnitude difference), individual consumption patterns |
| **Weather Dependencies** | Energy consumption correlates with temperature and weather conditions |
| **Modeling Strategy** | Each apartment modeled separately to capture individual characteristics |

---

## Data Quality

| **Aspect** | **Status** | **Handling** |
|------------|------------|--------------|
| **Missing Values** | Present | Filled with 0 (targets) or mean (features) |
| **Temporal Gaps** | Possible | Checked during preprocessing |
| **Outliers** | Possible | Handled by clipping after normalization |
| **Temporal Alignment** | Aligned | Weather and energy data aligned to 1-minute resolution |

---

## Usage Summary

| **Application** | **Description** |
|----------------|-----------------|
| **Primary Task** | Time series forecasting of energy consumption |
| **Model Types** | Conditional Variational Autoencoder (CVAE), Bidirectional LSTM with Attention |
| **Prediction Horizon** | Next time step (1 minute ahead) |
| **Evaluation Metrics** | MSE, RMSE, MAE, R² |

---

**Document Version**: 1.0  
**Last Updated**: 2024

