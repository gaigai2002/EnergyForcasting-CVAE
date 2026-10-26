# Dataset Description: Multi-Apartment Energy Consumption with Weather Data

## 1. Overview

This dataset contains minute-level energy consumption measurements from 114 residential apartments, combined with corresponding weather conditions. The dataset is designed for time series forecasting tasks, specifically for predicting future energy consumption based on historical consumption patterns and meteorological conditions.

**Dataset Name**: Multi-Apartment Energy Consumption Dataset (2016)  
**Time Resolution**: 1 minute  
**Time Period**: July 1, 2016 to December 15, 2016, 19:59:00  
**Total Apartments**: 114  
**File Format**: CSV  
**File Path**: `./data/slice_power_weather.csv`

---

## 2. Data Collection

### 2.1 Data Sources

The dataset is constructed by merging two primary data sources:

1. **Energy Consumption Data**: Individual CSV files for each apartment (`Apt1_2016.csv` to `Apt114_2016.csv`), containing timestamped power consumption measurements.

2. **Weather Data**: Meteorological data from `apartment2016.csv`, providing weather conditions at corresponding timestamps.

### 2.2 Data Preprocessing

- **Temporal Alignment**: All apartment consumption data and weather data are aligned to a uniform 1-minute resolution time index.
- **Missing Value Handling**: Missing values are filled with 0 for all columns.
- **Time Range Intersection**: The final dataset uses the intersection of time ranges from both energy and weather data sources.

---

## 3. Dataset Structure

### 3.1 Columns

The dataset contains the following columns:

#### **Timestamp Column**
- **`Datetime`**: Timestamp in `YYYY-MM-DD HH:MM:SS` format (1-minute resolution)

#### **Target Variables (114 columns)**
- **`Apt_1`, `Apt_2`, ..., `Apt_114`**: Energy consumption (in kWh) for each apartment at each timestamp
  - **Data Type**: Float32
  - **Range**: Non-negative values (typically 0-10+ kWh)
  - **Missing Values**: Filled with 0

#### **Continuous Weather Features (10 columns)**
- **`temperature`**: Air temperature (°C)
- **`humidity`**: Relative humidity (%)
- **`visibility`**: Visibility distance (km)
- **`apparentTemperature`**: Apparent/feels-like temperature (°C)
- **`pressure`**: Atmospheric pressure (hPa)
- **`windSpeed`**: Wind speed (m/s)
- **`windBearing`**: Wind direction (degrees, 0-360)
- **`precipIntensity`**: Precipitation intensity (mm/h)
- **`dewPoint`**: Dew point temperature (°C)
- **`precipProbability`**: Probability of precipitation (0-1)

**Note**: The original dataset may have included `cloudCover`, which was removed during preprocessing.

#### **Derived Temporal Features (2 columns)**
These features are extracted from the `Datetime` column during preprocessing:
- **`Hour`**: Hour of day (0-23), extracted from timestamp
- **`DayOfWeek`**: Day of week (0-6, where 0=Monday, 6=Sunday), extracted from timestamp

---

## 4. Dataset Statistics

### 4.1 Temporal Coverage

- **Start Date**: July 1, 2016, 00:00:00
- **End Date**: December 15, 2016, 19:59:00
- **Duration**: Approximately 5.5 months
- **Total Time Points**: ~237,600 minutes (assuming continuous 1-minute intervals)
- **Time Resolution**: 1 minute

### 4.2 Data Volume

- **Total Columns**: 127 (1 timestamp + 114 apartments + 10 weather + 2 temporal features)
- **Total Rows**: Approximately 237,600 (depending on exact time range)
- **Data Size**: ~200-300 MB (estimated)

### 4.3 Data Distribution

#### **Energy Consumption**
- **Range**: Varies significantly across apartments (0 to 10+ kWh per minute)
- **Characteristics**: 
  - Some apartments show high consumption (e.g., Apt_5: ~4.5 kWh/min)
  - Some apartments show very low consumption (e.g., Apt_3, Apt_4: ~0.008 kWh/min)
  - Consumption patterns exhibit daily and weekly cycles

#### **Weather Features**
- **Normalization**: All weather features are normalized to [0, 1] range using MinMaxScaler during preprocessing
- **Missing Values**: Filled with 0 (or mean for continuous features in some preprocessing versions)

---

## 5. Data Splitting Strategy

The dataset is split chronologically (time-ordered) to prevent data leakage:

- **Training Set**: 60% (earliest 60% of time points)
- **Validation Set**: 20% (next 20% of time points)
- **Test Set**: 20% (latest 20% of time points)

**Important**: The split is performed **after** temporal alignment and preprocessing, ensuring that:
- Training data comes from earlier time periods
- Validation and test data come from later time periods
- No future information leaks into training

---

## 6. Feature Engineering

### 6.1 Lag Features

For each apartment, lag features are computed to capture short-term and medium-term dependencies:

- **`lag1`**: Consumption at t-1 (1 minute ago)
- **`lag2`**: Consumption at t-2 (2 minutes ago)
- **`lag3`**: Consumption at t-3 (3 minutes ago)
- **`lag6`**: Consumption at t-6 (6 minutes ago)
- **`lag12`**: Consumption at t-12 (12 minutes ago)
- **`lag24`**: Consumption at t-24 (24 minutes ago)

**Purpose**: Explicitly provide historical values to help the model learn temporal dependencies.

### 6.2 Difference Features

First-order and second-order differences are computed to capture trend changes:

- **`diff1`**: First-order difference = `y[t-1] - y[t-2]` (rate of change)
- **`diff2`**: Second-order difference = `diff1[t-1] - diff1[t-2]` (acceleration)

**Purpose**: Capture trend changes and make the model more sensitive to sudden changes.

### 6.3 Rolling Statistics

Rolling window statistics are computed using a lookback window (typically 24 minutes):

- **`rolling_mean`**: Mean consumption over the past 24 minutes
- **`rolling_std`**: Standard deviation of consumption over the past 24 minutes

**Purpose**: Provide local statistical information and capture volatility.

### 6.4 Temporal Window Features

For each time point, a sliding window of the past N steps (typically 24 steps) is used, containing:
- Historical continuous features (weather)
- Historical categorical features (Hour, DayOfWeek, normalized)
- Historical target values (consumption)

**Purpose**: Provide temporal context for sequence modeling.

---

## 7. Data Preprocessing Pipeline

### 7.1 Normalization

All features and targets are normalized to [0, 1] range:

1. **Continuous Features**: MinMaxScaler applied to weather features
   - **Fit**: Only on training set
   - **Transform**: Applied to train, validation, and test sets

2. **Target Variables**: MinMaxScaler applied to each apartment's consumption
   - **Fit**: Only on training set
   - **Transform**: Applied to train, validation, and test sets

3. **Categorical Features**: Manual normalization
   - **Hour**: Divided by 23.0 (range: 0-1)
   - **DayOfWeek**: Divided by 6.0 (range: 0-1)

### 7.2 Clipping

After normalization, all values are clipped to [0, 1] to handle any outliers or numerical errors.

### 7.3 Missing Value Imputation

- **Target Variables**: Filled with 0
- **Continuous Features**: Filled with mean (computed on training set)

---

## 8. Dataset Characteristics

### 8.1 Temporal Patterns

- **Daily Cycles**: Energy consumption exhibits strong daily patterns (higher during day, lower at night)
- **Weekly Cycles**: Consumption patterns vary by day of week (weekdays vs. weekends)
- **Seasonal Trends**: Data spans summer to early winter (July to December), showing potential seasonal variations

### 8.2 Apartment Heterogeneity

- **High Variability**: Consumption levels vary significantly across apartments (orders of magnitude difference)
- **Individual Patterns**: Each apartment has unique consumption patterns
- **Modeling Strategy**: Each apartment is modeled separately to capture individual characteristics

### 8.3 Weather Dependencies

- **Temperature Correlation**: Energy consumption likely correlates with temperature (heating/cooling)
- **Weather Impact**: Multiple weather features provide rich context for consumption prediction

---

## 9. Usage in Models

### 9.1 Input Features

For each time point, the model receives:

1. **Temporal Window** (`x_window`): 
   - Shape: `(batch_size, 24, 14)` → flattened to `(batch_size, 336)`
   - Contains: 24 time steps × [11 weather + 2 categorical + 1 target]

2. **Current Continuous Features** (`c_cont`):
   - Shape: `(batch_size, 11)`
   - Contains: Current weather conditions

3. **Current Categorical Features** (`c_cat`):
   - Shape: `(batch_size, 2)`
   - Contains: Hour, DayOfWeek

4. **Lag Features** (`lag_feats`):
   - Shape: `(batch_size, 6)`
   - Contains: [lag1, lag2, lag3, lag6, lag12, lag24]

5. **Difference Features** (`diff_feats`):
   - Shape: `(batch_size, 2)`
   - Contains: [diff1, diff2]

6. **Rolling Statistics** (`roll_stats`):
   - Shape: `(batch_size, 2)`
   - Contains: [rolling_mean, rolling_std]

### 9.2 Target Variable

- **Target** (`y`):
  - Shape: `(batch_size, 1)`
  - Contains: Energy consumption at current time point (normalized)

---

## 10. Data Quality Considerations

### 10.1 Potential Issues

1. **Missing Values**: Some time points may have missing consumption data (filled with 0)
2. **Data Gaps**: Potential gaps in time series due to sensor failures or data collection issues
3. **Outliers**: Some apartments may have unusually high or low consumption values
4. **Temporal Alignment**: Weather data and energy data may have slight temporal misalignment

### 10.2 Data Validation

- **Temporal Continuity**: Check for gaps in timestamp sequence
- **Value Ranges**: Verify that consumption values are non-negative and within reasonable ranges
- **Feature Completeness**: Ensure all required features are present

---

## 11. Citation and Usage

### 11.1 Dataset Usage

This dataset is used for:
- **Time Series Forecasting**: Predicting future energy consumption
- **Multi-Apartment Modeling**: Individual models for each apartment
- **Weather-Aware Prediction**: Incorporating meteorological conditions

### 11.2 Model Applications

The dataset supports:
- **Conditional Variational Autoencoder (CVAE)** models
- **Bidirectional LSTM** with attention mechanisms
- **Hierarchical latent space** modeling
- **Feature engineering** with lag, difference, and rolling statistics

---

## 12. File Structure

```
data/
├── slice_power_weather.csv          # Main dataset file (used in models)
├── power_weather.csv                 # Full merged dataset (before slicing)
├── apartment2016.csv                # Original weather data
└── 2016/                            # Original apartment consumption files
    ├── Apt1_2016.csv
    ├── Apt2_2016.csv
    ...
    └── Apt114_2016.csv
```

---

## 13. Summary

This dataset provides a comprehensive multi-apartment energy consumption dataset with:
- **114 apartments** with individual consumption patterns
- **Minute-level resolution** for fine-grained temporal analysis
- **Rich weather features** (10 meteorological variables)
- **5.5 months** of continuous data
- **Time-ordered split** for proper train/validation/test evaluation
- **Extensive feature engineering** (lag, difference, rolling statistics)

The dataset is well-suited for developing and evaluating time series forecasting models that can capture both temporal dependencies and weather-related patterns in residential energy consumption.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Dataset Version**: slice_power_weather.csv (2016)

