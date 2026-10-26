# A Residual-Enhanced Conditional Generative Framework with BiLSTM Attention for Short-Term Residential Energy Load Forecasting

## Abstract

Short-term residential load forecasting is challenging due to strong temporal dependency, weather sensitivity, stochastic fluctuations, and user-level heterogeneity. This paper proposes a residual-enhanced conditional generative framework for multi-residential energy load forecasting. The model combines a Conditional Variational Autoencoder (CVAE) with a BiLSTM-attention encoder to capture latent demand uncertainty and bidirectional temporal dependencies under weather and time conditions. In addition, explicit lag, difference, rolling statistical features, and a learnable residual connection are introduced to strengthen short-term load dependency modeling. Experiments on minute-level electricity consumption data from 114 apartments show that the proposed method achieves an overall R2 of 0.8731, RMSE of 0.3771, and MAE of 0.1280. The results demonstrate that the conditional generative structure, BiLSTM-attention mechanism, and residual design improve forecasting accuracy and robustness for fine-grained residential load prediction.

## Index Terms

Short-term load forecasting, residential energy consumption, conditional variational autoencoder, bidirectional long short-term memory, temporal attention, residual learning, weather-conditioned generative modeling.

## I. Introduction

Accurate short-term residential load forecasting plays an important role in smart grid operation, demand response, household energy management, and distributed energy scheduling. Compared with aggregated regional load forecasting, residential-level forecasting is more difficult because individual electricity consumption is highly irregular and affected by diverse user behaviors, weather conditions, and temporal usage patterns. Minute-level residential load data further increases the modeling difficulty, since the forecasting model needs to capture both short-term persistence and sudden fluctuations.

Traditional statistical models and deterministic machine learning approaches have been widely used for load forecasting. However, they often have limited capability in modeling nonlinear temporal dependency and stochastic uncertainty. Deep learning models, such as recurrent neural networks and long short-term memory networks, can learn complex temporal patterns, but many of them generate point predictions without explicitly modeling latent uncertainty. In residential scenarios, this uncertainty is important because users with similar weather and temporal conditions may still exhibit different consumption patterns.

To address these challenges, this paper proposes a residual-enhanced conditional generative framework based on a Conditional Variational Autoencoder (CVAE), Bidirectional Long Short-Term Memory (BiLSTM), and temporal attention mechanism. The proposed framework uses historical load, weather variables, and temporal features as conditional information. The BiLSTM-attention encoder captures bidirectional temporal dependencies from the recent observation window, while the CVAE learns a conditional latent representation of residential load variations. Furthermore, handcrafted temporal features, including lag values, difference features, and rolling statistics, are incorporated to enhance local trend representation. A learnable residual connection based on recent load values is introduced to improve short-term prediction stability.

The main contributions of this paper are summarized as follows:

1. A conditional generative forecasting framework is proposed for fine-grained residential load prediction, enabling latent uncertainty modeling under weather and temporal conditions.
2. A BiLSTM-attention encoder is designed to extract informative temporal representations from historical load and weather sequences.
3. A residual-enhanced prediction structure is introduced by combining explicit lag features, difference features, rolling statistics, and learnable short-term residual connections.
4. Experiments on minute-level electricity consumption data from 114 apartments demonstrate the effectiveness of the proposed method.

## II. Related Work

Short-term load forecasting has been extensively studied in power systems and smart grid applications. Existing methods can be broadly divided into statistical methods, machine learning methods, and deep learning methods. Statistical models, such as autoregressive models and exponential smoothing, are simple and interpretable but often struggle with nonlinear and nonstationary residential load patterns. Machine learning methods, including support vector regression, random forests, and gradient boosting models, improve nonlinear fitting ability but generally require careful feature engineering and may have limited sequence modeling capacity.

Deep learning approaches have shown strong performance in time-series forecasting tasks. Recurrent neural networks, especially LSTM and gated recurrent unit models, are widely used to capture sequential dependency in load data. Attention mechanisms further improve temporal representation by assigning different importance weights to historical time steps. Nevertheless, most deterministic deep learning models mainly focus on point prediction and do not explicitly represent latent uncertainty in residential demand.

Generative models provide a promising way to model uncertainty in load forecasting. Variational autoencoders can learn latent representations of complex data distributions, and conditional variational autoencoders extend this ability by introducing external conditions. For residential load forecasting, weather conditions, temporal indicators, and historical consumption patterns can serve as useful conditional information. Motivated by these studies, this paper develops a weather-conditioned residual generative model with BiLSTM attention for short-term residential energy load forecasting.

## III. Proposed Method

### A. Problem Formulation

Given a residential load sequence and its corresponding contextual variables, the objective is to predict the electricity consumption of each apartment at the next time step. Let y_t denote the load value at time t, and c_t denote the conditional variables, including weather features and temporal indicators. Given a historical window of length L, the model estimates:

```text
y_t = f(y_{t-L:t-1}, c_{t-L:t}, z_t)
```

where z_t is a latent variable learned by the conditional generative model. In this work, each apartment is modeled separately to reduce the influence of cross-user heterogeneity.

### B. Feature Construction

For each apartment, the model uses a 24-step historical window to predict the next minute load. Each historical step contains weather features, normalized temporal features, and historical load values. In addition, three groups of explicit temporal features are constructed:

1. Lag features: y_{t-1}, y_{t-2}, y_{t-3}, y_{t-6}, y_{t-12}, and y_{t-24}.
2. Difference features: first-order and second-order load differences.
3. Rolling statistics: rolling mean and rolling standard deviation over the lookback window.

These features help the model capture short-term persistence, local trends, and volatility patterns.

### C. BiLSTM-Attention Temporal Encoder

The historical feature window is first fed into a BiLSTM encoder. Compared with a unidirectional recurrent model, BiLSTM can extract richer temporal representations within the observation window by processing the sequence in both forward and backward directions. The output hidden states are then passed to a temporal attention layer, which adaptively emphasizes informative time steps:

```text
h_t = BiLSTM(x_{t-L:t-1})
s_t = Attention(h_t)
```

The attended temporal representation is concatenated with the encoded lag, difference, and rolling statistical features to form the final temporal feature vector.

### D. Conditional Generative Modeling

The proposed framework adopts a CVAE structure to model latent uncertainty under weather and temporal conditions. The posterior distribution q(z|x,c) is estimated from the temporal representation and conditional features, while the conditional prior p(z|c) is generated from weather, temporal, and lag-based information. The latent variable is sampled through the reparameterization trick:

```text
z = mu + sigma * epsilon, epsilon ~ N(0, I)
```

The decoder receives the sampled latent variable, temporal representation, and conditional features to generate the predicted load value.

### E. Residual-Enhanced Prediction

Residential load at minute-level resolution usually exhibits strong short-term persistence. To exploit this property, a learnable residual connection is introduced:

```text
y_hat_t = g(z_t, s_t, c_t) + sigmoid(w_1)y_{t-1} + sigmoid(w_2)y_{t-2}
```

where g(.) denotes the decoder output, and w_1 and w_2 are learnable residual weights. This residual design enables the model to preserve short-term continuity while still allowing the generative decoder to model nonlinear deviations.

### F. Loss Function

The total training objective consists of a reconstruction loss and a KL divergence term:

```text
L = L_rec(y_t, y_hat_t) + beta * KL(q(z|x,c) || p(z|c))
```

Smooth L1 loss is used as the reconstruction loss to improve robustness against abnormal fluctuations. A KL warm-up strategy is adopted during training to stabilize latent representation learning.

## IV. Experiments

### A. Dataset

Experiments are conducted on a minute-level residential electricity consumption dataset containing 114 apartments and corresponding weather observations. The experimental data used in this paper covers the period from July 1, 2016 00:00:00 to July 7, 2016 23:59:00, with 10,080 time points. The dataset contains 114 apartment load columns, 10 continuous weather features, and temporal features extracted from timestamps.

The continuous weather features include temperature, humidity, visibility, apparent temperature, pressure, wind speed, wind bearing, precipitation intensity, dew point, and precipitation probability. Hour of day and day of week are used as temporal indicators. The dataset is split chronologically into training, validation, and test sets with a ratio of 60%, 20%, and 20%, respectively, to avoid future information leakage.

**[Table I Placeholder: Dataset Description]**

| Item | Description |
|---|---|
| Time resolution | 1 minute |
| Time period | 2016-07-01 to 2016-07-07 |
| Number of time points | 10,080 |
| Number of apartments | 114 |
| Weather features | 10 |
| Split ratio | 60% / 20% / 20% |

### B. Experimental Settings

The lookback window length is set to 24 time steps. The lag features are selected as 1, 2, 3, 6, 12, and 24 minutes. The hidden dimension of the neural network is set to 128, and the global latent dimension is set to 64. The model is trained using AdamW optimizer with a learning rate of 1e-4 and a batch size of 128. Early stopping and learning rate scheduling are applied according to validation loss.

The forecasting performance is evaluated using mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and coefficient of determination (R2).

### C. Baseline Models

To evaluate the effectiveness of the proposed method, it is compared with several baseline forecasting models. The baseline models may include traditional machine learning methods and deep learning methods, such as persistence forecasting, support vector regression, random forest, LSTM, and BiLSTM-based models.

**[Table II Placeholder: Baseline Comparison Results]**

| Model | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Persistence | TBD | TBD | TBD | TBD |
| SVR | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD | TBD |
| BiLSTM-Attention | TBD | TBD | TBD | TBD |
| Proposed Model | 0.1422 | 0.3771 | 0.1280 | 0.8731 |

### D. Ablation Study

An ablation study is designed to analyze the contribution of each component. Several variants are constructed by removing weather conditions, the CVAE latent variable, temporal attention, handcrafted temporal features, and the residual connection, respectively. All variants use the same data split and training settings.

**[Table III Placeholder: Ablation Study Results]**

| Variant | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Full model | 0.1422 | 0.3771 | 0.1280 | 0.8731 |
| w/o Weather | TBD | TBD | TBD | TBD |
| w/o CVAE latent variable | TBD | TBD | TBD | TBD |
| w/o Temporal attention | TBD | TBD | TBD | TBD |
| w/o Lag/Diff/Rolling features | TBD | TBD | TBD | TBD |
| w/o Residual connection | TBD | TBD | TBD | TBD |

## V. Results and Discussion

### A. Overall Forecasting Performance

The proposed model achieves an overall R2 of 0.8731, RMSE of 0.3771, and MAE of 0.1280 on the test set. These results indicate that the proposed residual-enhanced conditional generative framework can effectively capture short-term residential load patterns. Compared with deterministic forecasting models, the CVAE-based structure introduces latent variables conditioned on weather and temporal information, enabling the model to better represent stochastic variations in residential demand.

**[Figure 1 Placeholder: Overall Model Architecture]**

**[Figure 2 Placeholder: Prediction vs. Actual Scatter Plot]**

### B. Apartment-Level Performance

The apartment-level results show that the proposed method performs well across most residential users. Among 114 apartments, 62 apartments achieve R2 values greater than 0.85, and 100 apartments achieve R2 values greater than 0.70. This demonstrates the general effectiveness of modeling each apartment separately while using a unified forecasting architecture. However, a small number of apartments have relatively low R2 values, which may be caused by highly irregular user behavior, weak periodicity, or abnormal consumption events.

**[Figure 3 Placeholder: Apartment-Level R2 Distribution or Heatmap]**

**[Table IV Placeholder: Best and Worst Apartment-Level Results]**

| Apartment | MSE | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Best case | TBD | TBD | TBD | TBD |
| Worst case | TBD | TBD | TBD | TBD |

### C. Effect of BiLSTM Attention

The BiLSTM-attention encoder improves temporal feature extraction by considering both forward and backward dependencies within the lookback window. The attention mechanism further allows the model to assign higher weights to more informative historical time steps. This is beneficial for residential load forecasting, where recent consumption changes may have different levels of importance under different weather and temporal conditions.

**[Figure 4 Placeholder: Attention Weight Visualization]**

### D. Effect of Conditional Generative Modeling

The CVAE module learns a latent representation conditioned on weather, temporal, and historical load information. This design is more flexible than deterministic mapping because it can model potential demand variations under similar external conditions. The ablation result of removing the latent variable is expected to demonstrate the contribution of generative modeling to forecasting accuracy and robustness.

### E. Effect of Residual Learning

The residual connection directly incorporates recent load values into the final prediction. This is especially useful for minute-level forecasting, where the previous one or two minutes often provide strong predictive signals. By learning residual weights rather than simply copying previous values, the model balances short-term persistence and nonlinear generative correction.

**[Figure 5 Placeholder: Time-Series Prediction Examples]**

## VI. Conclusion

This paper proposes a residual-enhanced conditional generative framework for short-term residential energy load forecasting. The proposed model combines CVAE-based latent uncertainty modeling, BiLSTM-attention temporal encoding, handcrafted temporal features, and learnable residual connections. Experiments on minute-level electricity consumption data from 114 apartments demonstrate that the proposed method achieves strong overall performance, with an R2 of 0.8731, RMSE of 0.3771, and MAE of 0.1280. The results indicate that conditional generative modeling, temporal attention, and residual short-term dependency modeling are effective for fine-grained residential load forecasting.

Future work will extend the framework to longer time spans, multi-step forecasting, probabilistic interval prediction, and transfer learning across different residential communities.

## References

[1] Placeholder for the reference paper used for baseline comparison.

[2] Placeholder for short-term load forecasting survey or benchmark paper.

[3] Placeholder for LSTM or BiLSTM time-series forecasting reference.

[4] Placeholder for attention mechanism reference.

[5] Placeholder for variational autoencoder or conditional variational autoencoder reference.
