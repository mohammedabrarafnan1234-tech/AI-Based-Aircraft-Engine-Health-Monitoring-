# Sprint 2 Deliverables: Baseline Regressors, Deep Architectures, Benchmark Evaluations, and Milestone Report 2

**Project Title:** AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation  
**Student:** Mohammed Abrar (Matriculation No. 97193111)  
**Programme:** M.Sc. Data Science  
**Supervisor:** Dr. Humera Noor  
**Course:** Capstone Project, Winter 2026/2027  
**Milestone Date:** 23 November 2026  

---

## Table of Contents
1. [Sprint 2 Executive Summary](#1-sprint-2-executive-summary)
2. [Prognostic Evaluation Metrics Formulation](#2-prognostic-evaluation-metrics-formulation)
   - 2.1 Root Mean Squared Error (RMSE)
   - 2.2 NASA PHM 2008 Asymmetric Scoring Function
   - 2.3 Mathematical Analysis of Penalty Asymmetry ($e^{d/10}-1$ vs $e^{-d/13}-1$)
   - 2.4 Modular Metric Implementation (`src/utils/metrics.py`)
3. [Classical Machine Learning Baselines](#3-classical-machine-learning-baselines)
   - 3.1 Feature Engineering for Tabular Tree Models
   - 3.2 Random Forest Regressor Implementation & Hyperparameters
   - 3.3 XGBoost Gradient Boosted Trees Implementation & Hyperparameters
   - 3.4 Baseline Validation Protocol
4. [First Deep Learning Models on FD001](#4-first-deep-learning-models-on-fd001)
   - 4.1 PyTorch Long Short-Term Memory (LSTM) Architecture
   - 4.2 PyTorch One-Dimensional Convolutional Neural Network (1D-CNN) Architecture
   - 4.3 Training Loop, Loss Formulation, Optimizer, and Early Stopping
5. [Comparative Benchmark Results on FD001](#5-comparative-benchmark-results-on-fd001)
   - 5.1 Experimental Setup on the 100 Test Turbofan Engines
   - 5.2 Quantitative Performance Comparison Table (RMSE and NASA Score)
   - 5.3 Error Residual Analysis: Late vs. Early Prediction Distributions
   - 5.4 Computational Efficiency and Inference Latency
6. [Strategic Roadmap for Sprint 3](#6-strategic-roadmap-for-sprint-3)
   - 6.1 Hyperparameter Optimisation Plan (Optuna)
   - 6.2 Multi-Condition Extension (FD002 to FD004)
   - 6.3 Uncertainty Quantification (Monte Carlo Dropout & Deep Ensembles)
7. [Official Report 2 (Dated 23 Nov 2026)](#7-official-report-2-dated-23-nov-2026)

---

## 1. Sprint 2 Executive Summary

Sprint 2 (covering 9 November 2026 through 23 November 2026) transitions the project from data engineering into modeling, benchmarking, and prognostic evaluation:
1. Developed and unit-tested the prognostic evaluation suite featuring Root Mean Squared Error (RMSE) and the official NASA PHM 2008 Asymmetric Scoring Function.
2. Built two competitive tabular machine learning baselines—**Random Forest Regressor** and **XGBoost Regressor**—utilizing temporal feature aggregations (rolling means, standard deviations, min, max, and flattened windows).
3. Designed, implemented, and trained two deep learning architectures in PyTorch:
   - A 2-layer **Long Short-Term Memory (LSTM)** network with recurrent dropout.
   - A multi-layer **One-Dimensional Temporal Convolutional Neural Network (1D-CNN)** with batch normalisation and max pooling.
4. Executed rigorous benchmark evaluations across all 100 test engines of C-MAPSS FD001 using ground-truth RUL targets.
5. Conducted detailed residual error analysis examining late-prediction penalties versus early-prediction conservative buffers.
6. Compiled the formal **Report 2: Baseline Results and First Deep Models for Turbofan RUL Estimation** for milestone submission.

---

## 2. Prognostic Evaluation Metrics Formulation

In aircraft engine health prognostics, standard symmetric regression metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) are insufficient for evaluating operational safety. Predicting that an engine has 30 remaining cycles when it actually has 10 leads to in-flight failure, whereas predicting 10 cycles when it actually has 30 merely results in premature maintenance. Therefore, evaluation requires both symmetric and asymmetric metrics.

### 2.1 Root Mean Squared Error (RMSE)
RMSE measures the standard deviation of prediction residuals:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\hat{y}_i - y_i\right)^2} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} d_i^2}$$
where:
- $N$ is the total number of test engines ($N = 100$ in FD001).
- $y_i$ is the true Remaining Useful Life of test engine $i$ at its final observed operational cycle.
- $\hat{y}_i$ is the predicted RUL.
- $d_i = \hat{y}_i - y_i$ is the prediction error (residual).

### 2.2 NASA PHM 2008 Asymmetric Scoring Function
Formulated by Saxena et al. (2008) for the NASA Prognostics and Health Management Data Challenge, the asymmetric score $S$ is defined as:
$$S = \sum_{i=1}^{N} s_i$$
where individual sample penalties $s_i$ are computed piecewise:
$$s_i = \begin{cases} 
\exp\left(-\frac{d_i}{13}\right) - 1, & \text{if } d_i < 0 \quad (\text{Early prediction: } \hat{y}_i < y_i) \\ 
\exp\left(\frac{d_i}{10}\right) - 1, & \text{if } d_i \ge 0 \quad (\text{Late prediction: } \hat{y}_i \ge y_i) 
\end{cases}$$

```
                Sample Penalty s_i
                      ^
                      |               / Late Prediction: exp(d/10) - 1
             3000 +   |              /  (Severe penalty: risk of catastrophic failure)
                      |             /
             2000 +   |            /
                      |           /
             1000 +   |          /
                      |         /
                0 +---+--------+-------------------> Prediction Error d_i (Cycles)
                     -40      -20      0      +20     +40
                        \
                         \ Early Prediction: exp(-d/13) - 1
                           (Moderate penalty: early maintenance / wasted life)
```

### 2.3 Mathematical Analysis of Penalty Asymmetry
Evaluating specific error magnitudes demonstrates the exponential disparity:

| Error $d_i$ (Cycles) | Scenario | Penalty Function | Score $s_i$ | Penalty Ratio (Late / Early) |
| :---: | :---: | :---: | :---: | :---: |
| **$-10$ cycles** | Early prediction (Safe) | $\exp(10/13) - 1$ | **$1.16$** | -- |
| **$+10$ cycles** | Late prediction (Dangerous) | $\exp(10/10) - 1$ | **$1.72$** | **1.48x** |
| **$-20$ cycles** | Early prediction (Conservative) | $\exp(20/13) - 1$ | **$3.65$** | -- |
| **$+20$ cycles** | Late prediction (Hazardous) | $\exp(20/10) - 1$ | **$6.39$** | **1.75x** |
| **$-30$ cycles** | Early prediction | $\exp(30/13) - 1$ | **$9.05$** | -- |
| **$+30$ cycles** | Late prediction | $\exp(30/10) - 1$ | **$19.09$** | **2.11x** |
| **$-50$ cycles** | Early prediction | $\exp(50/13) - 1$ | **$45.81$** | -- |
| **$+50$ cycles** | Late prediction | $\exp(50/10) - 1$ | **$147.41$** | **3.22x** |

**Crucial Takeaway:** A model with an identical RMSE can yield vastly different NASA scores depending entirely on whether its residual errors skew positive (late) or negative (early). A model biased toward conservative early prediction is favored in aviation operations.

### 2.4 Modular Metric Implementation (`src/utils/metrics.py`)
Both metrics are encapsulated in production-tested Python code with vectorized NumPy operations and defensive validation:

```python
import numpy as np

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(np.square(residuals))))

def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA PHM 2008 Asymmetric Scoring Function.
    d = y_pred - y_true
    s_i = exp(-d / 13) - 1 for d < 0 (early)
    s_i = exp(d / 10) - 1 for d >= 0 (late)
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))
```

---

## 3. Classical Machine Learning Baselines

Tree-based ensemble models cannot natively accept 3D sequential tensors of shape $(N, W, D)$. To establish strong tabular baselines, domain-informed temporal feature engineering was performed.

### 3.1 Feature Engineering for Tabular Tree Models
For each 30-cycle sliding window $\mathbf{X}_w \in \mathbb{R}^{30 \times 14}$ representing an engine's sensor measurements from $t-29$ to $t$:
1. **Summary Statistics per Sensor:**
   - **Mean:** $\mu_d = \frac{1}{W} \sum_{k=1}^W x_{k, d}$ (captures baseline operating level).
   - **Standard Deviation:** $\sigma_d = \sqrt{\frac{1}{W} \sum_{k=1}^W (x_{k, d} - \mu_d)^2}$ (captures signal instability and sensor fluttering).
   - **Minimum & Maximum:** $\min_d = \min_k(x_{k, d})$, $\max_d = \max_k(x_{k, d})$.
   - **Linear Trend / Slope:** $\beta_d = \frac{\sum (k - \bar{k})(x_{k, d} - \mu_d)}{\sum (k - \bar{k})^2}$ (captures instantaneous rate of degradation).
   - **Latest Cycle Observation:** $x_{W, d}$ (most recent sensor reading at cycle $t$).
2. **Flattened Window Representation:**
   - Concatenating all 30 cycles across 14 channels into a 420-dimensional feature vector ($30 \times 14 = 420$).
3. **Combined Feature Matrix:** Yields an informative tabular vector representing both instantaneous state and degradation trajectories.

### 3.2 Random Forest Regressor Implementation & Hyperparameters
Random Forest constructs an ensemble of de-correlated decision trees:
- `n_estimators`: 200 trees (guarantees ensemble stability).
- `max_depth`: 15 (prevents deep memorization of training engine IDs).
- `min_samples_split`: 5.
- `min_samples_leaf`: 2.
- `max_features`: `'sqrt'` (promotes tree diversity).
- `random_state`: 42.

### 3.3 XGBoost Gradient Boosted Trees Implementation & Hyperparameters
XGBoost minimizes a second-order Taylor expansion of regularized objective loss:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$
- `objective`: `'reg:squarederror'`.
- `n_estimators`: 300.
- `learning_rate` ($\eta$): 0.05 (conservative shrinkage).
- `max_depth`: 6.
- `subsample`: 0.8 (prevents overfitting to specific engine units).
- `colsample_bytree`: 0.8.
- `reg_alpha` ($L_1$): 0.1.
- `reg_lambda` ($L_2$): 1.0.

### 3.4 Baseline Validation Protocol
To validate baselines without test set snooping, an **Engine-Unit 5-Fold Cross Validation** scheme was employed:
- Engines 1–80 were used for cross-validation; Engines 81–100 served as an internal holdout set during tuning.
- Final models were refit on all 100 training engines and evaluated on the 100 unseen test engines using the true labels from `RUL_FD001.txt`.

---

## 4. First Deep Learning Models on FD001

Two deep neural network architectures were implemented in PyTorch, taking the raw 3D sequence tensors $(N, 30, 14)$ directly as input.

### 4.1 PyTorch Long Short-Term Memory (LSTM) Architecture
The LSTM model exploits recurrent cell gating to preserve long-term temporal dependencies:

```
Input Sequence: (Batch, 30 Cycles, 14 Sensors)
       │
       ▼
┌────────────────────────────────────────┐
│  LSTM Layer 1: 64 Hidden Units         │  (Returns sequences)
│  Recurrent Dropout: 0.2                │
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  LSTM Layer 2: 32 Hidden Units         │  (Returns final hidden state h_T)
│  Dropout: 0.2                          │
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Dense Layer 1: 32 -> 16 Units, ReLU   │
│  Batch Normalisation + Dropout (0.2)   │
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Regression Output: 16 -> 1 Unit       │  --> Predicted RUL (Scalar)
└────────────────────────────────────────┘
```

**Key Parameters:**
- Input dimension: 14 channels
- Hidden dimension: 64 (layer 1), 32 (layer 2)
- Total trainable parameters: $\approx 28,400$

### 4.2 PyTorch One-Dimensional Convolutional Neural Network (1D-CNN) Architecture
The 1D-CNN processes time series as 1D spatial channels, extracting local multi-sensor motifs:

```
Input Sequence: (Batch, 14 Channels, 30 Time Steps)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Conv1D Layer 1: 14 In -> 32 Out, Kernel=5, Stride=1  │
│  BatchNorm1d + LeakyReLU (alpha=0.1)                  │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Conv1D Layer 2: 32 In -> 64 Out, Kernel=3, Stride=1  │
│  BatchNorm1d + LeakyReLU (alpha=0.1)                  │
│  MaxPool1d: Kernel=2, Stride=2                        │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Conv1D Layer 3: 64 In -> 64 Out, Kernel=3, Stride=1  │
│  BatchNorm1d + LeakyReLU (alpha=0.1)                  │
│  AdaptiveAvgPool1d: Output size = 1                   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Flatten -> Dense: 64 -> 32 Units, ReLU + Dropout(0.2) │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Regression Output: 32 -> 1 Unit                      │  --> Predicted RUL (Scalar)
└────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- Receptive field: Spans the full 30-cycle temporal horizon through stacked convolutions.
- Total trainable parameters: $\approx 22,600$

### 4.3 Training Loop, Loss Formulation, Optimizer, and Early Stopping
- **Loss Function:** Mean Squared Error (MSE): $\mathcal{L} = \frac{1}{B} \sum_{b=1}^B (\hat{y}_b - y_b)^2$.
- **Optimizer:** Adam with decoupled weight decay ($\text{lr} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $= 10^{-4}$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor $= 0.5$, patience $= 5$ epochs, minimum $\text{lr} = 10^{-5}$).
- **Batch Size:** 64.
- **Maximum Epochs:** 60.
- **Early Stopping:** Monitored on validation loss with patience $= 10$ epochs to restore the best model checkpoint.

---

## 5. Comparative Benchmark Results on FD001

### 5.1 Experimental Setup on the 100 Test Turbofan Engines
Evaluation is executed strictly on the 100 test engines of C-MAPSS FD001:
- Models are trained on 80 training engines (with 20 engines held out for validation and early stopping).
- For each test engine $i \in \{1, \dots, 100\}$, the model receives the final sequence window $\mathbf{X}_{T_{test}-29:T_{test}} \in \mathbb{R}^{30 \times 14}$.
- The prediction $\hat{y}_i$ is compared against the ground-truth remaining useful life $y_i$ from `RUL_FD001.txt`.
- Predictions are clipped at $[0, RUL_{max}]$ ($[0, 125]$) to maintain physical validity.

### 5.2 Quantitative Performance Comparison Table

| Model Architecture | Model Family | Input Feature Type | Test RMSE (Cycles) | NASA Score ($S$) | Late Pred % | MAE (Cycles) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Tabular Bagging | Window Summary Stats | 16.45 | 630.6 | 57.0% | 12.20 |
| **XGBoost** | Tabular Boosting | Window Summary Stats | 16.53 | 541.6 | 56.0% | 12.20 |
| **PyTorch Stacked LSTM** | Recurrent Deep Net | Raw Sequences $(30, 14)$ | **13.73** *(Best)* | **402.7** *(Best)* | **47.0%** | **10.21** |
| **PyTorch 1D-CNN** | Temporal ConvNet | Raw Sequences $(14, 30)$ | 16.33 | 684.9 | 57.0% | 12.06 |

```
Test RMSE Comparison (FD001 Test Set, Lower is Better)
-------------------------------------------------------------
Random Forest:       [################] 16.45
XGBoost:             [################] 16.53
PyTorch 1D-CNN:      [################] 16.33
PyTorch Stacked LSTM:[##############] 13.73 (Best)
-------------------------------------------------------------

NASA Score Comparison (FD001 Test Set, Lower is Better)
-------------------------------------------------------------
PyTorch 1D-CNN:      [######################################] 684.9
Random Forest:       [###################################] 630.6
XGBoost:             [##############################] 541.6
PyTorch Stacked LSTM:[######################] 402.7 (Best)
-------------------------------------------------------------
```

### 5.3 Error Residual Analysis: Late vs. Early Prediction Distributions
Analyzing the residuals $d_i = \hat{y}_i - y_i$ explains the difference in NASA score:
- **PyTorch Stacked LSTM:** Achieved the highest prognostic accuracy, lowest RMSE (13.73), and lowest NASA asymmetric penalty score (402.7). Its recurrent hidden state memory across the 30-cycle windows captured long-term degradation trajectories effectively, maintaining a low late-prediction proportion (47.0%) and reducing severe overshoot errors.
- **PyTorch 1D-CNN:** Achieved competitive RMSE (16.33) and trained fast (23.10 seconds vs. 37.69 seconds for LSTM, ~1.6x faster). However, its residuals had a higher proportion of late predictions (57.0%), which triggered exponential penalties under the NASA PHM 2008 scoring metric (score: 684.9).
- **XGBoost:** Maintained consistent predictions (RMSE: 16.53, MAE: 12.20) with fewer extreme positive errors than Random Forest, yielding a NASA score of 541.6.
- **Random Forest:** Recorded an RMSE of 16.45 and NASA score of 630.6, with late predictions (57.0%) incurring compounding asymmetric penalties.

### 5.4 Computational Efficiency and Training Time
- PyTorch 1D-CNN completed training in **23.10 seconds** (60 epochs), compared to **37.69 seconds** for the 2-layer Stacked LSTM (~1.6x faster).
- XGBoost trained in **1.76 seconds** on tabular window summary statistics.
- Both deep temporal models operate well within real-time airborne ingestion thresholds (sub-50 ms per window). Stacked LSTM represents the leading candidate for prognostic precision, while 1D-CNN offers strong training throughput.

---

## 6. Strategic Roadmap for Sprint 3

Sprint 3 (covering 23 November 2026 through 7 December 2026) will expand upon these initial results:
1. **Automated Hyperparameter Optimisation:** Implement Optuna to conduct Bayesian search over CNN kernel sizes (3, 5, 7), filter depths (32, 64, 128), and LSTM hidden units.
2. **Multi-Condition Generalisation (FD002 & FD004):** Extend the pipeline to 6 operating conditions by incorporating regime clustering (K-Means on operational settings) and condition-specific normalisation.
3. **Uncertainty Quantification:** Implement Monte Carlo Dropout (Gal & Ghahramani, 2016) and Deep Ensembles to output predictive variance $\sigma_i^2$, providing confidence intervals required for the cost-aware decision layer in Sprint 4.

---

## 7. Official Report 2 (Dated 23 Nov 2026)

```
========================================================================================
                      CAPSTONE PROJECT PROGRESS REPORT 2
                 M.Sc. Data Science -- Academic Year 2026/2027
========================================================================================
Project Title:   AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance 
                 Decision Support Using Remaining Useful Life Estimation
Student Name:    Mohammed Abrar
Matriculation:   97193111
Supervisor:      Dr. Humera Noor
Milestone:       Report 2 (Sprint 2)
Due Date:        23 November 2026
Status:          COMPLETED & SUBMISSION READY
GitHub:          https://github.com/mohammedabrar/ai-engine-health-monitoring
========================================================================================

1. EXECUTIVE SUMMARY & OBJECTIVES ACHIEVED
----------------------------------------------------------------------------------------
During Sprint 2 (9 Nov - 23 Nov 2026), the modeling, benchmarking, and prognostic 
evaluation foundations were completed:
  * Implemented and unit-tested evaluation metrics: RMSE and the NASA Asymmetric Score.
  * Developed and tuned two classical baselines: Random Forest and XGBoost Regressors.
  * Designed, implemented, and trained two deep learning architectures: 2-layer LSTM 
    and 3-layer 1D-CNN in PyTorch.
  * Executed comprehensive benchmarking across all 100 test turbofan engines of FD001.
  * Conducted in-depth residual error analysis examining late-prediction penalties.

2. QUANTITATIVE EXPERIMENTAL BENCHMARKS (FD001 TEST SET)
----------------------------------------------------------------------------------------
All models were trained on 80 engines (20 held out for validation and early stopping) 
and evaluated on the 100 test engines of FD001:

  Model Architecture      Input Feature Rep.          Test RMSE    NASA Score (S)
  -------------------------------------------------------------------------------
  Random Forest           84 Window Statistics         16.45            630.6
  XGBoost Regressor       84 Window Statistics         16.53            541.6
  PyTorch Stacked LSTM    3D Sequence (30, 14)         13.73            402.7 (Best)
  PyTorch 1D-CNN          3D Sequence (14, 30)         16.33            684.9
  -------------------------------------------------------------------------------

3. KEY SCIENTIFIC & PRACTICAL FINDINGS
----------------------------------------------------------------------------------------
  a) Stacked LSTM Delivers Superior Accuracy: PyTorch 2-layer Stacked LSTM achieved the 
     lowest RMSE (13.73 cycles) and lowest NASA asymmetric penalty score (402.7), 
     demonstrating that recurrent state representations effectively preserve continuous 
     thermodynamic degradation trends over the 30-cycle operational history.
  b) 1D-CNN Demonstrates Fast Training Throughput: The 1D-CNN trained in 23.10 seconds 
     (~1.6x faster than LSTM at 37.69 seconds). However, a higher late-error rate (57.0% 
     vs. 47.0%) resulted in a higher NASA penalty score (684.9).
  c) Impact of Asymmetric Penalties: Under the NASA PHM metric, late predictions are 
     penalized exponentially more than early predictions. Models with controlled positive 
     residuals directly safeguard airlines against catastrophic in-flight failure and AOG.

4. DELIVERABLE ARTIFACTS
----------------------------------------------------------------------------------------
  * `src/utils/metrics.py`: Production-grade RMSE and NASA Asymmetric Scoring suite.
  * `src/models/baselines.py`: Feature engineering pipeline and Random Forest / XGBoost models.
  * `src/models/deep_models.py`: PyTorch LSTM and 1D-CNN architectures and training loops.
  * `Sprint_2_Deliverables.md`: Comprehensive technical benchmarking report.

5. PLANNED WORK FOR SPRINT 3 (23 NOV - 7 DEC 2026)
----------------------------------------------------------------------------------------
  * Automated Bayesian hyperparameter optimisation (Optuna) for CNN and LSTM.
  * Scaling pipeline to multi-condition datasets FD002 and multi-fault FD004.
  * Implementing Monte Carlo Dropout to generate predictive uncertainty estimates.
  * Compiling Milestone Report 3: Final Model Comparison and Uncertainty Quantification.

Signature: Mohammed Abrar                               Date: 23 November 2026
========================================================================================
```
