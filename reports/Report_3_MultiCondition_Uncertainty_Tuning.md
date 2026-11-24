# Progress Report 3 (Sprint 3: 23 Nov – 7 Dec 2026)

**AI-Based Aircraft Engine Health Monitoring and Cost-Aware Decision Support Using Remaining Useful Life Estimation**  
**University of Europe for Applied Sciences — M.Sc. Data Science Capstone**  
**Student:** Mohammed Abrar (Matriculation No. 97193111) | **Supervisor:** Dr. Humera Noor  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)  
**Execution Environment:** Apple Silicon (macOS arm64), Python 3.12, PyTorch 2.15.0, Scikit-Learn 1.9.1, XGBoost 3.2.0, Optuna 5.1.0, ReportLab 5.1.0, Pandas 3.1.1, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2  

---

## Executive Summary

Sprint 3 substantially advances this capstone project by transitioning from single-regime sea-level baselines (FD001) to the complete multi-operating regime flight envelopes and compound degradation modes of the NASA C-MAPSS dataset (FD001–FD004, comprising 265,963 sensor telemetry records). 

To directly resolve the single-run evaluation limitation noted in Sprint 2 review feedback, all models were evaluated across **five fixed random seeds** (`[42, 123, 456, 789, 2026]`), reporting empirical **mean ± std** for RMSE, NASA Asymmetric Score ($S$), MAE, and Late Prediction Percentage. In multi-condition subsets (FD002 and FD004), the Stacked LSTM achieved test RMSE of 16.93 ± 0.58 and 19.97 ± 0.46 cycles, leading Random Forest by 0.76 and 0.60 cycles respectively—differences that lie within approximately one standard deviation of model runs. Hyperparameter optimization was conducted with **Optuna**, strictly monitoring validation engine performance to guarantee zero data leakage (winning configuration: window size 40, hidden units 32/64, learning rate 0.0059, achieving validation RMSE 11.31 cycles; a single evaluation on the test set yielded RMSE 13.52 cycles, with a 5-seed comparison planned for Sprint 4). 

Furthermore, this sprint implements **epistemic uncertainty quantification** using Monte Carlo Dropout (50 passes) and 5-seed Deep Ensembles (evaluated on the baseline Stacked LSTM, Seed 42, Window 30): MC-dropout 90% intervals covered 63% of test engines (ensemble 39%), while predictive uncertainty standard deviation ($\sigma_i$) correlated positively with absolute error ($r = 0.379, p < 0.001$). Finally, a diagnostic error audit across operating lifetime regimes informed an exploratory **conservative RUL shifting sensitivity analysis** ($\hat{y}_{shifted} = \hat{y} - \alpha \cdot \sigma$): at $\alpha = 1.0$, the NASA score fell by 34.0% (from 419.6 to 276.8) and late predictions fell from 57.0% to 32.0% (and down to 19.0% at $\alpha = 2.0$). Shifting up to $\alpha \approx 1.0$ initially acts as a directional bias correction on the baseline's 57% late predictions (improving RMSE from 14.00 down to 13.39 at $\alpha = 0.6$ and 13.79 at $\alpha = 1.0$), with genuine accuracy penalties beginning beyond $\alpha > 1.2$ (+1.10 cycles at $\alpha = 1.5$ and +3.12 cycles at $\alpha = 2.0$). This provides the empirical foundation for the upcoming Sprint 4 Cost-Aware Maintenance Decision Support Layer, where $\alpha^*$ will be formally calibrated on validation engine loss distributions.

---

## 1. Status Update: Sprint 3 Tasks Against Project Plan

Status of planned Sprint 3 tasks from the project charter:

| ID | Planned Task (from Project Charter) | Status | Comments & Verified Technical Findings |
| :--- | :--- | :---: | :--- |
| **3.1** | Hyperparameter tuning with Optuna on validation engines | **Done** | Tree-structured Parzen Estimator (TPE) Bayesian search conducted strictly on validation engines (zero test set exposure). Optimal configurations identified: Stacked LSTM (window 40, hidden 32/64, dropout 0.20, lr 0.0059; Val RMSE 11.31; single test evaluation RMSE 13.52, NASA 378.1; 5-seed comparison follows in Sprint 4); 1D-CNN (window 40, filters 32/64/64, kernels 7/3/5, lr 0.0007; Val RMSE 14.34, Test RMSE 15.49, NASA 468.2). |
| **3.2** | Multi-condition extension to FD002–FD004 (multi-regime & multi-fault) | **Done** | Per-subset sensor variance audit performed. 6-cluster KMeans operating regime normalization implemented for FD002 and FD004. Fixed window $W = 30$ used for multi-seed benchmark; left zero-padding implemented for short test sequences (< 30 cycles). 4 architectures benchmarked across all 4 subsets and 5 seeds. Multi-condition RMSE yielded 17–22 cycles; LSTM leads RF by 0.76 and 0.60 cycles on FD002 and FD004 (within ~1 std). |
| **3.3** | Uncertainty quantification (MC Dropout & Deep Ensemble) | **Done** | Implemented 50-pass Monte Carlo Dropout and 5-seed Deep Ensemble on baseline Stacked LSTM (Seed 42, Window 30). MC-dropout 90% intervals covered 63% of test engines (ensemble 39%); intervals are narrower than nominal, reflecting residual over-confidence. Predictive standard deviation ($\sigma$) correlates positively with absolute error ($r = 0.379, p < 0.001$). |
| **3.4** | Diagnostic error analysis & conservative RUL shifting | **Done** | Decomposed prediction errors across Critical ($[0, 30]$ cycles, $n=25$), Mid-life ($(30, 60]$, $n=14$), and Early-life ($(60, 125]$, $n=61$ capped; $n=50$ plus 11 engines $>125$ raw) regimes. Audited top-10 worst engines (9 of 10 in early-life; mean $\sigma = 8.31$ vs fleet 7.04). Exploratory conservative shifting sensitivity analysis reached empirical NASA minimum at $\alpha = 1.0$ (276.8, -34.0% penalty) via bias correction; reduced late predictions from 57.0% down to 19.0% at $\alpha = 2.0$. Operational $\alpha^*$ to be calibrated on validation in Sprint 4. |
| **3.5** | Sprint 3 documentation, PDF compilation & Sprint 4 roadmap | **Done** | Progress Report 3 generated in publication-quality PDF format via ReportLab. Champion model selected based on FD001 validation performance. Detailed architectural roadmap formulated for Sprint 4 cost-aware decision layer with illustrative cost assumptions. |

**Overall Status vs. Plan:** Fully on schedule. All planned deliverables for Sprint 3 have been executed, verified, and documented with zero synthetic fallbacks and leak-free validation protocols.

---

## 2. Multi-Operating Regime Preprocessing Pipeline (Task 3.2)

### 2.1 Why FD001 Sensor Pruning Cannot Be Blindly Reused
In Sprint 1, seven sensors (`s1, s5, s6, s10, s16, s18, s19`) and all three operational settings were pruned because FD001 simulates sea-level static flight conditions with zero flight envelope variation. 

However, in **FD002 and FD004**, aircraft engines operate across **six distinct operational regimes** spanning altitudes from sea level to 42,000 ft, Mach numbers from 0.0 to 0.84, and throttle resolver angles from 20° to 100°. Consequently, sensors that appear flat in FD001 exhibit substantial dynamic range in FD002 and FD004 as thermodynamic boundary conditions fluctuate. 

To prevent catastrophic information loss, an independent sensor variance audit was conducted per subset:

| Dataset | Operating Conditions | Failure Modes | Retained Channels | Dropped Channels | Justification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FD001** | 1 (Sea Level) | 1 (HPC Wear) | **14 Sensors** | 7 channels (`s1, s5, s6, s10, s16, s18, s19`) | Zero or near-zero variance across entire flight profile ($\sigma < 10^{-3}$). |
| **FD002** | 6 Operating Regimes | 1 (HPC Wear) | **17 Sensors** | 4 channels (`s1, s5, s18, s19`) | Sensors 1, 5, 18, 19 remain constant within each operating cluster; other channels vary with regime. |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan) | **16 Sensors** | 5 channels (`s1, s5, s16, s18, s19`) | Sensors 6 and 10 degrade under combined HPC and Fan failure modes; kept active. |
| **FD004** | 6 Operating Regimes | 2 (HPC + Fan) | **17 Sensors** | 4 channels (`s1, s5, s18, s19`) | Constant within each operating cluster; all 17 active thermodynamic sensors retained. |

### 2.2 Operating Regime Normalization via KMeans Clustering
For the multi-regime datasets (FD002 and FD004), raw global Min-Max normalization fails because flight condition shifts dwarf the subtle sensory drift caused by physical component wear.

To overcome this, we implemented **Operating Regime Normalization**:
1. **Regime Identification:** A `KMeans(n_clusters=6, random_state=42)` model was fitted strictly on the three operating settings (`setting_1, setting_2, setting_3`) of the training engines.
2. **Cluster Assignment:** Each flight cycle in both training and test data was mapped to its nearest operational cluster $k \in \{0, 1, 2, 3, 4, 5\}$.
3. **Within-Regime Scaling:** For each cluster $k$ and each retained sensor $s$, minimum and maximum bounds were computed exclusively on the training engines:
   $$\min(s|k) = \min_{train, c \in k} s_c, \quad \max(s|k) = \max_{train, c \in k} s_c$$
   Sensor readings were normalized within their respective regime:
   $$s_{norm} = \frac{s - \min(s|k)}{\max(s|k) - \min(s|k) + \epsilon}$$
   This normalization decouples flight altitude and Mach number variations from the true underlying engine wear trajectory.

### 2.3 Sequence Truncation, Window Dimensions, and Left Zero-Padding
The multi-seed benchmark across all four subsets utilized a fixed temporal sliding window length of **$W = 30$ flight cycles**.

In FD001 and FD003, all test engines contain at least 31 observed flight cycles prior to truncation. However, in FD002 and FD004, several test engines were truncated early in their operational lives:
- **FD002 ($W = 30$):** 6 test engines have fewer than 30 cycles (minimum length: 21 cycles).
- **FD004 ($W = 30$):** 11 test engines have fewer than 30 cycles (minimum length: 19 cycles).

To enable uniform temporal processing across fixed sliding window dimensions ($W = 30$), **left zero-padding** was implemented:
$$X_{pad} = \begin{bmatrix} \mathbf{0}_{(W - L) \times C} \\ X_{obs} \end{bmatrix}$$
where $L < W$ is the observed sequence length and $C$ is the number of active sensor channels. This aligns the latest observed flight cycle to the final temporal step of the sequence without distorting degradation memory.

*(Note: While the multi-seed benchmark standardized on $W = 30$, subsequent Optuna tuning on FD001 identified $W = 40$ as optimal for single-subset modeling. If $W = 40$ were extended to FD002 and FD004, 12 engines in FD002 and 16 engines in FD004 would have $< 40$ cycles and require padding).*

---

## 3. Comprehensive Multi-Subset Benchmark (Task 3.2)

### 3.1 Experimental Protocol
- **Architectures Evaluated:**
  1. *Random Forest Regressor:* 100 trees, `max_features='sqrt'`, trained on 84–102 summary window statistics.
  2. *XGBoost Regressor:* 100 estimators, learning rate 0.05, max depth 6.
  3. *PyTorch Stacked LSTM:* 2-layer recurrent network (64 and 32 hidden units), recurrent dropout 0.20, Adam optimizer ($lr = 0.001$), 16 epochs.
  4. *PyTorch 1D-CNN:* 3 temporal convolutional stages (32, 64, 64 filters), adaptive average pooling, Adam optimizer, 16 epochs.
- **Reproducibility Protocol:** All models were trained across **5 fixed random seeds** (`[42, 123, 456, 789, 2026]`). All performance metrics are reported as **empirical mean ± standard deviation**.
- **Data Splitting:** 80% train engines, 20% validation engines. Scalers and clustering models were fitted strictly on training engines.

### 3.2 Multi-Subset Benchmark Results Table

| Dataset | Architecture | Input Representation | Test RMSE (Cycles) | NASA Score ($S$) | Late Error % | Mean Train Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD001** | Random Forest | 84 Window Stats | 16.89 ± 0.07 | 628.0 ± 22.1 | 55.6 ± 1.2% | 0.9s |
| | XGBoost | 84 Window Stats | 16.77 ± 0.19 | 522.5 ± 20.3 | 56.6 ± 1.5% | 1.8s |
| | **Stacked LSTM** | 3D Sequence $(30, 14)$ | **14.26 ± 0.23** | **424.9 ± 48.7** | 56.4 ± 2.1% | 31.5s |
| | 1D-CNN | 3D Sequence $(14, 30)$ | 17.35 ± 1.03 | 783.6 ± 196.3 | 63.0 ± 4.2% | 36.8s |
| **FD002** | Random Forest | 102 Window Stats | 17.69 ± 0.10 | 1498.2 ± 32.3 | 44.2 ± 1.1% | 3.1s |
| | XGBoost | 102 Window Stats | 17.69 ± 0.25 | 1637.0 ± 95.6 | 43.2 ± 1.6% | 2.7s |
| | **Stacked LSTM** | 3D Sequence $(30, 17)$ | **16.93 ± 0.58** | **2911.2 ± 976.8** | 45.2 ± 2.9% | 48.0s |
| | 1D-CNN | 3D Sequence $(17, 30)$ | 19.11 ± 0.89 | 2102.6 ± 225.8 | 46.3 ± 13.6% | 121.0s |
| **FD003** | Random Forest | 96 Window Stats | 16.82 ± 0.13 | 1094.0 ± 40.7 | 56.8 ± 1.6% | 1.2s |
| | XGBoost | 96 Window Stats | 17.37 ± 0.10 | 1278.1 ± 61.0 | 58.2 ± 1.7% | 2.0s |
| | **Stacked LSTM** | 3D Sequence $(30, 16)$ | **15.12 ± 0.82** | **586.3 ± 187.2** | 57.6 ± 3.9% | 24.8s |
| | 1D-CNN | 3D Sequence $(16, 30)$ | 16.68 ± 0.71 | 661.3 ± 119.6 | 58.0 ± 7.7% | 58.3s |
| **FD004** | Random Forest | 102 Window Stats | 20.57 ± 0.16 | 2910.6 ± 84.5 | 51.3 ± 0.5% | 4.5s |
| | XGBoost | 102 Window Stats | 22.37 ± 0.51 | 4537.3 ± 508.5 | 53.0 ± 1.3% | 3.0s |
| | **Stacked LSTM** | 3D Sequence $(30, 17)$ | **19.97 ± 0.46** | **9297.6 ± 3672.4** | 54.1 ± 6.4% | 62.2s |
| | 1D-CNN | 3D Sequence $(17, 30)$ | 21.68 ± 1.20 | 3462.4 ± 534.1 | 46.7 ± 5.3% | 124.4s |

### 3.3 Key Findings Across Datasets
1. **RMSE Comparison Across Architectures:** Across all four C-MAPSS subsets, the PyTorch Stacked LSTM achieved the lowest test RMSE (14.26 cycles on FD001, 16.93 on FD002, 15.12 on FD003, and 19.97 on FD004). However, on the multi-condition subsets FD002 and FD004, the LSTM leads Random Forest by modest margins of 0.76 cycles (16.93 ± 0.58 vs. 17.69 ± 0.10) and 0.60 cycles (19.97 ± 0.46 vs. 20.57 ± 0.16) respectively—differences that lie within approximately one standard deviation of model runs.
2. **Asymmetric Score Trade-Off:** While the LSTM has the lowest RMSE on all four subsets, it exhibits a worse NASA score than Random Forest on FD002 (2911.2 vs 1498.2) and than the 1D-CNN on FD004 (9297.6 vs 3462.4). Because the NASA metric penalizes late predictions exponentially ($e^{d/10}-1$), a small number of late outlier predictions under complex multi-regime conditions substantially inflate the LSTM's total NASA penalty.
3. **Multi-Regime Complexity:** Operating regime clustering yielded test RMSE within approximately **17–22 cycles** across the multi-condition subsets (FD002 and FD004), whereas single-condition subsets (FD001 and FD003) achieved test RMSE of 14–17 cycles, avoiding the 28–32 cycle errors reported in unnormalized literature.

---

## 4. Hyperparameter Optimization with Optuna (Task 3.1)

### 4.1 Strict Validation-Only Optimization Protocol
A foundational academic requirement of this capstone is **zero data leakage from the test set**. Optuna's Tree-structured Parzen Estimator (TPE) algorithm was configured with the objective function evaluating **exclusively on the 20 validation engines** (units 81–100) across 30 trials per architecture. The test dataset was evaluated exactly once for the final winning configuration.

### 4.2 Search Space and Winning Configurations

| Model Architecture | Hyperparameter | Search Space | Distribution | Winning Optimal Value | Validation RMSE | Single Test Evaluation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacked LSTM** | Window Size ($W$) | $[20, 30, 40]$ | Categorical | **40 Cycles** | **11.31 Cycles** | **Test RMSE 13.52 Cycles**<br/>NASA Score: 378.1 |
| | Hidden Dim 1 | $[32, 64, 128]$ | Categorical | **32 Units** | | |
| | Hidden Dim 2 | $[16, 32, 64]$ | Categorical | **64 Units** | | |
| | Dropout Rate | $[0.1, 0.4]$ | Uniform | **0.20** | | |
| | Learning Rate | $[10^{-4}, 10^{-2}]$ | Log-Uniform | **0.0059** | | |
| **1D-CNN** | Window Size ($W$) | $[20, 30, 40]$ | Categorical | **40 Cycles** | **14.34 Cycles** | **Test RMSE 15.49 Cycles**<br/>NASA Score: 468.2 |
| | Filter Stages | $[16, 32, 64, 128]$ | Categorical | **(32, 64, 64)** | | |
| | Kernel Sizes | $[3, 5, 7]$ | Categorical | **(7, 3, 5)** | | |
| | Dropout Rate | $[0.1, 0.4]$ | Uniform | **0.25** | | |
| | Learning Rate | $[10^{-4}, 10^{-2}]$ | Log-Uniform | **0.0007** | | |

Convergence plots confirm that the TPE sampler rapidly settled on the $W = 40$, hidden dim $(32, 64)$, lr $0.0059$ region, improving validation RMSE from the baseline of 13.24 down to **11.31 cycles**.

*(Note on Tuning vs. 5-Seed Benchmark: The tuned configuration was evaluated once on the test set (RMSE 13.52 cycles, NASA 378.1) to confirm performance gains; a full 5-seed comparison with mean ± std will follow in Sprint 4).*

---

## 5. Epistemic Uncertainty Quantification & Calibration (Task 3.3)

In aviation operations, deterministic point estimates of RUL can be misleading. Predicting an RUL of 15 cycles with $\sigma = \pm 2$ cycles represents an actionable maintenance window, whereas the same prediction with $\sigma = \pm 18$ cycles is an unacceptable flight safety hazard.

### 5.1 Methodology
*Model Attribution:* Epistemic uncertainty quantification and shifting analyses are evaluated on the baseline Stacked LSTM, Seed 42, Window 30 (test RMSE 14.00, NASA 419.6).
1. **Monte Carlo Dropout (MCD):** Following Gal & Ghahramani (2016), dropout layers were kept active during inference ($p = 0.20$). For each test engine $i$, $M = 50$ stochastic forward passes were computed to calculate predictive mean $\mu_i$ and epistemic standard deviation $\sigma_{MCD, i}$.
2. **5-Seed Deep Ensemble:** The five models trained across seeds `[42, 123, 456, 789, 2026]` were aggregated to compute ensemble mean $\bar{y}_i$ and ensemble standard deviation $\sigma_{Ens, i}$.

### 5.2 Reliability & Calibration Findings

| Metric | Monte Carlo Dropout (50 Passes) | 5-Seed Deep Ensemble | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **Predictive RMSE** | 14.00 Cycles | 13.54 Cycles | Averaging predictions across passes/seeds reduces variance. |
| **Mean Uncertainty ($\bar{\sigma}$)** | 7.04 Cycles | 3.03 Cycles | Average epistemic standard deviation across test engines. |
| **Empirical 90% Coverage** | **63.0%** | **39.0%** | Percentage of test engines whose true RUL falls inside $[\mu \pm 1.645\sigma]$. |
| **Mean 90% Interval Width** | 23.2 Cycles | 10.0 Cycles | MC-Dropout produces wider, more conservative bounds than ensemble. |
| **Pearson Correlation ($r$)** | **0.379** ($p = 0.0001$) | **0.270** ($p = 0.0067$) | Statistically significant positive correlation: uncertainty scales with error. |
| **Spearman Rank ($\rho$)** | **0.423** ($p = 1.2 \times 10^{-5}$) | **0.318** ($p = 0.0013$) | Monotonic alignment between predictive uncertainty and absolute error. |

**Calibration Analysis:**
Both Monte Carlo Dropout (63.0% coverage) and Deep Ensembles (39.0% coverage) show empirical coverage below the nominal 90% level, indicating that the predicted intervals are narrower than nominal and over-confident on outlier residuals. However, the statistically significant Pearson correlation ($r = 0.379, p < 0.001$) and Spearman rank correlation ($\rho = 0.423, p < 0.001$) confirm that uncertainty standard deviation $\sigma_i$ reliably scales with true prediction error magnitude.

---

## 6. Diagnostic Error Analysis & Conservative RUL Shifting (Task 3.4)

### 6.1 Error Decomposition by Engine Lifetime Regime
To diagnose where model errors originate, the 100 test engines in FD001 were partitioned into three operational life regimes based on capped target RUL ($\min(y, 125)$):
- **Critical Zone** ($[0, 30]$ cycles, $n=25$)
- **Mid-Life Zone** ($(30, 60]$ cycles, $n=14$)
- **Early-Life Zone** ($(60, 125]$ cycles, $n=61$).

*(Note on Raw vs. Capped RUL: If partitioned by raw uncapped true RUL from `RUL_FD001.txt`, 11 engines have true RUL exceeding 125 cycles up to 145 cycles, yielding: $[0, 30]$ $n=25$, $(30, 60]$ $n=14$, $(60, 125]$ $n=50$, and $>125$ $n=11$. Under standard capped piecewise-linear evaluation ($RUL_{max} = 125$), the 11 engines $>125$ merge into the Early-Life zone, giving $50 + 11 = 61$ engines).*

| Operating Regime | True RUL Range | Engine Count | Sub-Group RMSE | Sub-Group MAE | NASA Score ($S$) | Late Error % | Operational Severity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Critical Zone** | $[0, 30]$ Cycles | 25 | **5.25 Cycles** | **3.73 Cycles** | **13.2** | 56.0% | Imminent failure risk; lowest RMSE (highest precision). |
| **Mid-Life Zone** | $(30, 60]$ Cycles | 14 | 12.92 Cycles | 10.20 Cycles | 38.2 | 78.6% | Maintenance planning window; moderate risk. |
| **Early-Life Zone** | $(60, 125]$ Cycles | 61 | 16.48 Cycles | 12.69 Cycles | 368.2 | 52.5% | Healthy operation; accounts for 87.7% of total NASA penalty. |

**Operational Insight:** In the Critical Zone ($[0, 30]$ cycles), the model achieves its lowest RMSE ($5.25\text{ cycles}$) and lowest MAE ($3.73\text{ cycles}$), generating only 13.2 of the total 419.6 NASA penalty score. The majority of the penalty ($368.2$, or $87.7\%$) originates in the Early-Life zone where healthy engines exhibit minor sensory fluctuations that are difficult to distinguish from early-stage degradation.

### 6.2 Top-10 Worst Predicted Engines Audit
Forensic examination of the 10 engines with the highest absolute prediction error:

| Rank | Engine ID | True RUL | Predicted RUL | Error ($d$) | Direction | Uncertainty $\sigma$ | Lifetime Zone |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 79 | 63.0 | 107.3 | +44.3 | LATE (Unsafe) | 9.29 | Early-Life |
| **2** | 27 | 66.0 | 109.1 | +43.1 | LATE (Unsafe) | 8.30 | Early-Life |
| **3** | 67 | 77.0 | 116.4 | +39.4 | LATE (Unsafe) | 8.99 | Early-Life |
| **4** | 93 | 85.0 | 47.9 | -37.1 | EARLY (Safe) | 5.69 | Early-Life |
| **5** | 45 | 114.0 | 81.3 | -32.7 | EARLY (Safe) | 8.62 | Early-Life |
| **6** | 86 | 89.0 | 116.4 | +27.4 | LATE (Unsafe) | 7.74 | Early-Life |
| **7** | 21 | 57.0 | 83.8 | +26.8 | LATE (Unsafe) | 8.14 | Mid-Life |
| **8** | 15 | 83.0 | 107.5 | +24.5 | LATE (Unsafe) | 10.26 | Early-Life |
| **9** | 2 | 98.0 | 119.8 | +21.8 | LATE (Unsafe) | 7.55 | Early-Life |
| **10** | 16 | 84.0 | 105.2 | +21.2 | LATE (Unsafe) | 8.51 | Early-Life |

**Audit Findings:** 9 of the 10 worst predicted engines are in the early-life zone (true RUL > 60 cycles). Crucially, the mean uncertainty $\sigma$ of these 10 engines is **$8.31\text{ cycles}$**, significantly above the fleet mean of **$7.04\text{ cycles}$**, demonstrating that predictive uncertainty effectively flags unreliable estimates.

### 6.3 Conservative Uncertainty-Guided RUL Shifting
In commercial aviation maintenance, overestimating RUL (late prediction) risks an uncontained in-flight shutdown, whereas underestimating RUL (early prediction) merely results in a premature scheduled shop visit. The NASA scoring function penalizes late predictions exponentially:
$$d = \hat{y} - y_{true}, \quad S = \begin{cases} e^{-d/13} - 1, & d < 0 \text{ (Early, Safe)} \\ e^{d/10} - 1, & d \ge 0 \text{ (Late, Dangerous)} \end{cases}$$

To mitigate this asymmetry, we evaluated **Conservative RUL Shifting**:
$$\hat{y}_{shifted}(\alpha) = \hat{y} - \alpha \cdot \sigma_i$$
where $\alpha \ge 0$ is a risk-aversion multiplier:

| Shift Multiplier ($\alpha$) | NASA Score ($S$) | Penalty Reduction (%) | Late Prediction % | Early Prediction % | Test RMSE (Cycles) | Test MAE (Cycles) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Unshifted Baseline)** | 419.6 | +0.0% | 57.0% | 43.0% | 14.00 | 10.10 |
| **0.2** | 368.2 | -12.2% | 54.0% | 46.0% | 13.64 | 9.90 |
| **0.4** | 329.4 | -21.5% | 49.0% | 51.0% | 13.44 | 9.82 |
| **0.6** | 301.8 | -28.1% | 44.0% | 56.0% | 13.39 | 9.85 |
| **0.8** | 284.4 | -32.2% | 40.0% | 60.0% | 13.51 | 10.02 |
| **1.0 (Sensitivity Minimum)** | **276.8** | **-34.0%** | **32.0%** | **68.0%** | **13.79** | **10.37** |
| **1.2** | 278.0 | -33.7% | 29.0% | 71.0% | 14.21 | 10.86 |
| **1.5** | 293.9 | -30.0% | 28.0% | 72.0% | 15.10 | 11.64 |
| **2.0 (Max Conservatism)** | 359.8 | -14.3% | **19.0%** | **81.0%** | 17.12 | 13.35 |

**Zero Test Leakage Safeguard & Operational Interpretation:**
- **Exploratory Sensitivity Analysis:** The parameter sweep across $\alpha \in [0.0, 2.0]$ in Table 6.3 represents an exploratory post-hoc sensitivity analysis to characterize the response surface of the NASA scoring function and late prediction rate under epistemic uncertainty shifting. To maintain strict zero test data leakage, $\alpha = 1.0$ is NOT selected as an operational decision parameter from test data; rather, the operational risk multiplier $\alpha^*$ will be formally calibrated on validation engine loss distributions in Sprint 4.
- **Directional Bias Correction vs. Accuracy Penalty:** Crucially, shifting by $\alpha \cdot \sigma$ for $\alpha$ up to approximately 1.0 simultaneously reduces the NASA score and improves test RMSE (from 14.00 down to 13.39 at $\alpha = 0.6$ and 13.79 at $\alpha = 1.0$). This occurs because the baseline model exhibits a late prediction bias (erring late 57.0% of the time); conservative shifting initially acts as a directional bias correction. The genuine accuracy cost (RMSE degradation) begins only beyond $\alpha > 1.2$ (rising to 14.21 at $\alpha = 1.2$, 15.10 at $\alpha = 1.5$, and 17.12 at $\alpha = 2.0$).
- **Safety Gain:** Shifting further to $\alpha = 2.0$ cuts late predictions from 57.0% down to **$19.0\%$**, at the cost of an RMSE increase of **$+3.12\text{ cycles}$** ($14.00 \to 17.12$). This empirical trade-off provides the exact mathematical justification for the formal financial optimization framework of Sprint 4.

---

## 7. Final Model Selection & Sprint 4 Decision Layer Roadmap

### 7.1 Final Model Selection
Following the governance protocol of selecting models **solely on validation set metrics**:
- **Selected Champion Model:** **PyTorch Stacked LSTM**
- **Validation Metric Justification:** Validated lowest RMSE on FD001 (11.31 cycles tuned, 13.24 baseline vs. 14.34 for 1D-CNN, 18.46 for XGBoost, and 19.44 for Random Forest).
- **Nuanced Multi-Subset Caveat:** While the LSTM achieves the lowest RMSE across all four subsets, its raw NASA scores on multi-regime datasets FD002 (2911.2 vs 1498.2 for RF) and FD004 (9297.6 vs 2910.6 for RF and 3462.4 for CNN) reflect vulnerability to occasional late outliers. This empirical result proves why Sprint 4's cost-aware decision layer and conservative shifting are necessary before deploying recurrent neural networks into operational aviation environments.

### 7.2 Sprint 4 Architectural Roadmap: Cost-Aware Decision Support Layer
Sprint 4 will translate the calibrated prognostic distributions $p(RUL|X)$ into an operational maintenance decision support engine:
1. **Asymmetric Financial Loss Matrix:** Formulate an industry-calibrated maintenance cost model:
   - Scheduled Maintenance Cost: $C_{sched} \approx \$5,000–\$10,000$ (illustrative assumption)
   - Unscheduled In-Flight Shutdown Failure Cost: $C_{fail} \approx \$100,000–\$250,000$ (illustrative assumption)
   - Wasted Useful Life Penalty: $C_{waste} \cdot (\hat{y} - y_{true})$ for premature replacements (illustrative assumption)
2. **Dynamic Risk-Optimal Replacement Policy:** Derive the critical failure probability threshold $\tau^* = \frac{C_{sched}}{C_{fail} + C_{waste}}$ to trigger maintenance shop visits when $P(RUL \le t) \ge \tau^*$.
3. **Fleet-Wide Cost Simulation:** Benchmark the cost-aware decision layer against standard Run-to-Failure (RTF) and fixed-interval Preventive Maintenance (PM) policies across all 100 test engines.
4. **Interactive Fleet Decision Dashboard:** Develop a Streamlit application featuring live fleet telemetry, confidence interval inspection, and automated maintenance alert dispatching.

---

## Verification & Academic Integrity Confirmation
Every metric, parameter, and distribution reported in this document traces directly to saved files:
- Benchmark: `results/sprint3_benchmark_summary.json`
- Optuna: `results/optuna_tuning_results.json`
- Uncertainty: `results/uncertainty_metrics.json`
- Predictions & Verification: `lstm_fd001_mc.csv` and `verify_uncertainty.py`
- Diagnostic Error & Shifting: `results/error_analysis_and_shifting.json`
