# AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation

**University of Europe for Applied Sciences — M.Sc. Data Science Capstone Project (2026/2027)**  
**Student:** Mohammed Abrar (Matriculation No. 97193111)  
**Supervisor:** Dr. Humera Noor  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)  
**Execution Environment:** Apple Silicon (macOS arm64), Python 3.12, PyTorch 2.15.0, Scikit-Learn 1.9.1, XGBoost 3.2.0, Optuna 5.1.0, ReportLab 5.1.0  

---

## 📌 Project Overview

This repository contains the end-to-end research, engineering, and predictive maintenance platform for estimating the **Remaining Useful Life (RUL)** of turbofan aircraft engines from multivariate sensor time-series telemetry (NASA C-MAPSS benchmark, datasets FD001 through FD004, comprising 265,963 operational flight records).

The core technical objective of this capstone is to solve two persistent challenges in industrial prognostic health management (PHM):
1. **Multi-Regime Operating Generalization:** Developing operating regime clustering (KMeans, $k=6$) and dynamic sensor selection to decouple complex flight envelope variations (altitude up to 42,000 ft, Mach 0.0–0.84, throttle angles 20°–100°) from physical thermodynamic degradation.
2. **Cost-Aware Asymmetric Decision Support:** Bridging the gap between mathematical prediction and operational aviation economics by quantifying **epistemic uncertainty** (Monte Carlo Dropout and Deep Ensembles) and implementing an asymmetric loss mitigation policy that penalizes catastrophic late predictions (in-flight shutdown risks) far more heavily than early maintenance visits.

---

## 🚀 Repository Structure

```
├── README.md                                   # Master project documentation & quickstart guide
├── Sprint_1_Deliverables.md                    # Sprint 1 charter: Literature review, EDA, preprocessing
├── Sprint_2_Deliverables.md                    # Sprint 2 charter: Baselines, deep models, and initial benchmarks
├── Sprint_3_Deliverables.md                    # Sprint 3 charter: Multi-condition, Optuna, UQ & error shifting
├── requirements.txt                            # Pinned production environment dependencies
├── verify_uncertainty.py                       # Single source of truth verification script for Sprint 3
├── lstm_fd001_mc.csv                           # Test inference predictions with 50-pass MC Dropout mean & sigma
├── lstm_fd001_ensemble.csv                     # Test inference predictions with 5-seed Deep Ensemble mean & sigma
├── capstone_run.py                             # Unified CLI runner for end-to-end execution
├── make_sprint_reports.py                      # Master PDF report generation orchestrator
│
├── data/
│   ├── raw/                                    # C-MAPSS datasets (FD001, FD002, FD003, FD004)
│   │   ├── train_FD001.txt ... train_FD004.txt # Run-to-failure engine trajectories
│   │   ├── test_FD001.txt ... test_FD004.txt   # Truncated operational test trajectories
│   │   └── RUL_FD001.txt ... RUL_FD004.txt     # Ground-truth remaining useful life targets
│   └── processed/                              # Normalized, clustered temporal arrays
│
├── reports/                                    # Formal academic milestone progress reports
│   ├── Progress_Report_1.pdf                   # Sprint 1 PDF (Branded, ReportLab)
│   ├── Progress_Report_2.pdf                   # Sprint 2 PDF (Branded, ReportLab)
│   ├── Progress_Report_3.pdf                   # Sprint 3 PDF (Branded, ReportLab)
│   ├── Report_1_Data_Understanding_Preprocessing.md
│   ├── Report_2_Baselines_and_Deep_Models.md
│   └── Report_3_MultiCondition_Uncertainty_Tuning.md
│
├── scripts/                                    # Pipeline execution & analysis scripts
│   ├── download_cmapss.py                      # Automated dataset downloader
│   ├── run_sprint3_benchmark.py                # 5-seed benchmark runner across FD001–FD004
│   ├── tune_optuna.py                          # Bayesian TPE hyperparameter optimization (validation-only)
│   ├── evaluate_uncertainty.py                 # MC Dropout & Deep Ensemble evaluator
│   ├── error_analysis_shifting.py              # Diagnostic regime breakdown & conservative shifting
│   ├── generate_sprint3_charts.py              # High-resolution publication chart generator
│   ├── make_sprint3_report.py                  # ReportLab PDF compiler for Progress Report 3
│   └── apply_report3_fixes.py                  # Automated review verification script
│
├── src/                                        # Core modular Python package
│   ├── eda.py                                  # Sensor variance and degradation exploration
│   ├── preprocessing/
│   │   ├── preprocess.py                       # Single-subset MinMax and sliding window generator
│   │   └── multicondition_preprocess.py        # 6-cluster KMeans normalization & left zero-padding
│   ├── models/
│   │   ├── baselines.py                        # Random Forest & XGBoost tabular regressors
│   │   ├── deep_models.py                      # PyTorch Stacked LSTM & 1D-CNN architectures
│   │   └── unified_trainer.py                  # Multi-seed training, validation, and evaluation harness
│   └── utils/
│       ├── data_loader.py                      # C-MAPSS dataset parser & schema validator
│       └── metrics.py                          # RMSE, MAE, and NASA PHM 2008 Asymmetric Scoring
│
└── results/                                    # Experimental outputs, benchmark summaries, and charts
    ├── sprint3_benchmark_summary.json          # 5-seed benchmark metrics across FD001–FD004
    ├── optuna_tuning_results.json              # Optuna study trials, winning params, and metrics
    ├── uncertainty_metrics.json                # Coverage, interval widths, and correlation metrics
    ├── error_analysis_and_shifting.json        # Regime breakdown, worst-10 audit, and shift table
    └── sprint3_charts/                         # Publication-ready visualization figures
        ├── benchmark_multisubset_comparison.png
        ├── optuna_tuning_convergence.png
        ├── uncertainty_calibration_coverage.png
        └── error_analysis_and_shifting_curve.png
```

---

## ⚙️ Setup Instructions

### Prerequisites
- **Operating System:** macOS (Apple Silicon / Intel), Linux, or Windows.
- **Python Version:** Python 3.10, 3.11, or 3.12.
- **Package Manager:** `pip` or `uv`.

### Step-by-Step Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-.git
   cd AI-Based-Aircraft-Engine-Health-Monitoring-
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install pinned dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🛠 Quickstart Guide

### 1. Download & Verify C-MAPSS Datasets
Download the official NASA C-MAPSS telemetry datasets into `data/raw/`:
```bash
python3 scripts/download_cmapss.py --dataset all
```

### 2. Multi-Condition Preprocessing & Regime Normalization
Execute KMeans operational regime clustering ($k=6$) and sliding window generation ($W=30$ with left zero-padding):
```bash
python3 src/preprocessing/multicondition_preprocess.py --subsets FD001 FD002 FD003 FD004
```

### 3. Run the 5-Seed Multi-Subset Benchmark (Task 3.2)
Train and evaluate Random Forest, XGBoost, Stacked LSTM, and 1D-CNN across 5 fixed random seeds (`[42, 123, 456, 789, 2026]`):
```bash
python3 scripts/run_sprint3_benchmark.py --subsets FD001 FD002 FD003 FD004 --seeds 42 123 456 789 2026
```

### 4. Bayesian Hyperparameter Optimization with Optuna (Task 3.1)
Run Tree-structured Parzen Estimator (TPE) optimization strictly evaluating on validation engines (zero test set leakage):
```bash
python3 scripts/tune_optuna.py --model lstm --n-trials 30
python3 scripts/tune_optuna.py --model cnn --n-trials 30
```

### 5. Epistemic Uncertainty Quantification (Task 3.3)
Run 50-pass Monte Carlo Dropout and 5-seed Deep Ensembles on FD001 test engines:
```bash
python3 scripts/evaluate_uncertainty.py --dataset FD001 --mc-passes 50
```

### 6. Single Source of Truth Verification
Empirically verify 90% coverage, error correlations, worst-10 audit, and conservative shifting directly from exported predictions:
```bash
python3 verify_uncertainty.py --csv lstm_fd001_mc.csv
```

### 7. Compile Publication PDF Progress Reports (Task 3.5)
Generate branded, submission-ready PDF milestone progress reports using ReportLab:
```bash
python3 scripts/make_sprint3_report.py
```

---

## 📊 Summary of Benchmark Results

### 1. Multi-Subset Multi-Seed Benchmark (Mean ± Std over 5 Seeds)

| Dataset | Operating Regimes | Fault Modes | Architecture | Test RMSE (Cycles) | NASA Asymmetric Score ($S$) | Late Error % | Mean Train Time |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 (Sea Level) | 1 (HPC) | Random Forest | 16.89 ± 0.07 | 628.0 ± 22.1 | 55.6 ± 1.2% | 0.9s |
| | | | XGBoost | 16.77 ± 0.19 | 522.5 ± 20.3 | 56.6 ± 1.5% | 1.8s |
| | | | **Stacked LSTM** | **14.26 ± 0.23** | **424.9 ± 48.7** | 56.4 ± 2.1% | 31.5s |
| | | | 1D-CNN | 17.35 ± 1.03 | 783.6 ± 196.3 | 63.0 ± 4.2% | 36.8s |
| **FD002** | 6 Regimes | 1 (HPC) | Random Forest | 17.69 ± 0.10 | 1498.2 ± 32.3 | 44.2 ± 1.1% | 3.1s |
| | | | XGBoost | 17.69 ± 0.25 | 1637.0 ± 95.6 | 43.2 ± 1.6% | 2.7s |
| | | | **Stacked LSTM** | **16.93 ± 0.58** | **2911.2 ± 976.8** | 45.2 ± 2.9% | 48.0s |
| | | | 1D-CNN | 19.11 ± 0.89 | 2102.6 ± 225.8 | 46.3 ± 13.6% | 121.0s |
| **FD003** | 1 (Sea Level) | 2 (HPC, Fan) | Random Forest | 16.82 ± 0.13 | 1094.0 ± 40.7 | 56.8 ± 1.6% | 1.2s |
| | | | XGBoost | 17.37 ± 0.10 | 1278.1 ± 61.0 | 58.2 ± 1.7% | 2.0s |
| | | | **Stacked LSTM** | **15.12 ± 0.82** | **586.3 ± 187.2** | 57.6 ± 3.9% | 24.8s |
| | | | 1D-CNN | 16.68 ± 0.71 | 661.3 ± 119.6 | 58.0 ± 7.7% | 58.3s |
| **FD004** | 6 Regimes | 2 (HPC, Fan) | Random Forest | 20.57 ± 0.16 | 2910.6 ± 84.5 | 51.3 ± 0.5% | 4.5s |
| | | | XGBoost | 22.37 ± 0.51 | 4537.3 ± 508.5 | 53.0 ± 1.3% | 3.0s |
| | | | **Stacked LSTM** | **19.97 ± 0.46** | **9297.6 ± 3672.4** | 54.1 ± 6.4% | 62.2s |
| | | | 1D-CNN | 21.68 ± 1.20 | 3462.4 ± 534.1 | 46.7 ± 5.3% | 124.4s |

*Key Findings & Analytical Hedging:*
- **RMSE Superiority:** The PyTorch Stacked LSTM achieved the lowest test RMSE across all four subsets. However, on multi-condition subsets FD002 and FD004, the LSTM leads Random Forest by modest margins of 0.76 cycles (16.93 vs 17.69) and 0.60 cycles (19.97 vs 20.57) respectively—differences that lie within approximately one standard deviation of model runs.
- **Asymmetric Score Nuance:** While the LSTM achieved lowest RMSE on all subsets, it recorded higher NASA scores than Random Forest on FD002 (2911.2 vs 1498.2) and than 1D-CNN on FD004 (9297.6 vs 3462.4). Because the NASA score penalizes late predictions exponentially ($e^{d/10}-1$), occasional late outliers under multi-regime conditions substantially inflate the total penalty.
- **Multi-Regime Error Bound:** Operating regime clustering yielded test RMSE within approximately **17–22 cycles** on multi-condition subsets (FD002 and FD004), whereas single-condition subsets (FD001 and FD003) achieved **14–17 cycles**.

---

### 2. Bayesian Hyperparameter Optimization Results (Optuna)

Tuned strictly on validation engines (zero test set leakage) across 30 TPE trials:

| Model Architecture | Optimal Hyperparameters Found | Validation RMSE | Single Test Evaluation |
| :--- | :--- | :---: | :---: |
| **Stacked LSTM** | Window $W=40$, Hidden $(32, 64)$, Dropout $0.20$, LR $0.0059$ | **11.31 Cycles** | **Test RMSE 13.52 Cycles** (NASA: 378.1) |
| **1D-CNN** | Window $W=40$, Filters $(32, 64, 64)$, Kernels $(7, 3, 5)$, Dropout $0.25$, LR $0.0007$ | **14.34 Cycles** | **Test RMSE 15.49 Cycles** (NASA: 468.2) |

*(Note: The tuned configuration was evaluated once on the test set; a full 5-seed comparison follows in Sprint 4).*

---

### 3. Epistemic Uncertainty Quantification & Reliability Calibration

Evaluated on baseline Stacked LSTM, Seed 42, Window 30 (test RMSE 14.00, NASA 419.6):

| Uncertainty Method | Predictive RMSE | Mean Uncertainty ($\bar{\sigma}$) | Empirical 90% Coverage | Mean Interval Width | Pearson Correlation ($r$) | Spearman Rank ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monte Carlo Dropout (50 Passes)** | 14.00 Cycles | 7.04 Cycles | **63.0%** | 23.2 Cycles | **0.379** ($p = 0.0001$) | **0.423** ($p < 0.001$) |
| **5-Seed Deep Ensemble** | 13.54 Cycles | 3.03 Cycles | **39.0%** | 10.0 Cycles | **0.270** ($p = 0.0067$) | **0.318** ($p = 0.0013$) |

- **Calibration Finding:** Both methods exhibit empirical coverage below nominal 90%, reflecting narrower intervals and residual over-confidence.
- **Error Awareness:** Statistically significant correlation ($r = 0.379, p < 0.001$) confirms that predictive uncertainty standard deviation $\sigma_i$ reliably scales with true prediction error magnitude.

---

### 4. Conservative Uncertainty-Guided RUL Shifting Sensitivity Analysis

$$\hat{y}_{shifted}(\alpha) = \hat{y} - \alpha \cdot \sigma_i$$

| Multiplier ($\alpha$) | NASA Score ($S$) | Penalty Reduction (%) | Late Prediction % | Early Prediction % | Test RMSE (Cycles) | Test MAE (Cycles) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Baseline)** | 419.6 | +0.0% | 57.0% | 43.0% | 14.00 | 10.10 |
| **0.2** | 368.2 | -12.2% | 54.0% | 46.0% | 13.64 | 9.90 |
| **0.4** | 329.4 | -21.5% | 49.0% | 51.0% | 13.44 | 9.82 |
| **0.6** | 301.8 | -28.1% | 44.0% | 56.0% | 13.39 | 9.85 |
| **0.8** | 284.4 | -32.2% | 40.0% | 60.0% | 13.51 | 10.02 |
| **1.0 (Sensitivity Min)** | **276.8** | **-34.0%** | **32.0%** | **68.0%** | **13.79** | **10.37** |
| **1.2** | 278.0 | -33.7% | 29.0% | 71.0% | 14.21 | 10.86 |
| **1.5** | 293.9 | -30.0% | 28.0% | 72.0% | 15.10 | 11.64 |
| **2.0 (Max Conservatism)** | 359.8 | -14.3% | **19.0%** | **81.0%** | 17.12 | 13.35 |

- **Zero Test Leakage Safeguard:** This sensitivity table explores the empirical response surface of the NASA scoring function under uncertainty shifting. To maintain strict zero data leakage, $\alpha = 1.0$ is NOT selected from test scores; the operational risk multiplier $\alpha^*$ will be formally calibrated on validation engine loss distributions in Sprint 4.
- **Directional Bias Correction:** Shifting by $\alpha \cdot \sigma$ up to $\alpha \approx 1.0$ simultaneously lowers the NASA score and improves test RMSE ($14.00 \to 13.39$ at $\alpha=0.6$, and $13.79$ at $\alpha=1.0$) because the baseline model errs late 57.0% of the time. The shift initially acts as a directional bias correction. The genuine accuracy cost (RMSE degradation) begins only beyond $\alpha > 1.2$ ($14.21 \to 15.10 \to 17.12$).
- **Safety Gain:** Shifting to $\alpha = 2.0$ cuts late predictions from 57.0% down to **19.0%**, establishing the exact mathematical foundation for Sprint 4's cost-aware maintenance decision layer.

---

## 📖 Milestone Deliverables

| Milestone | Date / Period | Deliverable Document | Formats Available |
| :--- | :---: | :--- | :--- |
| **Sprint 1** | 26 Oct – 9 Nov 2026 | **Data Understanding & Preprocessing Pipeline** | • [Progress_Report_1.pdf](reports/Progress_Report_1.pdf)<br/>• [Report_1 Markdown](reports/Report_1_Data_Understanding_Preprocessing.md)<br/>• [Sprint_1_Deliverables.md](Sprint_1_Deliverables.md) |
| **Sprint 2** | 9 Nov – 23 Nov 2026 | **Baselines & First Deep Models** | • [Progress_Report_2.pdf](reports/Progress_Report_2.pdf)<br/>• [Report_2 Markdown](reports/Report_2_Baselines_and_Deep_Models.md)<br/>• [Sprint_2_Deliverables.md](Sprint_2_Deliverables.md) |
| **Sprint 3** | 23 Nov – 7 Dec 2026 | **Multi-Regime Prognostics, Optuna Tuning & Uncertainty** | • [Progress_Report_3.pdf](reports/Progress_Report_3.pdf)<br/>• [Report_3 Markdown](reports/Report_3_MultiCondition_Uncertainty_Tuning.md)<br/>• [Sprint_3_Deliverables.md](Sprint_3_Deliverables.md) |
| **Sprint 4** | 7 Dec – 21 Dec 2026 | **Cost-Aware Decision Support Layer & Dashboard** | • *Upcoming:* Dynamic replacement threshold policy $\tau^* = C_{sched} / (C_{fail} + C_{waste})$<br/>• *Upcoming:* Streamlit fleet health monitoring dashboard |
