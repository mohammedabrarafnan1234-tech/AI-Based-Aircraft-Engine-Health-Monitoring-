# Sprint 3 Deliverables: Multi-Operating Regime Prognostics, Hyperparameter Optimization, and Uncertainty Quantification

**M.Sc. Data Science Capstone Project**  
**Institution:** University of Europe for Applied Sciences  
**Student:** Mohammed Abrar (Matriculation No. 97193111)  
**Supervisor:** Dr. Humera Noor  
**Sprint Duration:** 23 November 2026 – 7 December 2026  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)  

---

## Executive Overview

Sprint 3 achieves all targeted objectives defined in the capstone project plan, substantially advancing the scientific depth, statistical rigor, and operational relevance of the turbofan health monitoring framework.

### Major Milestone Breakthroughs in Sprint 3:
1. **Multi-Condition Generalization (Task 3.2):** Extended the prognostics pipeline from the single-regime sea-level baseline (FD001) to all four NASA C-MAPSS subsets (FD001, FD002, FD003, FD004; 265,963 total records). Developed dynamic per-subset sensor variance selection and 6-cluster KMeans operating regime normalization to decouple flight envelope swings from underlying component degradation.
2. **Multi-Seed Robustness Framework:** Resolved the single-run evaluation limitation of Sprint 2 by benchmarking all four architectures (Random Forest, XGBoost, PyTorch Stacked LSTM, PyTorch 1D-CNN) across **5 fixed random seeds** (`[42, 123, 456, 789, 2026]`), reporting empirical **mean ± std**.
3. **Leak-Free Optuna Hyperparameter Optimization (Task 3.1):** Conducted Tree-structured Parzen Estimator (TPE) Bayesian hyperparameter tuning evaluating strictly on validation engines (zero test set exposure), discovering optimal configurations for temporal sequence modeling (Stacked LSTM: window 40, hidden 32/64, lr 0.0059; Val RMSE 11.31, Test RMSE 13.52, NASA 378.1).
4. **Epistemic Uncertainty Quantification (Task 3.3):** Implemented 50-pass Monte Carlo Dropout and 5-seed Deep Ensembles: MC-dropout 90% intervals covered 63% of test engines (ensemble 39%); $\sigma$–$|error|$ correlation was $r = 0.379$ ($p < 0.001$).
5. **Conservative Uncertainty-Guided RUL Shifting (Task 3.4):** Formulated an asymmetric risk-mitigation policy ($\hat{y}_{shifted} = \hat{y} - \alpha \cdot \sigma$): at $\alpha = 1.0$, the NASA score fell by 34.0% (from 419.6 to 276.8) and late predictions fell from 57.0% to 32.0% (and down to 19.0% at $\alpha = 2.0$), paving the way for the Sprint 4 cost-aware decision layer.

---

## Deliverables Checklist & Task Status

| Task ID | Deliverable / Planned Objective | Status | Artifact / Verification File |
| :---: | :--- | :---: | :--- |
| **3.1** | Hyperparameter tuning with Optuna on validation engines | **Completed** | `scripts/tune_optuna.py`<br/>`results/optuna_tuning_results.json`<br/>`results/sprint3_charts/optuna_tuning_convergence.png` |
| **3.2** | Extension to FD002–FD004 with regime clustering & 5-seed benchmark | **Completed** | `src/preprocessing/multicondition_preprocess.py`<br/>`scripts/run_sprint3_benchmark.py`<br/>`results/sprint3_benchmark_summary.json`<br/>`results/sprint3_charts/benchmark_multisubset_comparison.png` |
| **3.3** | Epistemic uncertainty quantification (MC Dropout & Deep Ensemble) | **Completed** | `scripts/evaluate_uncertainty.py`<br/>`verify_uncertainty.py`<br/>`lstm_fd001_mc.csv`<br/>`results/uncertainty_metrics.json`<br/>`results/sprint3_charts/uncertainty_calibration_coverage.png` |
| **3.4** | Diagnostic error analysis, 10 worst engines audit & conservative shifting | **Completed** | `scripts/error_analysis_shifting.py`<br/>`results/error_analysis_and_shifting.json`<br/>`results/sprint3_charts/error_analysis_and_shifting_curve.png` |
| **3.5** | Progress Report 3 (PDF & Markdown) & Sprint 4 Roadmap | **Completed** | `reports/Progress_Report_3.pdf`<br/>`reports/Report_3_MultiCondition_Uncertainty_Tuning.md`<br/>`Sprint_3_Deliverables.md` |

---

## Quantitative Summary Across C-MAPSS Subsets (Mean ± Std over 5 Seeds)

| Dataset | Operating Conditions | Fault Modes | Champion Architecture | Test RMSE (Cycles) | NASA Asymmetric Score | Late Error % | Mean Train Time |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 (Sea Level) | 1 (HPC Wear) | **Stacked LSTM** | **14.26 ± 0.23** | **424.9 ± 48.7** | 56.4 ± 2.1% | 31.5s |
| **FD002** | 6 Operating Regimes | 1 (HPC Wear) | **Stacked LSTM** | **16.93 ± 0.58** | **2,911.2 ± 976.8** | 45.2 ± 2.9% | 48.0s |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan) | **Stacked LSTM** | **15.12 ± 0.82** | **586.3 ± 187.2** | 57.6 ± 3.9% | 24.8s |
| **FD004** | 6 Operating Regimes | 2 (HPC + Fan) | **Stacked LSTM** | **19.97 ± 0.46** | **9,297.6 ± 3,672.4** | 54.1 ± 6.4% | 62.2s |

*Key Findings Across Datasets:*
- The Stacked LSTM achieved the lowest RMSE across all four subsets.
- However, the LSTM had a worse NASA score than Random Forest on FD002 (2911.2 vs 1498.2) and than 1D-CNN on FD004 (9297.6 vs 3462.4), demonstrating that occasional late prediction outliers incur severe exponential penalties under complex operating regimes.
- Across all multi-condition subsets, operating regime normalization maintained RMSE in the range of **17–22 cycles**.

---

## Uncertainty Quantification & Calibration Summary (FD001 Test Set)

| Methodology | Predictive RMSE | Mean Uncertainty ($\bar{\sigma}$) | Empirical 90% Coverage | Mean Interval Width | Pearson Correlation ($r$) | Spearman Rank ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monte Carlo Dropout (50 Passes)** | 14.00 Cycles | 7.04 Cycles | **63.0%** | 23.2 Cycles | **0.379** ($p < 0.001$) | **0.423** ($p < 0.001$) |
| **5-Seed Deep Ensemble** | 13.54 Cycles | 3.03 Cycles | **39.0%** | 10.0 Cycles | **0.270** ($p = 0.0067$) | **0.318** ($p = 0.0013$) |

- **Coverage Analysis:** Both methods produce empirical coverage below the nominal 90% level, indicating intervals are narrower than nominal and over-confident on outlier residuals.
- **Error Awareness:** Statistically significant Pearson correlation ($r = 0.379, p < 0.001$) confirms that predictive uncertainty scales proportionally with true prediction error.

---

## Conservative RUL Shifting Impact ($\hat{y}_{shifted} = \hat{y} - \alpha \cdot \sigma$)

| Shift Multiplier ($\alpha$) | NASA Score ($S$) | Penalty Reduction (%) | Late Prediction % | Early Prediction % | Test RMSE (Cycles) | Test MAE (Cycles) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0 (Unshifted Baseline)** | 419.6 | +0.0% | 57.0% | 43.0% | 14.00 | 10.10 |
| **0.2** | 368.2 | -12.2% | 54.0% | 46.0% | 13.64 | 9.90 |
| **0.4** | 329.4 | -21.5% | 49.0% | 51.0% | 13.44 | 9.82 |
| **0.6** | 301.8 | -28.1% | 44.0% | 56.0% | 13.39 | 9.85 |
| **0.8** | 284.4 | -32.2% | 40.0% | 60.0% | 13.51 | 10.02 |
| **1.0 (Optimal NASA Score)** | **276.8** | **-34.0%** | **32.0%** | **68.0%** | **13.79** | **10.37** |
| **1.2** | 278.0 | -33.7% | 29.0% | 71.0% | 14.21 | 10.86 |
| **1.5** | 293.9 | -29.9% | 28.0% | 72.0% | 15.10 | 11.64 |
| **2.0 (Max Conservatism)** | 359.8 | -14.2% | **19.0%** | **81.0%** | 17.12 | 13.35 |

---

## Champion Model Selection (Strictly on Validation Performance)

- **Selected Champion Model:** **PyTorch Stacked LSTM**
- **Selection Basis:** Validation RMSE of **11.31 cycles** on FD001 (superior to 14.34 for 1D-CNN, 18.46 for XGBoost, and 19.44 for Random Forest).
- **Caveat:** High raw NASA scores on multi-regime FD002 and FD004 indicate vulnerability to late outliers, establishing the operational necessity of Sprint 4's cost-aware decision layer.

---

## Sprint 4 Roadmap: Cost-Aware Decision Support Layer
1. **Asymmetric Maintenance Loss Formulation:** Quantifying scheduled visit cost ($C_{sched} \approx \$5,000–\$10,000$, illustrative assumption), unscheduled failure penalty ($C_{fail} \approx \$100,000–\$250,000$, illustrative assumption), and wasted remaining life ($C_{waste}$, illustrative assumption).
2. **Threshold Policy Optimization:** Solving for optimal maintenance threshold $\tau^*$ as a function of the predictive distribution $p(RUL|X)$.
3. **Fleet Scheduling Benchmark:** Simulating fleet-wide operational costs against Run-to-Failure and Preventive Maintenance baselines across 100 aircraft engines.
4. **Production Decision UI:** Interactive Streamlit dashboard delivering real-time telemetry, RUL probability density curves, and cost-optimal dispatch recommendations.
