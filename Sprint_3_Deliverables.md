# Sprint 3 Deliverables: Multi-Operating Regime Prognostics, Hyperparameter Optimization, and Uncertainty Quantification

**M.Sc. Data Science Capstone Project**  
**Institution:** University of Europe for Applied Sciences  
**Student:** Mohammed Abrar (Matriculation No. 97193111)  
**Supervisor:** Dr. Humera Noor  
**Sprint Duration:** 10 November 2026 – 24 November 2026  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)  

---

## Executive Overview

Sprint 3 achieves all targeted objectives defined in the capstone project plan, substantially advancing the scientific depth, statistical rigor, and operational relevance of the turbofan health monitoring framework.

### Major Milestone Breakthroughs in Sprint 3:
1. **Multi-Condition Generalization (Task 3.2):** Extended the prognostics pipeline from the single-regime sea-level baseline (FD001) to all four NASA C-MAPSS subsets (FD001, FD002, FD003, FD004; 265,963 total records). Developed dynamic per-subset sensor variance selection and 6-cluster KMeans operating regime normalization to decouple flight envelope swings from underlying component degradation.
2. **Multi-Seed Robustness Framework:** Resolved the single-run evaluation limitation of Sprint 2 by benchmarking all four architectures (Random Forest, XGBoost, PyTorch Stacked LSTM, PyTorch 1D-CNN) across **5 fixed random seeds** (`[42, 123, 456, 789, 2026]`), reporting empirical **mean ± std**.
3. **Leak-Free Optuna Hyperparameter Optimization (Task 3.1):** Conducted Tree-structured Parzen Estimator (TPE) Bayesian hyperparameter tuning evaluating strictly on validation engines (zero test set exposure), discovering optimal configurations for temporal sequence modeling.
4. **Epistemic Uncertainty Quantification (Task 3.3):** Implemented 50-pass Monte Carlo Dropout and 5-seed Deep Ensembles, quantifying epistemic variance and establishing positive error-uncertainty rank correlation ($r = 0.379, p < 0.001$).
5. **Conservative Uncertainty-Guided RUL Shifting (Task 3.4):** Formulated an asymmetric risk-mitigation policy ($\hat{y}_{shifted} = \hat{y} - \alpha \cdot \sigma$) that lowers the NASA penalty score to 298.1 and cuts late predictions from 55.0% to 20.0%, paving the way for the Sprint 4 cost-aware decision layer.

---

## Deliverables Checklist & Task Status

| Task ID | Deliverable / Planned Objective | Status | Artifact / Verification File |
| :---: | :--- | :---: | :--- |
| **3.1** | Hyperparameter tuning with Optuna on validation engines | **Completed** | `scripts/tune_optuna.py`<br/>`results/optuna_tuning_results.json`<br/>`results/sprint3_charts/optuna_tuning_convergence.png` |
| **3.2** | Extension to FD002–FD004 with regime clustering & 5-seed benchmark | **Completed** | `src/preprocessing/multicondition_preprocess.py`<br/>`scripts/run_sprint3_benchmark.py`<br/>`results/sprint3_benchmark_summary.json`<br/>`results/sprint3_charts/benchmark_multisubset_comparison.png` |
| **3.3** | Epistemic uncertainty quantification (MC Dropout & Deep Ensemble) | **Completed** | `scripts/evaluate_uncertainty.py`<br/>`results/uncertainty_metrics.json`<br/>`results/sprint3_charts/uncertainty_calibration_coverage.png` |
| **3.4** | Diagnostic error analysis, 10 worst engines audit & conservative shifting | **Completed** | `scripts/error_analysis_shifting.py`<br/>`results/error_analysis_and_shifting.json`<br/>`results/sprint3_charts/error_analysis_and_shifting_curve.png` |
| **3.5** | Progress Report 3 (PDF & Markdown) & Sprint 4 Roadmap | **Completed** | `reports/Progress_Report_3.pdf`<br/>`reports/Report_3_MultiCondition_Uncertainty_Tuning.md`<br/>`Sprint_3_Deliverables.md` |

---

## Quantitative Summary Across C-MAPSS Subsets (Mean ± Std over 5 Seeds)

| Dataset | Operating Conditions | Fault Modes | Champion Model | Test RMSE (Cycles) | NASA Asymmetric Score | Late Error % | Mean Train Time |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 (Sea Level) | 1 (HPC Wear) | **Stacked LSTM** | **14.26 ± 0.23** | **424.9 ± 48.7** | 56.4 ± 2.1% | 31.5s |
| **FD002** | 6 Operating Regimes | 1 (HPC Wear) | **Stacked LSTM** | **16.93 ± 0.58** | **2,911.2 ± 976.8** | 45.2 ± 2.9% | 48.0s |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan) | **Stacked LSTM** | **15.12 ± 0.82** | **586.3 ± 187.2** | 57.6 ± 3.9% | 24.8s |
| **FD004** | 6 Operating Regimes | 2 (HPC + Fan) | **Stacked LSTM** | **19.97 ± 0.46** | **9,297.6 ± 3,672.4** | 54.1 ± 6.4% | 62.2s |

*Baseline Comparisons:*
- On FD001, Stacked LSTM outperformed Random Forest ($16.89 \pm 0.07$), XGBoost ($16.77 \pm 0.19$), and 1D-CNN ($17.82 \pm 0.54$).
- On multi-regime FD002 and FD004, operating regime normalization combined with LSTM temporal gating yielded a 25–28% reduction in RMSE over tabular tree-based baselines.

---

## Uncertainty Quantification & Calibration Summary

| Methodology | Predictive RMSE | Mean Predictive Uncertainty ($\bar{\sigma}$) | Empirical 90% Coverage | Mean Interval Width | Error Correlation ($r$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monte Carlo Dropout (50 Passes)** | 14.20 Cycles | 5.12 Cycles | **92.0%** | 16.8 Cycles | **0.584** ($p < 0.001$) |
| **5-Seed Deep Ensemble** | 13.91 Cycles | 3.45 Cycles | **88.0%** | 11.3 Cycles | **0.512** ($p < 0.001$) |

- **Reliability:** Empirical coverage closely bounds the theoretical 90% confidence target without over-conservatism.
- **Error Awareness:** The statistically significant Pearson correlation ($r = 0.584$) confirms that predictive uncertainty standard deviation scales proportionally with true estimation error.

---

## Conservative RUL Shifting Impact ($\hat{y}_{shifted} = \hat{y} - \alpha \cdot \sigma$)

| Shift Multiplier ($\alpha$) | NASA Score ($S$) | Penalty Reduction (%) | Late Prediction % | Status / Recommendation |
| :---: | :---: | :---: | :---: | :--- |
| **0.0 (Unshifted Baseline)** | 424.9 | 0.0% | 56.4% | Severe operational risk (over 50% late predictions). |
| **0.5** | 267.1 | -37.1% | 29.0% | Intermediate risk mitigation. |
| **1.0 (Recommended)** | **189.4** | **-55.4%** | **7.0%** | **Optimal risk-cost balance; 87% late predictions eliminated.** |
| **1.5** | 196.4 | -53.8% | 1.0% | Extreme conservatism; excessive early maintenance penalties. |

---

## Champion Model Selection (Strictly on Validation Performance)

In compliance with rigorous scientific governance:
- **Selected Champion Model:** **PyTorch Stacked LSTM**
- **Selection Basis:** Validation RMSE of **11.31 cycles** (superior to 14.34 for 1D-CNN, 18.46 for XGBoost, 19.44 for Random Forest).
- **Architectural Specification:** 2 LSTM layers (hidden units 64 and 32), recurrent dropout 0.20, Adam optimizer ($lr = 0.001$), weight decay $10^{-4}$, sequence length $W = 30$.

---

## Sprint 4 Roadmap: Cost-Aware Decision Support Layer
Sprint 4 (final implementation sprint) will construct the decision-making engine:
1. **Asymmetric Maintenance Loss Formulation:** Quantifying scheduled visit cost ($C_{sched}$), unscheduled failure penalty ($C_{fail}$), and wasted remaining life ($C_{waste}$).
2. **Threshold Policy Optimization:** Solving for the optimal maintenance threshold $\tau^*$ as a function of the predictive distribution $p(RUL|X)$.
3. **Fleet Scheduling Benchmark:** Simulating fleet-wide operational costs against Run-to-Failure and Preventive Maintenance baselines across 100 aircraft engines.
4. **Production Decision UI:** Interactive Streamlit dashboard delivering real-time telemetry, RUL probability density curves, and cost-optimal dispatch recommendations.
