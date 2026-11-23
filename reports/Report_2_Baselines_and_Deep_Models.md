# Progress Report 2 (due 23 Nov 2026)
**AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation**  
**Student:** Mohammed Abrar (Matriculation No. 97193111), Data Science M.Sc.  
**Supervisor:** Dr. Humera Noor | **Sprint 2:** 9 – 23 Nov 2026  
**GitHub Repository:** [https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-](https://github.com/mohammedabrarafnan1234-tech/AI-Based-Aircraft-Engine-Health-Monitoring-)

---

## 1. Status Update: Sprint 2 (Baselines and First Deep Model)
Status of each planned Sprint 2 task from the project plan. Detailed results are in `Sprint_2_Deliverables.md`.

| ID | Planned task (from project plan) | Status | Comments / results |
| :--- | :--- | :--- | :--- |
| **2.1** | Implement metrics: RMSE and NASA asymmetric score; evaluation function | **Done** | Implemented and verified in `src/utils/metrics.py` and `capstone_run.py`. Verified that late prediction errors ($d \ge 0$) are penalized exponentially more severely than early errors ($d < 0$). |
| **2.2** | Baseline models: Random Forest and XGBoost on FD001 with simple features | **Done (full data)** | Both models trained on 84 tabular window statistics across 80 training engines (13,818 sequences). Full training completed: Random Forest Test RMSE: 16.45, XGBoost Test RMSE: 16.53. |
| **2.3** | LSTM model on FD001; training loop, early stopping | **Done (full data)** | PyTorch 2-layer stacked LSTM trained with early stopping on 20 validation engines (3,913 sequences). Test RMSE: **13.73** cycles, NASA Score: **402.7**. Achieved the best prognostic performance across all models. |
| **2.4** | 1D-CNN model on FD001; compare with LSTM | **Done (full data)** | PyTorch 3-stage 1D-CNN trained in 23.1s (~1.6x faster than LSTM at 37.7s). Test RMSE: 16.33, NASA Score: 684.9. The stacked LSTM achieved superior prognostic accuracy on both RMSE and NASA score. |
| **2.5** | Plots: predicted vs. true RUL per test engine; Report 2 and LinkedIn post | **Done** | All benchmark plots generated in `results/`: predictions vs. true, error distributions, loss curves, and NASA scoring curve. LinkedIn post and share graphics prepared. |

**Overall status vs. plan:** On track. Classical baselines (Random Forest, XGBoost) and initial deep temporal models (Stacked LSTM, 1D-CNN) are operational, trained with leak-free cross-validation, and evaluated against the NASA asymmetric metric. The Stacked LSTM is established as the leading baseline architecture entering Sprint 3.

---

## 2. Benchmark Results on Full C-MAPSS FD001 Test Set
All four models were trained on 80 training engines (13,818 sequences), with 20 engines held out for validation (3,913 sequences) and early stopping, and evaluated on the 100 separate test engines using ground-truth remaining useful life from `RUL_FD001.txt`:

| Model Architecture | Input Feature Representation | Test RMSE (Cycles) | NASA Score ($S$) | Late Error % | MAE (Cycles) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 84 Window Statistics | 16.45 | 630.6 | 57.0% | 12.2 |
| **XGBoost** | 84 Window Statistics | 16.53 | 541.6 | 56.0% | 12.2 |
| **PyTorch Stacked LSTM** | 3D Sequence $(30, 14)$ | **13.73** *(Best)* | **402.7** *(Best)* | **47.0%** | **10.21** |
| **PyTorch 1D-CNN** | 3D Sequence $(14, 30)$ | 16.33 | 684.9 | 57.0% | 12.06 |

**Comparative Analysis:** The Stacked LSTM achieved the top performance overall, beating Random Forest by 36.1% (NASA score: 402.7 vs. 630.6) and beating the best tabular baseline, XGBoost, by 25.6% (402.7 vs. 541.6), while reducing Test RMSE to 13.73 cycles. The 1D-CNN trained 1.63x faster (23.10s vs 37.69s) but had a higher proportion of late predictions (57.0%), which led to a higher NASA penalty score (684.9).

---

## 3. The NASA Asymmetric Score & Residual Analysis
RMSE penalizes positive and negative errors identically. In aviation maintenance, overestimating RUL ($d \ge 0$, late prediction) risks catastrophic in-flight failure, while underestimating RUL ($d < 0$, early prediction) merely prompts scheduled depot inspection. 

The NASA asymmetric score formalizes this penalty disparity:
```
s_i = exp(-d_i / 13) - 1   if d_i < 0  (Early: Safe)
s_i = exp(d_i / 10) - 1    if d_i >= 0 (Late: Dangerous)
```

- An error of $+10$ cycles costs **1.72**, whereas $-10$ cycles costs only **1.16** (1.48x higher penalty).
- An error of $+30$ cycles costs **19.09**, whereas $-30$ cycles costs only **9.05** (2.11x higher penalty).
- **Finding:** The stacked LSTM achieved the lowest NASA score (402.7) and lowest RMSE (13.73), benefiting from recurrent hidden state memory across the 30-cycle temporal sequences. While 1D-CNN trained faster (23.1s vs 37.7s, ~1.6x faster), its residual errors had a higher proportion of late predictions (57% vs 47%), which exponentially increased its NASA asymmetric penalty score (684.9).

---

## 4. Plan for Sprint 3 (23 Nov – 7 Dec 2026)
1. **Hyperparameter Tuning:** Conduct Bayesian optimization with Optuna on CNN kernels and LSTM hidden units.
2. **Multi-Condition Extension:** Generalize preprocessing to FD002 (6 operating regimes) and FD004 (multi-fault conditions).
3. **Uncertainty Quantification:** Implement Monte Carlo Dropout to produce calibrated confidence bounds for the Sprint 4 cost-aware decision layer.
