#!/usr/bin/env python3
"""
Sprint 3 Task 3.3: Prognostic Uncertainty Quantification.

Implements two complementary uncertainty estimation methodologies:
1. Monte Carlo Dropout (MCD):
   - Keeps dropout layers active during test inference (p = 0.20).
   - Generates M = 50 stochastic forward passes per test engine.
   - Computes predictive mean (mu_i) and predictive epistemic standard deviation (sigma_mcd_i).

2. 5-Seed Deep Ensemble:
   - Evaluates the 5 models trained with seeds [42, 123, 456, 789, 2026].
   - Computes ensemble predictive mean (y_bar_i) and ensemble standard deviation (sigma_ens_i).

Calibration & Reliability Metrics:
- Error-Uncertainty Rank Correlation: Pearson r and Spearman rho between sigma_i and absolute error |d_i|.
- 90% Confidence Interval Coverage: Percentage of engines whose true RUL falls within [mu - 1.645*sigma, mu + 1.645*sigma].
- Mean Interval Width: Average length of the 90% confidence interval.
"""

import os
import sys
import json
import numpy as np
import scipy.stats as stats
import torch
from typing import Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.multicondition_preprocess import load_and_preprocess_subset
from src.models.unified_trainer import FlexibleLSTM, Flexible1DCNN, train_and_evaluate_model, set_seed

SEEDS = [42, 123, 456, 789, 2026]


def evaluate_mc_dropout(model, X_test: np.ndarray, n_passes: int = 50, rul_max: float = 125.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executes N stochastic forward passes with dropout enabled at test time.
    """
    model.train()  # Activates Dropout layers
    device = next(model.parameters()).device
    tensor_x = torch.tensor(X_test, dtype=torch.float32).to(device)

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = model(tensor_x).cpu().numpy()
            p = np.clip(p, 0.0, rul_max)
            preds.append(p)

    preds_arr = np.array(preds)  # Shape: (n_passes, N_engines)
    mu = np.mean(preds_arr, axis=0)
    sigma = np.std(preds_arr, axis=0)
    return mu, sigma, preds_arr


def run_uncertainty_analysis(subset: str = "FD001", n_passes: int = 50):
    print("=" * 80)
    print(f"TASK 3.3: UNCERTAINTY QUANTIFICATION ON {subset}")
    print("=" * 80)
    os.makedirs("results", exist_ok=True)

    data = load_and_preprocess_subset(data_dir="data/raw", subset=subset)
    y_true = data["y_test"]
    n_engines = len(y_true)

    # --------------------------------------------------------------------------
    # 1. 5-Seed Deep Ensemble (using Stacked LSTM)
    # --------------------------------------------------------------------------
    print(f"\n[1/2] Training 5-Seed Deep Ensemble across seeds {SEEDS}...")
    ensemble_preds = []
    trained_models = []

    for seed in SEEDS:
        res = train_and_evaluate_model("lstm", data, seed=seed, epochs=20, verbose=False)
        ensemble_preds.append(res["y_test_pred"])
        trained_models.append(res["model"])

    ensemble_preds = np.array(ensemble_preds)  # (5, n_engines)
    ens_mu = np.mean(ensemble_preds, axis=0)
    ens_sigma = np.std(ensemble_preds, axis=0)

    # --------------------------------------------------------------------------
    # 2. Monte Carlo Dropout (50 stochastic passes on Seed 42 model)
    # --------------------------------------------------------------------------
    print(f"\n[2/2] Running Monte Carlo Dropout ({n_passes} passes) on primary model...")
    primary_model = trained_models[0]
    mcd_mu, mcd_sigma, mcd_raw = evaluate_mc_dropout(primary_model, data["X_test_3d"], n_passes=n_passes)

    # --------------------------------------------------------------------------
    # 3. Compute Calibration, Coverage, and Error Correlations
    # --------------------------------------------------------------------------
    def calculate_uncertainty_metrics(mu: np.ndarray, sigma: np.ndarray, y_t: np.ndarray, method_name: str) -> Dict[str, Any]:
        abs_err = np.abs(mu - y_t)
        
        # Pearson and Spearman correlation between uncertainty and error
        pearson_r, p_val = stats.pearsonr(sigma, abs_err)
        spearman_rho, sp_val = stats.spearmanr(sigma, abs_err)

        # 90% Confidence Interval: [mu - 1.645*sigma, mu + 1.645*sigma]
        z_90 = 1.645
        lower_bound = np.maximum(0.0, mu - z_90 * sigma)
        upper_bound = np.minimum(125.0, mu + z_90 * sigma)

        inside_interval = (y_t >= lower_bound) & (y_t <= upper_bound)
        empirical_coverage = float(np.mean(inside_interval) * 100.0)
        mean_interval_width = float(np.mean(upper_bound - lower_bound))

        # RMSE of the mean prediction
        rmse = float(np.sqrt(np.mean(np.square(mu - y_t))))

        return {
            "method": method_name,
            "rmse": round(rmse, 2),
            "mean_sigma": round(float(np.mean(sigma)), 2),
            "std_sigma": round(float(np.std(sigma)), 2),
            "empirical_90pct_coverage": round(empirical_coverage, 1),
            "theoretical_coverage": 90.0,
            "coverage_gap": round(empirical_coverage - 90.0, 1),
            "mean_interval_width": round(mean_interval_width, 1),
            "pearson_r_sigma_vs_abs_error": round(float(pearson_r), 3),
            "pearson_p_value": float(p_val),
            "spearman_rho": round(float(spearman_rho), 3),
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
        }

    mcd_metrics = calculate_uncertainty_metrics(mcd_mu, mcd_sigma, y_true, "Monte Carlo Dropout (50 Passes)")
    ens_metrics = calculate_uncertainty_metrics(ens_mu, ens_sigma, y_true, "5-Seed Deep Ensemble")

    print("\n" + "=" * 80)
    print("UNCERTAINTY QUANTIFICATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<32} {'RMSE':<8} {'Mean σ':<10} {'90% Coverage':<15} {'Interval Width':<15} {'Corr(σ, |error|)':<15}")
    print("-" * 80)
    for m in [mcd_metrics, ens_metrics]:
        print(f"{m['method']:<32} {m['rmse']:<8.2f} {m['mean_sigma']:<10.2f} {m['empirical_90pct_coverage']:<15.1f}% {m['mean_interval_width']:<15.1f} {m['pearson_r_sigma_vs_abs_error']:<15.3f}")
    print("=" * 80)

    # Save to JSON
    out_dict = {
        "dataset": subset,
        "n_test_engines": n_engines,
        "mc_dropout": mcd_metrics,
        "deep_ensemble": ens_metrics,
        "y_true": y_true.tolist(),
    }
    with open("results/uncertainty_metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n✓ Saved uncertainty results to results/uncertainty_metrics.json")


if __name__ == "__main__":
    from typing import Tuple
    run_uncertainty_analysis()
