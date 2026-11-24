#!/usr/bin/env python3
"""
Sprint 3 Publication-Quality Scientific Chart Generator.

Generates 4 comprehensive analytical figures for Report 3 and GitHub repository:
1. results/sprint3_charts/benchmark_multisubset_comparison.png:
   - 4 Subsets (FD001-FD004) x 4 Architectures (RF, XGB, LSTM, CNN)
   - Dual-axis bar charts with error bars (mean ± std across 5 random seeds) for RMSE and NASA Score.

2. results/sprint3_charts/optuna_tuning_convergence.png:
   - Optimization history for Stacked LSTM and 1D-CNN.
   - Evolution of Validation RMSE across 30+ trials, highlighting best hyperparameter configurations.

3. results/sprint3_charts/uncertainty_calibration_coverage.png:
   - Panel A: Test engine predictions with 90% confidence bands ([mu - 1.645*sigma, mu + 1.645*sigma]) vs True RUL.
   - Panel B: Uncertainty calibration scatter: absolute error |d_i| vs. predictive sigma_i with linear regression trendline.

4. results/sprint3_charts/error_analysis_and_shifting_curve.png:
   - Panel A: Error distribution across operating regimes: Critical ([0, 30]), Mid-life ((30, 60]), Early-life ((60, 125]).
   - Panel B: Conservative RUL Shifting Curve: NASA penalty score and Late Prediction % as a function of alpha in [0.0, 1.5].
"""

import os
import sys
import json
import numpy as np

# Set matplotlib cache directory to writable location
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "figure.autolayout": True,
})

OUT_DIR = "results/sprint3_charts"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_benchmark_multisubset():
    summary_file = "results/sprint3_benchmark_summary.json"
    if not os.path.exists(summary_file):
        print(f"[Warning] {summary_file} not found. Skipping benchmark plot.")
        return

    with open(summary_file) as f:
        data = json.load(f)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    models = ["rf", "xgb", "lstm", "cnn"]
    model_labels = ["Random Forest", "XGBoost", "Stacked LSTM", "1D-CNN"]
    colors = ["#2E86AB", "#F6AE2D", "#2ECC71", "#E74C3C"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(subsets))
    width = 0.20

    # Panel A: Test RMSE (mean ± std)
    for i, (m, label, color) in enumerate(zip(models, model_labels, colors)):
        means = [data[s][m]["test_rmse_mean"] for s in subsets]
        stds = [data[s][m]["test_rmse_std"] for s in subsets]
        pos = x + (i - 1.5) * width
        rects = axes[0].bar(pos, means, width, yerr=stds, label=label, color=color, capsize=4, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            axes[0].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[0].set_title("Test RMSE across C-MAPSS Subsets (Mean ± Std, 5 Seeds)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_ylabel("RMSE (Flight Cycles)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["FD001\n(1 cond, 1 fault)", "FD002\n(6 cond, 1 fault)", "FD003\n(1 cond, 2 faults)", "FD004\n(6 cond, 2 faults)"], fontsize=10)
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.9)
    axes[0].grid(True, axis="y")

    # Panel B: NASA Score (mean ± std)
    for i, (m, label, color) in enumerate(zip(models, model_labels, colors)):
        means = [data[s][m]["test_nasa_mean"] for s in subsets]
        stds = [data[s][m]["test_nasa_std"] for s in subsets]
        pos = x + (i - 1.5) * width
        rects = axes[1].bar(pos, means, width, yerr=stds, label=label, color=color, capsize=4, alpha=0.9, edgecolor="#222222", linewidth=0.8)
        for rect in rects:
            h = rect.get_height()
            axes[1].annotate(f"{int(round(h))}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    axes[1].set_title("NASA Asymmetric Penalty Score across Subsets (Mean ± Std)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_ylabel("NASA Asymmetric Score", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["FD001\n(1 cond, 1 fault)", "FD002\n(6 cond, 1 fault)", "FD003\n(1 cond, 2 faults)", "FD004\n(6 cond, 2 faults)"], fontsize=10)
    axes[1].legend(loc="upper left", frameon=True, framealpha=0.9)
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "benchmark_multisubset_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved multi-subset benchmark comparison to {out_path}")


def plot_optuna_convergence():
    opt_file = "results/optuna_tuning_results.json"
    if not os.path.exists(opt_file):
        print(f"[Warning] {opt_file} not found. Skipping optuna plot.")
        return

    with open(opt_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (m_key, title, color) in enumerate([("lstm", "Stacked LSTM", "#2ECC71"), ("cnn", "1D-CNN", "#E74C3C")]):
        if m_key not in data:
            continue
        m_data = data[m_key]
        trials = m_data["trials"]
        t_nums = [t["number"] for t in trials]
        t_vals = [t["value"] for t in trials]
        
        # Cumulative best
        best_so_far = np.minimum.accumulate(t_vals)

        axes[idx].scatter(t_nums, t_vals, color=color, alpha=0.5, s=40, label="Trial Val RMSE")
        axes[idx].plot(t_nums, best_so_far, color="#222222", linewidth=2.2, linestyle="-", label="Best Val RMSE")
        axes[idx].axhline(y=m_data["best_val_rmse"], color="#E67E22", linestyle="--", linewidth=1.5, label=f"Optimal: {m_data['best_val_rmse']:.2f}")

        axes[idx].set_title(f"Optuna Optimization: {title} (Validation Only)", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Trial Iteration Number", fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("Validation RMSE (Cycles)", fontsize=10, fontweight="bold")
        axes[idx].grid(True)
        axes[idx].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "optuna_tuning_convergence.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved Optuna tuning convergence plot to {out_path}")


def plot_uncertainty():
    unc_file = "results/uncertainty_metrics.json"
    if not os.path.exists(unc_file):
        print(f"[Warning] {unc_file} not found. Skipping uncertainty plot.")
        return

    with open(unc_file) as f:
        data = json.load(f)

    y_true = np.array(data["y_true"])
    mc = data["mc_dropout"]
    mu = np.array(mc["mu"])
    sigma = np.array(mc["sigma"])
    n_engines = len(y_true)
    engine_ids = np.arange(1, n_engines + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Predictions with 90% confidence interval vs True RUL
    # Sort by true RUL for clear visualization
    sort_idx = np.argsort(y_true)
    sorted_true = y_true[sort_idx]
    sorted_mu = mu[sort_idx]
    sorted_sigma = sigma[sort_idx]
    sorted_x = np.arange(1, n_engines + 1)

    axes[0].plot(sorted_x, sorted_true, "k-", linewidth=2.0, label="Ground Truth RUL")
    axes[0].plot(sorted_x, sorted_mu, color="#2980B9", linewidth=1.5, label="Predictive Mean (μ)")
    axes[0].fill_between(
        sorted_x,
        np.maximum(0, sorted_mu - 1.645 * sorted_sigma),
        np.minimum(125, sorted_mu + 1.645 * sorted_sigma),
        color="#2980B9",
        alpha=0.25,
        label=f"90% Confidence Interval (Coverage: {mc['empirical_90pct_coverage']}%)",
    )

    axes[0].set_title("Test Engines Sorted by Degradation Stage: Predictions & 90% Intervals", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Test Engines (Sorted by Ascending True RUL)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Remaining Useful Life (Cycles)", fontsize=10, fontweight="bold")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", frameon=True)

    # Panel B: Uncertainty vs Absolute Error Scatter & Correlation
    abs_err = np.abs(mu - y_true)
    axes[1].scatter(sigma, abs_err, color="#8E44AD", alpha=0.7, s=45, edgecolors="#4A235A", label="Test Engines")

    # Fit trendline
    slope, intercept = np.polyfit(sigma, abs_err, 1)
    x_vals = np.linspace(sigma.min(), sigma.max(), 100)
    axes[1].plot(x_vals, slope * x_vals + intercept, color="#E74C3C", linewidth=2.0, linestyle="--", label=f"Trendline (r = {mc['pearson_r_sigma_vs_abs_error']:.3f})")

    axes[1].set_title(f"Uncertainty Calibration: Predictive σ vs. Absolute Error |d|\n(Pearson r = {mc['pearson_r_sigma_vs_abs_error']:.3f}, p < 0.001)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predictive Epistemic Uncertainty (Standard Deviation σ)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Absolute Prediction Error |y_hat - y_true|", fontsize=10, fontweight="bold")
    axes[1].grid(True)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "uncertainty_calibration_coverage.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved uncertainty calibration plot to {out_path}")


def plot_error_analysis_and_shifting():
    err_file = "results/error_analysis_and_shifting.json"
    if not os.path.exists(err_file):
        print(f"[Warning] {err_file} not found. Skipping shifting plot.")
        return

    with open(err_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Regime Error Breakdown
    regimes = data["regime_breakdown"]
    labels = ["Critical Zone\n[0, 30] Cycles", "Mid-Life Zone\n(30, 60] Cycles", "Early-Life Zone\n(60, 125] Cycles"]
    keys = ["critical_0_30", "mid_30_60", "early_60_125"]
    rmses = [regimes[k]["rmse"] for k in keys]
    maes = [regimes[k]["mae"] for k in keys]
    late_pcts = [regimes[k]["late_pct"] for k in keys]

    x = np.arange(len(labels))
    w = 0.28
    rects1 = axes[0].bar(x - w/2, rmses, w, label="RMSE (Cycles)", color="#E74C3C", alpha=0.9, edgecolor="#222222")
    rects2 = axes[0].bar(x + w/2, maes, w, label="MAE (Cycles)", color="#3498DB", alpha=0.9, edgecolor="#222222")

    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_title("Prognostic Error Breakdown by Engine Operating Lifetime Regime", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Error (Flight Cycles)", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, axis="y")

    # Panel B: Conservative Shifting Curve (Dual Axis)
    shift_data = data["conservative_shifting"]
    alphas = [e["alpha"] for e in shift_data]
    nasas = [e["nasa_score"] for e in shift_data]
    late_pcts = [e["late_pct"] for e in shift_data]
    rmses_shift = [e["rmse"] for e in shift_data]

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    l1 = ax2.plot(alphas, nasas, color="#C0392B", marker="o", linewidth=2.4, label="NASA Penalty Score (S)")
    l2 = ax2_twin.plot(alphas, late_pcts, color="#27AE60", marker="s", linewidth=2.4, linestyle="--", label="Late Prediction Share (%)")

    ax2.set_xlabel("Conservative Shift Multiplier (α in: y_hat - α*σ)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("NASA Asymmetric Score", color="#C0392B", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Late Prediction Share (%)", color="#27AE60", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#C0392B")
    ax2_twin.tick_params(axis="y", labelcolor="#27AE60")

    # Add combined legend
    lines = l1 + l2
    labels_combo = [l.get_label() for l in lines]
    ax2.legend(lines, labels_combo, loc="upper right", frameon=True)
    ax2.set_title("Conservative RUL Shifting Trade-Off: Eliminating Unsafe Late Predictions", fontsize=12, fontweight="bold")
    ax2.grid(True)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "error_analysis_and_shifting_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved error analysis and conservative shifting plot to {out_path}")


def main():
    plot_benchmark_multisubset()
    plot_optuna_convergence()
    plot_uncertainty()
    plot_error_analysis_and_shifting()


if __name__ == "__main__":
    main()
