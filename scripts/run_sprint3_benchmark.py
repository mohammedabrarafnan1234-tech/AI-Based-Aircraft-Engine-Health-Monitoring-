#!/usr/bin/env python3
"""
Sprint 3 Task 3.2: Multi-Subset Multi-Seed Prognostic Benchmark (High-Throughput).

Evaluates 4 architectures across 4 NASA C-MAPSS subsets across 5 fixed random seeds:
- FD001: 1 operating regime, 1 fault mode (HPC degradation)
- FD002: 6 operating regimes (KMeans clustered), 1 fault mode
- FD003: 1 operating regime, 2 fault modes (HPC + Fan)
- FD004: 6 operating regimes (KMeans clustered), 2 fault modes

Seeds: [42, 123, 456, 789, 2026]
Optimizations:
- Multi-threaded PyTorch execution (torch.set_num_threads(4))
- Adaptive batch sizing (256 for large datasets FD002/FD004; 128 for FD001/FD003)
- Incremental checkpointing (saves after every architecture so results are never lost)
- Resumes automatically if partial results already exist.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.multicondition_preprocess import load_and_preprocess_subset
from src.models.unified_trainer import train_and_evaluate_model

SEEDS = [42, 123, 456, 789, 2026]
MODELS = ["rf", "xgb", "lstm", "cnn"]
MODEL_DISPLAY_NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lstm": "Stacked LSTM",
    "cnn": "1D-CNN",
}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def run_benchmark():
    os.makedirs("results", exist_ok=True)
    raw_path = "results/sprint3_benchmark_raw.json"
    summary_path = "results/sprint3_benchmark_summary.json"

    benchmark_raw: Dict[str, Any] = {}
    benchmark_summary: Dict[str, Any] = {}

    if os.path.exists(raw_path):
        try:
            with open(raw_path) as f:
                benchmark_raw = json.load(f)
        except Exception:
            benchmark_raw = {}

    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                benchmark_summary = json.load(f)
        except Exception:
            benchmark_summary = {}

    print("=" * 90)
    print("SPRINT 3 TASK 3.2: MULTI-SUBSET MULTI-SEED BENCHMARK (HIGH-THROUGHPUT)")
    print(f"Random Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Subsets ({len(SUBSETS)}): {SUBSETS}")
    print("=" * 90)

    for subset in SUBSETS:
        benchmark_raw.setdefault(subset, {})
        benchmark_summary.setdefault(subset, {})

        # Check if all models already completed for this subset
        all_done = all(m in benchmark_summary[subset] for m in MODELS)
        if all_done:
            print(f"\n[Checkpoint] All 4 models already completed for {subset}. Skipping.")
            continue

        print(f"\n" + "#" * 90)
        print(f"### PROCESSING SUBSET: {subset} ###")
        print("#" * 90)

        t0 = time.time()
        print(f"[{subset}] Loading and preprocessing data with regime clustering & zero-pad...")
        data = load_and_preprocess_subset(data_dir="data/raw", subset=subset, window_size=30, rul_max=125.0)
        prep_time = time.time() - t0
        print(f"[{subset}] Preprocessed in {prep_time:.1f}s | Retained Sensors: {len(data['retained_sensors'])} | Train 3D: {data['X_train_3d'].shape} | Test Engines: {data['X_test_3d'].shape[0]}")

        batch_sz = 256 if subset in ["FD002", "FD004"] else 128
        epochs = 16

        for m in MODELS:
            disp_name = MODEL_DISPLAY_NAMES[m]
            if m in benchmark_summary[subset]:
                print(f"[{subset}] {disp_name} already benchmarked. Skipping.")
                continue

            print(f"\n--> Benchmarking {disp_name} across {len(SEEDS)} seeds on {subset} (batch_size={batch_sz}, epochs={epochs})...")
            m_runs: List[Dict[str, Any]] = []

            for seed in SEEDS:
                res = train_and_evaluate_model(
                    model_name=m,
                    data=data,
                    seed=seed,
                    epochs=epochs,
                    batch_size=batch_sz,
                    rul_max=125.0,
                    verbose=True,
                )
                m_runs.append({
                    "seed": seed,
                    "val_rmse": res["val_metrics"]["rmse"],
                    "val_nasa": res["val_metrics"]["nasa_score"],
                    "test_rmse": res["test_metrics"]["rmse"],
                    "test_nasa": res["test_metrics"]["nasa_score"],
                    "test_mae": res["test_metrics"]["mae"],
                    "late_pct": res["test_metrics"]["late_pct"],
                    "train_time": res["train_time"],
                })

            test_rmses = [r["test_rmse"] for r in m_runs]
            test_nasas = [r["test_nasa"] for r in m_runs]
            test_maes = [r["test_mae"] for r in m_runs]
            late_pcts = [r["late_pct"] for r in m_runs]
            val_rmses = [r["val_rmse"] for r in m_runs]
            val_nasas = [r["val_nasa"] for r in m_runs]
            train_times = [r["train_time"] for r in m_runs]

            summary_entry = {
                "model_name": disp_name,
                "subset": subset,
                "n_seeds": len(SEEDS),
                "test_rmse_mean": float(np.round(np.mean(test_rmses), 2)),
                "test_rmse_std": float(np.round(np.std(test_rmses), 2)),
                "test_nasa_mean": float(np.round(np.mean(test_nasas), 1)),
                "test_nasa_std": float(np.round(np.std(test_nasas), 1)),
                "test_mae_mean": float(np.round(np.mean(test_maes), 2)),
                "test_mae_std": float(np.round(np.std(test_maes), 2)),
                "late_pct_mean": float(np.round(np.mean(late_pcts), 1)),
                "late_pct_std": float(np.round(np.std(late_pcts), 1)),
                "val_rmse_mean": float(np.round(np.mean(val_rmses), 2)),
                "val_rmse_std": float(np.round(np.std(val_rmses), 2)),
                "val_nasa_mean": float(np.round(np.mean(val_nasas), 1)),
                "val_nasa_std": float(np.round(np.std(val_nasas), 1)),
                "train_time_mean": float(np.round(np.mean(train_times), 1)),
            }

            benchmark_raw[subset][m] = m_runs
            benchmark_summary[subset][m] = summary_entry

            # Save incremental checkpoints
            with open(raw_path, "w") as f:
                json.dump(benchmark_raw, f, indent=2)
            with open(summary_path, "w") as f:
                json.dump(benchmark_summary, f, indent=2)

            print(f"    => {disp_name} Summary: Test RMSE = {summary_entry['test_rmse_mean']:.2f} ± {summary_entry['test_rmse_std']:.2f} | NASA = {summary_entry['test_nasa_mean']:.1f} ± {summary_entry['test_nasa_std']:.1f} | Late% = {summary_entry['late_pct_mean']:.1f}% | Time = {summary_entry['train_time_mean']:.1f}s")

    # Print full table
    print("\n" + "=" * 105)
    print("FINAL SPRINT 3 MULTI-SUBSET PROGNOSTIC BENCHMARK (MEAN ± STD OVER 5 SEEDS)")
    print("=" * 105)
    print(f"{'Subset':<8} {'Architecture':<22} {'Test RMSE (Cycles)':<22} {'NASA Score':<22} {'Late Error %':<16} {'Avg Time':<10}")
    print("-" * 105)
    for subset in SUBSETS:
        for m in MODELS:
            s = benchmark_summary[subset][m]
            rmse_str = f"{s['test_rmse_mean']:.2f} ± {s['test_rmse_std']:.2f}"
            nasa_str = f"{s['test_nasa_mean']:.1f} ± {s['test_nasa_std']:.1f}"
            late_str = f"{s['late_pct_mean']:.1f} ± {s['late_pct_std']:.1f}%"
            time_str = f"{s['train_time_mean']:.1f}s"
            print(f"{subset:<8} {s['model_name']:<22} {rmse_str:<22} {nasa_str:<22} {late_str:<16} {time_str:<10}")
        print("-" * 105)
    print("=" * 105 + "\n")


if __name__ == "__main__":
    run_benchmark()
