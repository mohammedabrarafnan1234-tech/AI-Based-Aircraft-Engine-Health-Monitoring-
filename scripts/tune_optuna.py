#!/usr/bin/env python3
"""
Sprint 3 Task 3.1: Hyperparameter Optimization with Optuna.

Rules strictly adhered to:
1. Optimization objective is evaluated on VALIDATION ENGINES ONLY (zero leakage from test set).
2. Test evaluation is conducted ONCE at the end for the final winning configuration.
3. Search Space:
   - LSTM:
     * hidden_dim1: [32, 64, 128]
     * hidden_dim2: [16, 32, 64]
     * dropout: [0.1, 0.4]
     * learning_rate: [1e-4, 1e-2] (log scale)
     * window_size: [20, 30, 40]
   - 1D-CNN:
     * filters1: [16, 32, 64]
     * filters2: [32, 64, 128]
     * filters3: [32, 64, 128]
     * kernel_size1: [3, 5, 7]
     * kernel_size2: [3, 5]
     * kernel_size3: [3, 5]
     * dropout: [0.1, 0.4]
     * learning_rate: [1e-4, 1e-2] (log scale)
4. Saves optimization study, best parameters, and comparison table into results/optuna_tuning_results.json.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any
import numpy as np
import optuna

# Prevent verbose Optuna logging clutter
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing.multicondition_preprocess import load_and_preprocess_subset
from src.models.unified_trainer import train_and_evaluate_model, set_seed


def optimize_lstm(subset: str = "FD001", n_trials: int = 35) -> Dict[str, Any]:
    print(f"\n" + "=" * 80)
    print(f"STARTING OPTUNA TUNING: PYTORCH STACKED LSTM ({subset}, {n_trials} Trials)")
    print("Optimization Objective: Minimize Validation RMSE (Zero Test Exposure)")
    print("=" * 80)

    # Pre-cache datasets for different window sizes
    cached_data = {}
    for w in [20, 30, 40]:
        cached_data[w] = load_and_preprocess_subset(data_dir="data/raw", subset=subset, window_size=w)

    def objective(trial: optuna.Trial) -> float:
        w = trial.suggest_categorical("window_size", [20, 30, 40])
        h1 = trial.suggest_categorical("hidden_dim1", [32, 64, 128])
        h2 = trial.suggest_categorical("hidden_dim2", [16, 32, 64])
        drop = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        params = {
            "hidden_dim1": h1,
            "hidden_dim2": h2,
            "dropout": drop,
            "learning_rate": lr,
        }

        # Train on train engines, validate on validation engines
        res = train_and_evaluate_model(
            model_name="lstm",
            data=cached_data[w],
            seed=42,
            params=params,
            epochs=15,
            batch_size=128,
            verbose=False,
        )
        return float(res["val_metrics"]["rmse"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[LSTM Tuning Completed]")
    print(f"Best Validation RMSE: {study.best_value:.3f}")
    print(f"Best Hyperparameters: {study.best_params}")

    # Now evaluate best model ONCE on test set
    best_w = study.best_params["window_size"]
    best_data = cached_data[best_w]
    best_params = {
        "hidden_dim1": study.best_params["hidden_dim1"],
        "hidden_dim2": study.best_params["hidden_dim2"],
        "dropout": study.best_params["dropout"],
        "learning_rate": study.best_params["learning_rate"],
    }
    final_test_res = train_and_evaluate_model(
        model_name="lstm",
        data=best_data,
        seed=42,
        params=best_params,
        epochs=35,
        batch_size=128,
        verbose=False,
    )

    print(f"-> Final Single Test Evaluation for Tuned LSTM:")
    print(f"   Test RMSE: {final_test_res['test_metrics']['rmse']:.2f}")
    print(f"   Test NASA Score: {final_test_res['test_metrics']['nasa_score']:.1f}")

    return {
        "model": "Stacked LSTM",
        "subset": subset,
        "n_trials": n_trials,
        "best_val_rmse": float(study.best_value),
        "best_params": study.best_params,
        "final_test_metrics": final_test_res["test_metrics"],
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }


def optimize_cnn(subset: str = "FD001", n_trials: int = 35) -> Dict[str, Any]:
    print(f"\n" + "=" * 80)
    print(f"STARTING OPTUNA TUNING: PYTORCH 1D-CNN ({subset}, {n_trials} Trials)")
    print("Optimization Objective: Minimize Validation RMSE (Zero Test Exposure)")
    print("=" * 80)

    # Pre-cache datasets for different window sizes
    cached_data = {}
    for w in [20, 30, 40]:
        cached_data[w] = load_and_preprocess_subset(data_dir="data/raw", subset=subset, window_size=w)

    def objective(trial: optuna.Trial) -> float:
        w = trial.suggest_categorical("window_size", [20, 30, 40])
        f1 = trial.suggest_categorical("filters1", [16, 32, 64])
        f2 = trial.suggest_categorical("filters2", [32, 64, 128])
        f3 = trial.suggest_categorical("filters3", [32, 64, 128])
        k1 = trial.suggest_categorical("kernel_size1", [3, 5, 7])
        k2 = trial.suggest_categorical("kernel_size2", [3, 5])
        k3 = trial.suggest_categorical("kernel_size3", [3, 5])
        drop = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        params = {
            "filters1": f1,
            "filters2": f2,
            "filters3": f3,
            "kernel_size1": k1,
            "kernel_size2": k2,
            "kernel_size3": k3,
            "dropout": drop,
            "learning_rate": lr,
        }

        res = train_and_evaluate_model(
            model_name="cnn",
            data=cached_data[w],
            seed=42,
            params=params,
            epochs=15,
            batch_size=128,
            verbose=False,
        )
        return float(res["val_metrics"]["rmse"])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[1D-CNN Tuning Completed]")
    print(f"Best Validation RMSE: {study.best_value:.3f}")
    print(f"Best Hyperparameters: {study.best_params}")

    best_w = study.best_params["window_size"]
    best_data = cached_data[best_w]
    best_params = {k: v for k, v in study.best_params.items() if k != "window_size"}

    final_test_res = train_and_evaluate_model(
        model_name="cnn",
        data=best_data,
        seed=42,
        params=best_params,
        epochs=35,
        batch_size=128,
        verbose=False,
    )

    print(f"-> Final Single Test Evaluation for Tuned 1D-CNN:")
    print(f"   Test RMSE: {final_test_res['test_metrics']['rmse']:.2f}")
    print(f"   Test NASA Score: {final_test_res['test_metrics']['nasa_score']:.1f}")

    return {
        "model": "1D-CNN",
        "subset": subset,
        "n_trials": n_trials,
        "best_val_rmse": float(study.best_value),
        "best_params": study.best_params,
        "final_test_metrics": final_test_res["test_metrics"],
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    
    # Run LSTM tuning
    lstm_res = optimize_lstm(subset="FD001", n_trials=30)
    results["lstm"] = lstm_res

    # Run CNN tuning
    cnn_res = optimize_cnn(subset="FD001", n_trials=30)
    results["cnn"] = cnn_res

    out_file = "results/optuna_tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved all Optuna tuning results to {out_file}")


if __name__ == "__main__":
    main()
