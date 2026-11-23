"""
End-to-End Orchestration Script for Sprint 1 & Sprint 2.

Executes:
1. Data verification and EDA (Sprint 1)
2. Preprocessing: invariant sensor pruning, normalisation, sliding window generation (Sprint 1)
3. Baseline Regressors: Random Forest & XGBoost (Sprint 2)
4. Deep Neural Networks: LSTM & 1D-CNN (Sprint 2)
5. Comparative Benchmarking on FD001 Test Set with RMSE & NASA Score (Sprint 2)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data_loader import load_cmapss_raw, load_cmapss_test_rul
from src.utils.metrics import evaluate_prognostics
from src.eda import run_eda
from src.preprocessing.preprocess import CMAPSSPreprocessor
from src.models.baselines import RandomForestBaseline, XGBoostBaseline
from src.models.deep_models import LSTMPrognosticModel, CNN1DPrognosticModel
from scripts.download_cmapss import download_or_generate


def main():
    parser = argparse.ArgumentParser(description="Run full Capstone Sprint 1 & 2 pipeline.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset subset (FD001-FD004)")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window length")
    parser.add_argument("--rul-max", type=float, default=125.0, help="Piecewise linear RUL cap")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CAPSTONE PROJECT PIPELINE EXECUTION: SPRINT 1 & SPRINT 2")
    print("AI-Based Turbofan Engine Health Monitoring & RUL Estimation")
    print(f"Target Subset: {args.dataset} | Window Size: {args.window_size} | RUL Cap: {args.rul_max}")
    print("=" * 80)

    # 1. Check or generate data
    download_or_generate(data_dir=args.data_dir, subset=args.dataset)

    # 2. Run EDA
    print("\n>>> STAGE 1: EXPLORATORY DATA ANALYSIS (SPRINT 1)")
    run_eda(data_dir=args.data_dir, subset=args.dataset)

    # 3. Preprocess
    print("\n>>> STAGE 2: DATA PREPROCESSING PIPELINE (SPRINT 1)")
    raw_train = load_cmapss_raw(args.data_dir, subset=args.dataset, split="train")
    raw_test = load_cmapss_raw(args.data_dir, subset=args.dataset, split="test")
    test_ruls = load_cmapss_test_rul(args.data_dir, subset=args.dataset)

    preprocessor = CMAPSSPreprocessor(window_size=args.window_size, rul_max=args.rul_max)
    X_train_seq, y_train = preprocessor.transform_train(raw_train)
    X_test_seq, y_test = preprocessor.transform_test(raw_test, test_ruls)

    X_train_tab = preprocessor.extract_tabular_features(X_train_seq)
    X_test_tab = preprocessor.extract_tabular_features(X_test_seq)

    print(f"✓ Preprocessed Training Samples (3D): {len(X_train_seq)} windows of shape ({args.window_size}, {len(preprocessor.feature_cols)})")
    print(f"✓ Preprocessed Test Engines (3D):    {len(X_test_seq)} test engines")
    print(f"✓ Tabular Features Matrix (2D):       {len(X_train_tab)}x{len(X_train_tab[0])}")

    # 4. Baselines
    print("\n>>> STAGE 3: CLASSICAL ML BASELINES (SPRINT 2)")
    print("[1] Fitting Random Forest Regressor...")
    rf = RandomForestBaseline(n_estimators=100)
    rf.fit(X_train_tab, y_train)
    rf_preds = rf.predict(X_test_tab)
    rf_metrics = evaluate_prognostics(y_test, rf_preds)

    print("[2] Fitting XGBoost Regressor...")
    xgb_mod = XGBoostBaseline(n_estimators=150)
    xgb_mod.fit(X_train_tab, y_train)
    xgb_preds = xgb_mod.predict(X_test_tab)
    xgb_metrics = evaluate_prognostics(y_test, xgb_preds)

    # 5. Deep Models
    print("\n>>> STAGE 4: DEEP LEARNING PROGNOSTICS (SPRINT 2)")
    print("[3] Training PyTorch Stacked LSTM...")
    lstm = LSTMPrognosticModel(model_type="lstm", input_dim=len(preprocessor.feature_cols), max_epochs=20)
    lstm.fit(X_train_seq, y_train)
    lstm_preds = lstm.predict(X_test_seq)
    lstm_metrics = evaluate_prognostics(y_test, lstm_preds)

    print("[4] Training PyTorch 1D-CNN...")
    cnn = CNN1DPrognosticModel(model_type="cnn1d", input_dim=len(preprocessor.feature_cols), max_epochs=20)
    cnn.fit(X_train_seq, y_train)
    cnn_preds = cnn.predict(X_test_seq)
    cnn_metrics = evaluate_prognostics(y_test, cnn_preds)

    # 6. Benchmark Summary Table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE (FD001 TEST SET)")
    print("=" * 80)
    print(f"{'Model Architecture':<25} {'RMSE (Cycles)':<15} {'NASA Score':<15} {'Late Pred %':<12}")
    print("-" * 80)
    print(f"{'Random Forest':<25} {rf_metrics['rmse']:<15.2f} {rf_metrics['nasa_score']:<15.1f} {rf_metrics['late_prediction_pct']:<12.1f}%")
    print(f"{'XGBoost':<25} {xgb_metrics['rmse']:<15.2f} {xgb_metrics['nasa_score']:<15.1f} {xgb_metrics['late_prediction_pct']:<12.1f}%")
    print(f"{'PyTorch Stacked LSTM':<25} {lstm_metrics['rmse']:<15.2f} {lstm_metrics['nasa_score']:<15.1f} {lstm_metrics['late_prediction_pct']:<12.1f}%")
    print(f"{'PyTorch 1D-CNN':<25} {cnn_metrics['rmse']:<15.2f} {cnn_metrics['nasa_score']:<15.1f} {cnn_metrics['late_prediction_pct']:<12.1f}%")
    print("=" * 80)
    print("✓ Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
