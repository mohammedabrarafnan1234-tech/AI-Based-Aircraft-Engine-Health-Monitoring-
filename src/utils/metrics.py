"""
Prognostic Evaluation Metrics for Turbofan Remaining Useful Life (RUL) Estimation.

Implements:
1. Root Mean Squared Error (RMSE)
2. Mean Absolute Error (MAE)
3. NASA PHM 2008 Data Challenge Asymmetric Scoring Function
"""

import math
from typing import Dict, Union, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def compute_rmse(y_true: Union[List[float], "np.ndarray"], y_pred: Union[List[float], "np.ndarray"]) -> float:
    """
    Computes Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth Remaining Useful Life targets.
        y_pred: Predicted Remaining Useful Life values.

    Returns:
        float: RMSE in operational cycles.
    """
    if HAS_NUMPY:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")
        return float(np.sqrt(np.mean(np.square(y_p - y_t))))
    else:
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        n = len(y_true)
        mse = sum((p - t) ** 2 for t, p in zip(y_true, y_pred)) / n
        return math.sqrt(mse)


def compute_mae(y_true: Union[List[float], "np.ndarray"], y_pred: Union[List[float], "np.ndarray"]) -> float:
    """Computes Mean Absolute Error (MAE)."""
    if HAS_NUMPY:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        return float(np.mean(np.abs(y_p - y_t)))
    else:
        return sum(abs(p - t) for t, p in zip(y_true, y_pred)) / len(y_true)


def compute_nasa_score(y_true: Union[List[float], "np.ndarray"], y_pred: Union[List[float], "np.ndarray"]) -> float:
    """
    Computes the official NASA PHM 2008 Asymmetric Scoring Function.

    d_i = y_pred_i - y_true_i
    s_i = exp(-d_i / 13) - 1   if d_i < 0  (Early prediction: conservative/safe)
    s_i = exp(d_i / 10) - 1    if d_i >= 0 (Late prediction: dangerous/unplanned failure)

    Args:
        y_true: Ground truth Remaining Useful Life targets.
        y_pred: Predicted Remaining Useful Life values.

    Returns:
        float: Total cumulative NASA asymmetric score (lower is better).
    """
    if HAS_NUMPY:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        if len(y_t) != len(y_p):
            raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")
        d = y_p - y_t
        scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
        return float(np.sum(scores))
    else:
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        total_score = 0.0
        for t, p in zip(y_true, y_pred):
            d = p - t
            if d < 0:
                s = math.exp(-d / 13.0) - 1.0
            else:
                s = math.exp(d / 10.0) - 1.0
            total_score += s
        return total_score


def evaluate_prognostics(
    y_true: Union[List[float], "np.ndarray"],
    y_pred: Union[List[float], "np.ndarray"],
) -> Dict[str, float]:
    """
    Computes a comprehensive suite of prognostic metrics including directional residual analysis.

    Returns:
        Dict containing:
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'nasa_score': NASA PHM 2008 Asymmetric Score
        - 'early_prediction_pct': Percentage of predictions that were early (d < 0)
        - 'late_prediction_pct': Percentage of predictions that were late (d >= 0)
        - 'max_early_error': Largest early error (negative residual)
        - 'max_late_error': Largest late error (positive residual)
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    rmse_val = compute_rmse(y_true, y_pred)
    mae_val = compute_mae(y_true, y_pred)
    nasa_val = compute_nasa_score(y_true, y_pred)

    if HAS_NUMPY:
        y_t = np.asarray(y_true, dtype=np.float64)
        y_p = np.asarray(y_pred, dtype=np.float64)
        residuals = y_p - y_t
        early_mask = residuals < 0
        late_mask = residuals >= 0
        early_pct = float(np.mean(early_mask) * 100.0)
        late_pct = float(np.mean(late_mask) * 100.0)
        max_early = float(np.min(residuals)) if np.any(early_mask) else 0.0
        max_late = float(np.max(residuals)) if np.any(late_mask) else 0.0
    else:
        residuals = [p - t for t, p in zip(y_true, y_pred)]
        early_count = sum(1 for r in residuals if r < 0)
        early_pct = (early_count / n) * 100.0
        late_pct = 100.0 - early_pct
        max_early = min(residuals)
        max_late = max(residuals)

    return {
        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        "nasa_score": round(nasa_val, 2),
        "early_prediction_pct": round(early_pct, 2),
        "late_prediction_pct": round(late_pct, 2),
        "max_early_error": round(max_early, 2),
        "max_late_error": round(max_late, 2),
    }


if __name__ == "__main__":
    print("Testing Prognostic Metrics Suite...")
    y_test_true = [112.0, 98.0, 69.0, 82.0, 91.0]
    y_test_pred_early = [102.0, 88.0, 59.0, 72.0, 81.0]  # All 10 cycles early
    y_test_pred_late = [122.0, 108.0, 79.0, 92.0, 101.0]  # All 10 cycles late

    metrics_early = evaluate_prognostics(y_test_true, y_test_pred_early)
    metrics_late = evaluate_prognostics(y_test_true, y_test_pred_late)

    print("\n[Early Prediction Scenario (-10 cycles each)]:")
    for k, v in metrics_early.items():
        print(f"  {k}: {v}")

    print("\n[Late Prediction Scenario (+10 cycles each)]:")
    for k, v in metrics_late.items():
        print(f"  {k}: {v}")

    print(f"\nNASA Score Ratio (Late / Early for same RMSE=10): {metrics_late['nasa_score'] / metrics_early['nasa_score']:.2f}x")
    assert metrics_early["rmse"] == metrics_late["rmse"] == 10.0
    assert metrics_late["nasa_score"] > metrics_early["nasa_score"]
    print("✓ All prognostic metric assertions passed successfully.")
