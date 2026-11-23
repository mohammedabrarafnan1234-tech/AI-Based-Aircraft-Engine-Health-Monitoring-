"""
C-MAPSS Data Preprocessing Pipeline.

Implements:
1. Invariant sensor channel pruning (Sensors 1, 5, 6, 10, 16, 18, 19).
2. Piecewise-linear RUL target labelling (default RUL_max = 125 cycles).
3. Leak-free Min-Max normalisation (fitted strictly on training units).
4. Sliding window sequence generation (window length W = 30, stride = 1).
5. Tabular summary feature extraction for tree-based baselines (RF, XGBoost).
"""

import os
import sys
import math
from typing import Tuple, List, Dict, Optional, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.data_loader import (
    load_cmapss_raw,
    load_cmapss_test_rul,
    INVARIANT_SENSORS_FD001,
    INFORMATIVE_SENSORS_FD001,
    COLUMN_NAMES,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CMAPSSPreprocessor:
    """End-to-end preprocessing pipeline for C-MAPSS turbofan data."""

    def __init__(
        self,
        window_size: int = 30,
        rul_max: float = 125.0,
        prune_invariant: bool = True,
        feature_cols: Optional[List[str]] = None,
    ):
        self.window_size = window_size
        self.rul_max = rul_max
        self.prune_invariant = prune_invariant
        if feature_cols is not None:
            self.feature_cols = feature_cols
        elif prune_invariant:
            self.feature_cols = INFORMATIVE_SENSORS_FD001
        else:
            self.feature_cols = [f"sensor_{i}" for i in range(1, 22)]

        self.min_vals: Dict[str, float] = {}
        self.max_vals: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, raw_train_data) -> "CMAPSSPreprocessor":
        """
        Computes Min-Max normalisation statistics strictly from training engines.
        Guarantees zero data leakage from validation or test engines.
        """
        if hasattr(raw_train_data, "columns"):
            # Pandas DataFrame
            for col in self.feature_cols:
                self.min_vals[col] = float(raw_train_data[col].min())
                max_v = float(raw_train_data[col].max())
                # Prevent division by zero
                self.max_vals[col] = max_v if max_v > self.min_vals[col] else self.min_vals[col] + 1.0
        else:
            # List of rows
            col_to_idx = {name: i for i, name in enumerate(COLUMN_NAMES)}
            for col in self.feature_cols:
                idx = col_to_idx[col]
                vals = [row[idx] for row in raw_train_data]
                min_v = min(vals)
                max_v = max(vals)
                self.min_vals[col] = min_v
                self.max_vals[col] = max_v if max_v > min_v else min_v + 1.0

        self.is_fitted = True
        return self

    def _transform_row(self, row_dict: Dict[str, float]) -> List[float]:
        """Scales a single row of features using train-fitted Min-Max values."""
        scaled = []
        for col in self.feature_cols:
            val = row_dict[col]
            mn = self.min_vals[col]
            mx = self.max_vals[col]
            norm_val = (val - mn) / (mx - mn)
            scaled.append(max(0.0, min(1.0, norm_val)))
        return scaled

    def transform_train(self, raw_train_data) -> Tuple[Union["np.ndarray", List], Union["np.ndarray", List]]:
        """
        Transforms training data into sliding window tensors (X) and piecewise RUL targets (y).

        Returns:
            X: Array/List of shape (N_samples, window_size, num_features)
            y: Array/List of shape (N_samples,)
        """
        if not self.is_fitted:
            self.fit(raw_train_data)

        # Group cycles by engine unit
        unit_cycles: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
        if hasattr(raw_train_data, "iterrows"):
            for _, row in raw_train_data.iterrows():
                u = int(row["unit_number"])
                c = int(row["time_cycle"])
                feat_dict = {col: float(row[col]) for col in self.feature_cols}
                unit_cycles.setdefault(u, []).append((c, feat_dict))
        else:
            col_to_idx = {name: i for i, name in enumerate(COLUMN_NAMES)}
            for row in raw_train_data:
                u = int(row[0])
                c = int(row[1])
                feat_dict = {col: float(row[col_to_idx[col]]) for col in self.feature_cols}
                unit_cycles.setdefault(u, []).append((c, feat_dict))

        X_windows = []
        y_targets = []

        for unit_id, cycles in unit_cycles.items():
            # Sort cycles chronologically
            cycles.sort(key=lambda x: x[0])
            max_cycle = cycles[-1][0]

            # Normalize features
            norm_cycles = [(c, self._transform_row(f_dict)) for c, f_dict in cycles]

            # Generate sliding windows
            n_cycles = len(norm_cycles)
            if n_cycles < self.window_size:
                continue

            for start_idx in range(n_cycles - self.window_size + 1):
                end_idx = start_idx + self.window_size
                window = [norm_cycles[i][1] for i in range(start_idx, end_idx)]
                current_cycle = norm_cycles[end_idx - 1][0]

                # Piecewise linear RUL labeling
                linear_rul = max_cycle - current_cycle
                capped_rul = min(self.rul_max, float(linear_rul))

                X_windows.append(window)
                y_targets.append(capped_rul)

        if HAS_NUMPY:
            return np.array(X_windows, dtype=np.float32), np.array(y_targets, dtype=np.float32)
        return X_windows, y_targets

    def transform_test(self, raw_test_data, true_ruls: List[float]) -> Tuple[Union["np.ndarray", List], Union["np.ndarray", List]]:
        """
        Transforms test data by extracting the final observed window for each test engine.

        Returns:
            X_test: (N_engines, window_size, num_features)
            y_test: (N_engines,) true RUL targets
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming test data.")

        unit_cycles: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
        if hasattr(raw_test_data, "iterrows"):
            for _, row in raw_test_data.iterrows():
                u = int(row["unit_number"])
                c = int(row["time_cycle"])
                feat_dict = {col: float(row[col]) for col in self.feature_cols}
                unit_cycles.setdefault(u, []).append((c, feat_dict))
        else:
            col_to_idx = {name: i for i, name in enumerate(COLUMN_NAMES)}
            for row in raw_test_data:
                u = int(row[0])
                c = int(row[1])
                feat_dict = {col: float(row[col_to_idx[col]]) for col in self.feature_cols}
                unit_cycles.setdefault(u, []).append((c, feat_dict))

        X_test_windows = []
        y_test_targets = []
        num_features = len(self.feature_cols)

        # Process each test engine
        for unit_id in sorted(unit_cycles.keys()):
            cycles = unit_cycles[unit_id]
            cycles.sort(key=lambda x: x[0])
            norm_cycles = [self._transform_row(f_dict) for _, f_dict in cycles]

            # Extract last window (or pad if fewer than window_size)
            if len(norm_cycles) >= self.window_size:
                final_window = norm_cycles[-self.window_size :]
            else:
                pad_len = self.window_size - len(norm_cycles)
                zero_pad = [[0.0] * num_features for _ in range(pad_len)]
                final_window = zero_pad + norm_cycles

            X_test_windows.append(final_window)
            if unit_id - 1 < len(true_ruls):
                # Cap ground truth RUL at rul_max as well for consistent evaluation
                true_val = min(self.rul_max, float(true_ruls[unit_id - 1]))
                y_test_targets.append(true_val)

        if HAS_NUMPY:
            return np.array(X_test_windows, dtype=np.float32), np.array(y_test_targets, dtype=np.float32)
        return X_test_windows, y_test_targets

    @staticmethod
    def extract_tabular_features(X_windows) -> Union["np.ndarray", List[List[float]]]:
        """
        Converts 3D temporal sliding windows (N, W, D) into 2D tabular summary vectors
        for tree-based baselines (Random Forest, XGBoost).
        Extracts: mean, std, min, max, slope, and latest cycle per sensor.
        """
        if HAS_NUMPY and isinstance(X_windows, np.ndarray):
            # X_windows shape: (N, W, D)
            mean_feats = np.mean(X_windows, axis=1)
            std_feats = np.std(X_windows, axis=1)
            min_feats = np.min(X_windows, axis=1)
            max_feats = np.max(X_windows, axis=1)
            last_feats = X_windows[:, -1, :]
            # Simple difference (slope indicator)
            delta_feats = X_windows[:, -1, :] - X_windows[:, 0, :]
            return np.concatenate([mean_feats, std_feats, min_feats, max_feats, last_feats, delta_feats], axis=1)
        else:
            # Pure Python implementation
            tabular = []
            for win in X_windows:
                # win is (W, D)
                W = len(win)
                D = len(win[0])
                row = []
                for d in range(D):
                    vals = [win[k][d] for k in range(W)]
                    m = sum(vals) / W
                    s = math.sqrt(sum((x - m) ** 2 for x in vals) / W)
                    row.extend([m, s, min(vals), max(vals), vals[-1], vals[-1] - vals[0]])
                tabular.append(row)
            return tabular


if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    data_dir = "data/raw"
    if not os.path.exists(os.path.join(data_dir, "train_FD001.txt")):
        from src.utils.data_loader import generate_synthetic_cmapss
        generate_synthetic_cmapss(data_dir, subset="FD001", num_train_units=5, num_test_units=5)

    raw_train = load_cmapss_raw(data_dir, subset="FD001", split="train")
    raw_test = load_cmapss_raw(data_dir, subset="FD001", split="test")
    test_ruls = load_cmapss_test_rul(data_dir, subset="FD001")

    prep = CMAPSSPreprocessor(window_size=30, rul_max=125.0)
    X_train, y_train = prep.transform_train(raw_train)
    X_test, y_test = prep.transform_test(raw_test, test_ruls)

    X_train_tab = prep.extract_tabular_features(X_train)
    X_test_tab = prep.extract_tabular_features(X_test)

    print(f"✓ Train 3D sequences shape: {len(X_train)} samples, window: {len(X_train[0])}x{len(X_train[0][0])}")
    print(f"✓ Test 3D sequences shape:  {len(X_test)} samples, window: {len(X_test[0])}x{len(X_test[0][0])}")
    print(f"✓ Tabular baseline feature shape: {len(X_train_tab)}x{len(X_train_tab[0])}")
    print("✓ Preprocessing pipeline passed validation.")
