"""
Multi-Condition C-MAPSS Preprocessing and Operating Regime Normalization.

Supports FD001, FD002, FD003, and FD004:
1. Dynamic sensor variance audit per subset (does NOT blindly drop FD001 sensors on FD002/FD004).
2. Operating regime clustering via KMeans (6 clusters) on operational settings for FD002 and FD004.
3. Within-regime Min-Max normalization fitted strictly on training engines (zero data leakage).
4. Sliding window sequence generation with left zero-padding for short sequences (< window_size).
5. Tabular summary feature extraction (6 statistics per sensor: mean, std, min, max, last, delta).
6. Strict engine-level train/validation splitting (80% train engines, 20% validation engines).
"""

import os
import sys
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

COLUMN_NAMES = [
    "unit_number", "time_cycle", "setting_1", "setting_2", "setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

ALL_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]


class MultiConditionPreprocessor:
    """
    Robust preprocessing pipeline supporting single-regime (FD001, FD003)
    and multi-regime (FD002, FD004) NASA C-MAPSS datasets.
    """

    def __init__(
        self,
        subset: str = "FD001",
        window_size: int = 30,
        rul_max: float = 125.0,
        n_clusters: int = 6,
        val_ratio: float = 0.20,
        random_state: int = 42,
    ):
        self.subset = subset.upper()
        self.window_size = window_size
        self.rul_max = rul_max
        self.n_clusters = n_clusters
        self.val_ratio = val_ratio
        self.random_state = random_state

        self.is_multi_regime = self.subset in ["FD002", "FD004"]
        self.kmeans: Optional[KMeans] = None
        self.retained_sensors: List[str] = []
        self.dropped_sensors: List[str] = []
        
        # Scaling parameters: cluster_id -> {sensor: (min_val, max_val)}
        self.cluster_scalers: Dict[int, Dict[str, Tuple[float, float]]] = {}
        # Single-regime global scaler
        self.global_scalers: Dict[str, Tuple[float, float]] = {}
        
        self.train_units: List[int] = []
        self.val_units: List[int] = []
        self.is_fitted: bool = False

    def _determine_sensor_subsets(self, df_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Dynamically determine informative vs. invariant sensor channels.
        In FD001: 7 sensors constant/near-constant.
        In FD003: 5 sensors constant.
        In FD002/FD004: drops sensors with zero/near-zero variance across all operating clusters.
        """
        dropped = []
        retained = []

        if not self.is_multi_regime:
            # Single condition: global standard deviation check
            for s in ALL_SENSORS:
                std_val = df_train[s].std()
                # sensor_6 in FD001 has std ~ 0.0013 and only 2 discrete values (21.60 and 21.61)
                if std_val < 1e-3 or (self.subset == "FD001" and s == "sensor_6"):
                    dropped.append(s)
                else:
                    retained.append(s)
        else:
            # Multi-regime: Check variance within clusters
            # Sensors 1, 5, 18, 19 are physical constants in C-MAPSS across all regimes
            # Let's inspect std within each KMeans cluster
            temp_km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            clusters = temp_km.fit_predict(df_train[SETTING_COLS])
            
            for s in ALL_SENSORS:
                cluster_stds = [df_train.loc[clusters == k, s].std() for k in range(self.n_clusters)]
                # If sensor has zero variance in all clusters, it conveys zero degradation information
                if all(np.isnan(st) or st < 1e-4 for st in cluster_stds):
                    dropped.append(s)
                else:
                    retained.append(s)

        return retained, dropped

    def fit(self, df_train: pd.DataFrame, train_units: Optional[List[int]] = None) -> "MultiConditionPreprocessor":
        """
        Fits normalization parameters strictly on training engines.
        """
        all_units = sorted(df_train["unit_number"].unique())
        n_units = len(all_units)
        
        if train_units is None:
            n_val = max(1, int(round(n_units * self.val_ratio)))
            n_train = n_units - n_val
            # Deterministic split: first n_train engines for training, last n_val for validation
            self.train_units = all_units[:n_train]
            self.val_units = all_units[n_train:]
        else:
            self.train_units = train_units
            self.val_units = [u for u in all_units if u not in self.train_units]

        df_train_fit = df_train[df_train["unit_number"].isin(self.train_units)].copy()

        # Step 1: Detect informative sensors
        self.retained_sensors, self.dropped_sensors = self._determine_sensor_subsets(df_train_fit)

        # Step 2: Fit scalers
        if self.is_multi_regime:
            # Fit KMeans on 3 settings
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            self.kmeans.fit(df_train_fit[SETTING_COLS])
            clusters = self.kmeans.predict(df_train_fit[SETTING_COLS])
            df_train_fit["cluster"] = clusters

            # Fit min-max per cluster for each retained sensor
            for k in range(self.n_clusters):
                cluster_mask = df_train_fit["cluster"] == k
                c_data = df_train_fit[cluster_mask]
                self.cluster_scalers[k] = {}
                for s in self.retained_sensors:
                    if len(c_data) > 0:
                        min_v = float(c_data[s].min())
                        max_v = float(c_data[s].max())
                    else:
                        min_v, max_v = 0.0, 1.0
                    if max_v - min_v < 1e-6:
                        max_v = min_v + 1.0
                    self.cluster_scalers[k][s] = (min_v, max_v)
        else:
            # Single-condition global min-max
            for s in self.retained_sensors:
                min_v = float(df_train_fit[s].min())
                max_v = float(df_train_fit[s].max())
                if max_v - min_v < 1e-6:
                    max_v = min_v + 1.0
                self.global_scalers[s] = (min_v, max_v)

        self.is_fitted = True
        return self

    def normalize(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned normalization to input dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before normalizing.")

        df = df_input.copy()
        for s in self.retained_sensors:
            df[s] = df[s].astype(np.float32)

        if self.is_multi_regime:
            clusters = self.kmeans.predict(df[SETTING_COLS])
            df["cluster"] = clusters
            
            # Normalize per cluster
            for k in range(self.n_clusters):
                mask = df["cluster"] == k
                if not mask.any():
                    continue
                for s in self.retained_sensors:
                    min_v, max_v = self.cluster_scalers[k][s]
                    scaled_vals = (df.loc[mask, s].to_numpy(dtype=np.float32) - min_v) / (max_v - min_v)
                    df.loc[mask, s] = np.clip(scaled_vals, 0.0, 1.0)
        else:
            for s in self.retained_sensors:
                min_v, max_v = self.global_scalers[s]
                scaled_vals = (df[s].to_numpy(dtype=np.float32) - min_v) / (max_v - min_v)
                df[s] = np.clip(scaled_vals, 0.0, 1.0)

        return df

    def generate_windows(
        self,
        df_norm: pd.DataFrame,
        unit_list: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sliding window sequences and piecewise RUL targets.
        Shape: X (N_samples, window_size, n_sensors), y (N_samples,)
        """
        X_wins: List[np.ndarray] = []
        y_wins: List[float] = []

        for u in unit_list:
            u_df = df_norm[df_norm["unit_number"] == u].sort_values("time_cycle")
            max_c = u_df["time_cycle"].max()
            feat_vals = u_df[self.retained_sensors].values
            cycles = u_df["time_cycle"].values
            n_c = len(cycles)

            if n_c < self.window_size:
                # Left zero-pad if sequence is shorter than window_size
                pad_len = self.window_size - n_c
                pad = np.zeros((pad_len, len(self.retained_sensors)), dtype=np.float32)
                padded_feats = np.vstack([pad, feat_vals])
                X_wins.append(padded_feats)
                rul_linear = max_c - cycles[-1]
                y_wins.append(min(self.rul_max, float(rul_linear)))
                continue

            for start in range(n_c - self.window_size + 1):
                end = start + self.window_size
                X_wins.append(feat_vals[start:end])
                current_cycle = cycles[end - 1]
                rul_linear = max_c - current_cycle
                y_wins.append(min(self.rul_max, float(rul_linear)))

        return np.array(X_wins, dtype=np.float32), np.array(y_wins, dtype=np.float32)

    def generate_test_windows(
        self,
        df_test_norm: pd.DataFrame,
    ) -> np.ndarray:
        """
        Extracts the last window_size flight cycles for each test engine.
        Applies left zero-padding for engines with fewer cycles than window_size.
        Returns: X_test of shape (N_test_engines, window_size, n_sensors)
        """
        test_units = sorted(df_test_norm["unit_number"].unique())
        X_test: List[np.ndarray] = []

        for u in test_units:
            u_df = df_test_norm[df_test_norm["unit_number"] == u].sort_values("time_cycle")
            feat_vals = u_df[self.retained_sensors].values
            n_c = len(feat_vals)

            if n_c >= self.window_size:
                win = feat_vals[-self.window_size:]
            else:
                # Left zero-padding for short sequences (< 30 cycles)
                pad_len = self.window_size - n_c
                pad = np.zeros((pad_len, len(self.retained_sensors)), dtype=np.float32)
                win = np.vstack([pad, feat_vals])
            X_test.append(win)

        return np.array(X_test, dtype=np.float32)

    @staticmethod
    def extract_tabular_features(X_windows: np.ndarray) -> np.ndarray:
        """
        Extracts 6 statistical features per sensor channel:
        [mean, std, min, max, last_val, delta (last - first)]
        Input: (N, window_size, n_channels)
        Output: (N, n_channels * 6)
        """
        means = np.mean(X_windows, axis=1)
        stds = np.std(X_windows, axis=1)
        mins = np.min(X_windows, axis=1)
        maxs = np.max(X_windows, axis=1)
        lasts = X_windows[:, -1, :]
        deltas = X_windows[:, -1, :] - X_windows[:, 0, :]
        return np.concatenate([means, stds, mins, maxs, lasts, deltas], axis=1).astype(np.float32)


def load_and_preprocess_subset(
    data_dir: str = "data/raw",
    subset: str = "FD001",
    window_size: int = 30,
    rul_max: float = 125.0,
    random_state: int = 42,
) -> Dict[str, Union[np.ndarray, List, MultiConditionPreprocessor]]:
    """
    Convenience function to load raw data, run multi-condition preprocessing,
    and return complete datasets ready for training and evaluation.
    """
    train_file = os.path.join(data_dir, f"train_{subset}.txt")
    test_file = os.path.join(data_dir, f"test_{subset}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{subset}.txt")

    df_train = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df_test = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    y_test_raw = np.loadtxt(rul_file, dtype=np.float32)
    y_test_capped = np.clip(y_test_raw, 0.0, rul_max).astype(np.float32)

    preprocessor = MultiConditionPreprocessor(
        subset=subset,
        window_size=window_size,
        rul_max=rul_max,
        n_clusters=6,
        val_ratio=0.20,
        random_state=random_state,
    )
    preprocessor.fit(df_train)

    df_train_norm = preprocessor.normalize(df_train)
    df_test_norm = preprocessor.normalize(df_test)

    # Generate 3D sequences
    X_train_3d, y_train_3d = preprocessor.generate_windows(df_train_norm, preprocessor.train_units)
    X_val_3d, y_val_3d = preprocessor.generate_windows(df_train_norm, preprocessor.val_units)
    X_train_full_3d, y_train_full_3d = preprocessor.generate_windows(
        df_train_norm, sorted(df_train["unit_number"].unique())
    )
    X_test_3d = preprocessor.generate_test_windows(df_test_norm)

    # Generate tabular features
    X_train_tab = preprocessor.extract_tabular_features(X_train_3d)
    X_val_tab = preprocessor.extract_tabular_features(X_val_3d)
    X_train_full_tab = preprocessor.extract_tabular_features(X_train_full_3d)
    X_test_tab = preprocessor.extract_tabular_features(X_test_3d)

    return {
        "subset": subset,
        "preprocessor": preprocessor,
        "retained_sensors": preprocessor.retained_sensors,
        "dropped_sensors": preprocessor.dropped_sensors,
        "train_units": preprocessor.train_units,
        "val_units": preprocessor.val_units,
        # 3D sequences
        "X_train_3d": X_train_3d,
        "y_train_3d": y_train_3d,
        "X_val_3d": X_val_3d,
        "y_val_3d": y_val_3d,
        "X_train_full_3d": X_train_full_3d,
        "y_train_full_3d": y_train_full_3d,
        "X_test_3d": X_test_3d,
        "y_test": y_test_capped,
        # Tabular features
        "X_train_tab": X_train_tab,
        "X_val_tab": X_val_tab,
        "X_train_full_tab": X_train_full_tab,
        "X_test_tab": X_test_tab,
    }
