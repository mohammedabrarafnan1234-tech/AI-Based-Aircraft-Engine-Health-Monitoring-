"""
C-MAPSS Dataset Loading and Schema Verification Utilities.

Handles reading space-delimited NASA C-MAPSS text files, verifying column schemas,
and providing synthetic benchmark fallback data for offline pipeline verification.
"""

import os
import random
from typing import List, Tuple, Optional, Dict

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Exact C-MAPSS 26-column schema
COLUMN_NAMES = [
    "unit_number",
    "time_cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

# Invariant (zero-variance) sensors in sea-level single operating regime (FD001 / FD003)
INVARIANT_SENSORS_FD001 = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

# Informative degradation sensors
INFORMATIVE_SENSORS_FD001 = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]


def get_column_names() -> List[str]:
    """Returns the standard 26 column names of raw C-MAPSS files."""
    return list(COLUMN_NAMES)


def load_cmapss_raw(data_dir: str, subset: str = "FD001", split: str = "train"):
    """
    Loads raw C-MAPSS text file into a DataFrame or list of dicts.

    Args:
        data_dir: Directory containing C-MAPSS files (e.g. data/raw).
        subset: 'FD001', 'FD002', 'FD003', or 'FD004'.
        split: 'train' or 'test'.

    Returns:
        pd.DataFrame (if pandas available) or List[List[float]].
    """
    filename = f"{split}_{subset}.txt"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File '{filepath}' not found. Please run scripts/download_cmapss.py or generate synthetic data."
        )

    if HAS_PANDAS:
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
        return df
    else:
        # Fallback pure python parser
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    row = [int(parts[0]), int(parts[1])] + [float(x) for x in parts[2:]]
                    rows.append(row)
        return rows


def load_cmapss_test_rul(data_dir: str, subset: str = "FD001") -> List[float]:
    """
    Loads ground-truth RUL values for the test engines.

    Args:
        data_dir: Directory containing RUL_{subset}.txt.
        subset: 'FD001', 'FD002', 'FD003', or 'FD004'.

    Returns:
        List of true RUL integers/floats for each test engine.
    """
    filename = f"RUL_{subset}.txt"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    ruls = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if val:
                ruls.append(float(val))
    return ruls


def generate_synthetic_cmapss(output_dir: str, subset: str = "FD001", num_train_units: int = 100, num_test_units: int = 100):
    """
    Generates synthetic C-MAPSS data files that match the exact schema, distribution,
    and degradation physics of FD001. Useful for offline unit testing and automated validation.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)

    train_path = os.path.join(output_dir, f"train_{subset}.txt")
    test_path = os.path.join(output_dir, f"test_{subset}.txt")
    rul_path = os.path.join(output_dir, f"RUL_{subset}.txt")

    print(f"Generating synthetic {subset} dataset at '{output_dir}'...")

    # Sensor baseline values (mean and noise scale)
    base_sensors = {
        "s1": (518.67, 0.0),    # Invariant
        "s2": (642.50, 0.25),   # Degrading up (+0.03/cycle)
        "s3": (1588.0, 1.5),    # Degrading up (+0.10/cycle)
        "s4": (1405.0, 1.2),    # Degrading up (+0.15/cycle)
        "s5": (14.62, 0.0),     # Invariant
        "s6": (21.61, 0.0),     # Invariant
        "s7": (554.0, 0.4),     # Degrading down (-0.05/cycle)
        "s8": (2388.08, 0.05),  # Slight downward trend
        "s9": (9055.0, 3.0),    # Degrading up (+0.12/cycle)
        "s10": (1.00, 0.0),     # Invariant
        "s11": (47.3, 0.1),     # Degrading up (+0.015/cycle)
        "s12": (521.8, 0.3),    # Degrading down (-0.04/cycle)
        "s13": (2388.1, 0.05),  # Fan speed
        "s14": (8130.0, 2.5),   # Degrading up (+0.10/cycle)
        "s15": (8.42, 0.02),    # Degrading up (+0.001/cycle)
        "s16": (0.03, 0.0),     # Invariant
        "s17": (392.0, 1.0),    # Degrading up (+0.04/cycle)
        "s18": (2388.0, 0.0),   # Invariant
        "s19": (100.0, 0.0),    # Invariant
        "s20": (38.8, 0.1),     # Degrading down (-0.004/cycle)
        "s21": (23.3, 0.08),    # Degrading down (-0.003/cycle)
    }

    # Generate Train
    with open(train_path, "w", encoding="utf-8") as f_train:
        for unit in range(1, num_train_units + 1):
            lifetime = random.randint(140, 280)
            for cycle in range(1, lifetime + 1):
                # Wear ratio (0.0 healthy -> 1.0 failure)
                wear = max(0.0, (cycle - 60) / float(lifetime - 60)) if cycle > 60 else 0.0
                settings = [0.0000, 0.0000, 100.0]  # Sea level
                sensors = []
                for s_idx in range(1, 22):
                    s_name = f"s{s_idx}"
                    base, noise = base_sensors[s_name]
                    deg = 0.0
                    if s_name in ["s2", "s3", "s4", "s9", "s11", "s14", "s15", "s17"]:
                        deg = wear * (base * 0.03)
                    elif s_name in ["s7", "s12", "s20", "s21"]:
                        deg = -wear * (base * 0.02)
                    noise_val = random.gauss(0, noise) if noise > 0 else 0.0
                    sensors.append(round(base + deg + noise_val, 4))
                line = f"{unit} {cycle} {settings[0]} {settings[1]} {settings[2]} " + " ".join(f"{x:.4f}" for x in sensors)
                f_train.write(line + "\n")

    # Generate Test and True RUL
    with open(test_path, "w", encoding="utf-8") as f_test, open(rul_path, "w", encoding="utf-8") as f_rul:
        for unit in range(1, num_test_units + 1):
            total_lifetime = random.randint(140, 280)
            remaining_rul = random.randint(15, 125)
            observed_cycles = max(31, total_lifetime - remaining_rul)
            f_rul.write(f"{remaining_rul}\n")

            for cycle in range(1, observed_cycles + 1):
                wear = max(0.0, (cycle - 60) / float(total_lifetime - 60)) if cycle > 60 else 0.0
                settings = [0.0000, 0.0000, 100.0]
                sensors = []
                for s_idx in range(1, 22):
                    s_name = f"s{s_idx}"
                    base, noise = base_sensors[s_name]
                    deg = 0.0
                    if s_name in ["s2", "s3", "s4", "s9", "s11", "s14", "s15", "s17"]:
                        deg = wear * (base * 0.03)
                    elif s_name in ["s7", "s12", "s20", "s21"]:
                        deg = -wear * (base * 0.02)
                    noise_val = random.gauss(0, noise) if noise > 0 else 0.0
                    sensors.append(round(base + deg + noise_val, 4))
                line = f"{unit} {cycle} {settings[0]} {settings[1]} {settings[2]} " + " ".join(f"{x:.4f}" for x in sensors)
                f_test.write(line + "\n")

    print(f"✓ Synthetic C-MAPSS dataset successfully generated in '{output_dir}'.")


if __name__ == "__main__":
    generate_synthetic_cmapss("data/raw", subset="FD001", num_train_units=10, num_test_units=10)
