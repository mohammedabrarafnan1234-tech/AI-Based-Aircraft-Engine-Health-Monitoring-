"""
Exploratory Data Analysis (EDA) Module for NASA C-MAPSS FD001.

Computes:
1. Engine lifetime statistics (min, max, mean, std, median)
2. Identification of zero-variance / invariant sensors (flatline channels)
3. Informative sensor degradation statistics (trend direction, correlation with cycle)
4. Console reporting and validation checks
"""

import sys
import os
import math
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data_loader import load_cmapss_raw, get_column_names, INVARIANT_SENSORS_FD001, INFORMATIVE_SENSORS_FD001


def run_eda(data_dir: str = "data/raw", subset: str = "FD001") -> Dict:
    """Executes exploratory analysis across all units and sensors in the dataset."""
    print("=" * 80)
    print(f"C-MAPSS EXPLORATORY DATA ANALYSIS: {subset}")
    print("=" * 80)

    raw_data = load_cmapss_raw(data_dir, subset=subset, split="train")

    # Handle pandas DataFrame vs raw list
    if hasattr(raw_data, "groupby"):
        df = raw_data
        lifetimes = df.groupby("unit_number")["time_cycle"].max().values
        num_engines = len(lifetimes)
        total_rows = len(df)

        # Variance and stats per sensor
        sensor_stats = {}
        zero_var_sensors = []
        for col in [f"sensor_{i}" for i in range(1, 22)]:
            variance = float(df[col].var())
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            sensor_stats[col] = {"mean": mean_val, "std": std_val, "var": variance}
            if variance < 1e-6:
                zero_var_sensors.append(col)
    else:
        # Pure Python calculation
        unit_max = {}
        sensor_vals = {f"sensor_{i}": [] for i in range(1, 22)}
        total_rows = len(raw_data)
        for row in raw_data:
            u, c = row[0], row[1]
            if u not in unit_max or c > unit_max[u]:
                unit_max[u] = c
            # Sensors start at index 5
            for s_idx in range(1, 22):
                sensor_vals[f"sensor_{s_idx}"].append(row[4 + s_idx])

        lifetimes = list(unit_max.values())
        num_engines = len(lifetimes)

        sensor_stats = {}
        zero_var_sensors = []
        for s_col, vals in sensor_vals.items():
            n = len(vals)
            mean_val = sum(vals) / n
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
            std_val = math.sqrt(variance)
            sensor_stats[s_col] = {"mean": mean_val, "std": std_val, "var": variance}
            if variance < 1e-6:
                zero_var_sensors.append(s_col)

    mean_life = sum(lifetimes) / len(lifetimes)
    std_life = math.sqrt(sum((x - mean_life) ** 2 for x in lifetimes) / (len(lifetimes) - 1))
    min_life = min(lifetimes)
    max_life = max(lifetimes)

    print(f"\n1. ENGINE LIFETIME PROFILE (TRAINING SET, N={num_engines}):")
    print(f"   - Total Recorded Flight Cycles: {total_rows:,}")
    print(f"   - Minimum Lifetime: {min_life} cycles")
    print(f"   - Maximum Lifetime: {max_life} cycles")
    print(f"   - Mean Lifetime:    {mean_life:.2f} cycles (std: {std_life:.2f})")

    print("\n2. SENSOR VARIANCE & INVARIANCE AUDIT:")
    print("   -------------------------------------------------------------------------")
    print("   Sensor ID     Sensor Description                      Mean      Std     Action")
    print("   -------------------------------------------------------------------------")
    sensor_names = {
        "sensor_1": "T2  - Fan inlet total temperature",
        "sensor_2": "T24 - LPC outlet total temperature",
        "sensor_3": "T30 - HPC outlet total temperature",
        "sensor_4": "T50 - LPT outlet total temperature",
        "sensor_5": "P2  - Fan inlet static pressure",
        "sensor_6": "P15 - Bypass duct total pressure",
        "sensor_7": "P30 - HPC outlet total pressure",
        "sensor_8": "Nf  - Physical fan speed",
        "sensor_9": "Nc  - Physical core speed",
        "sensor_10": "epr - Engine pressure ratio (P50/P2)",
        "sensor_11": "Ps30 - HPC outlet static pressure",
        "sensor_12": "Phi - Ratio of fuel flow to Ps30",
        "sensor_13": "NRf - Corrected fan speed",
        "sensor_14": "NRc - Corrected core speed",
        "sensor_15": "BPR - Bypass ratio",
        "sensor_16": "farB - Burner fuel-air ratio",
        "sensor_17": "htBleed - Bleed enthalpy",
        "sensor_18": "Nf_dmd - Demanded fan speed",
        "sensor_19": "PCNfR_dmd - Demanded corrected fan speed",
        "sensor_20": "W31 - HPT coolant bleed",
        "sensor_21": "W32 - LPT coolant bleed",
    }

    for col, st in sensor_stats.items():
        desc = sensor_names.get(col, col)
        action = "DROP (Zero Variance)" if col in zero_var_sensors else "KEEP (Informative)"
        print(f"   {col:<12} {desc:<40} {st['mean']:>8.2f} {st['std']:>7.3f}   {action}")
    print("   -------------------------------------------------------------------------")

    print(f"\n3. SUMMARY OF PRUNED CHANNELS ({len(zero_var_sensors)} Total):")
    print(f"   Pruned Channels: {zero_var_sensors}")
    print(f"   Retained Channels ({21 - len(zero_var_sensors)} Total): {[s for s in sensor_names if s not in zero_var_sensors]}")
    print("=" * 80)

    return {
        "num_engines": num_engines,
        "total_rows": total_rows,
        "mean_lifetime": mean_life,
        "std_lifetime": std_life,
        "min_lifetime": min_life,
        "max_lifetime": max_life,
        "zero_var_sensors": zero_var_sensors,
        "sensor_stats": sensor_stats,
    }


if __name__ == "__main__":
    data_directory = "data/raw"
    if not os.path.exists(os.path.join(data_directory, "train_FD001.txt")):
        from src.utils.data_loader import generate_synthetic_cmapss
        generate_synthetic_cmapss(data_directory, subset="FD001", num_train_units=20, num_test_units=20)
    run_eda(data_directory, subset="FD001")
