"""
C-MAPSS Dataset Downloader and Schema Ingestion Script.

Attempts to retrieve original NASA C-MAPSS benchmark archives. If network is unavailable
(e.g. offline sandbox or firewall), automatically generates verified schema-identical
data files so the entire engineering and modeling pipeline can be executed immediately.
"""

import os
import sys
import argparse
import urllib.request
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.data_loader import generate_synthetic_cmapss

# NASA Prognostics Data Repository Mirror
DATASET_URL = "https://raw.githubusercontent.com/Azure/MachineLearningNotebooks/master/how-to-use-azureml/automated-machine-learning/continuous-retraining/CMAPSSData.tar.gz"


def download_or_generate(data_dir: str = "data/raw", subset: str = "FD001", force_synthetic: bool = False):
    """Downloads dataset from web or falls back to synthetic data generation."""
    os.makedirs(data_dir, exist_ok=True)
    train_file = os.path.join(data_dir, f"train_{subset}.txt")
    test_file = os.path.join(data_dir, f"test_{subset}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{subset}.txt")

    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(rul_file):
        print(f"✓ Dataset {subset} already exists in '{data_dir}'. Ready to proceed.")
        return

    if force_synthetic:
        print(f"Generating synthetic {subset} data as requested...")
        generate_synthetic_cmapss(data_dir, subset=subset, num_train_units=100, num_test_units=100)
        return

    print(f"Attempting download of C-MAPSS dataset...")
    try:
        # Try download
        tar_path = os.path.join(data_dir, "cmapss.tar.gz")
        urllib.request.urlretrieve(DATASET_URL, tar_path)
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print("✓ Successfully downloaded and unpacked C-MAPSS data.")
    except Exception as e:
        print(f"[Notice] Network download unavailable ({e}).")
        print("-> Activating built-in synthetic generator to produce exact-schema FD001 data...")
        generate_synthetic_cmapss(data_dir, subset=subset, num_train_units=100, num_test_units=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate NASA C-MAPSS benchmark data.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Target raw data directory")
    parser.add_argument("--dataset", type=str, default="FD001", choices=["FD001", "FD002", "FD003", "FD004"], help="Dataset subset")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic generation")
    args = parser.parse_args()

    download_or_generate(data_dir=args.data_dir, subset=args.dataset, force_synthetic=args.synthetic)
