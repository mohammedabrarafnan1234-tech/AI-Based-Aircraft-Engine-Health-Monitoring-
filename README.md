# AI-Based Aircraft Engine Health Monitoring and Cost-Aware Maintenance Decision Support Using Remaining Useful Life Estimation

**University of Europe for Applied Sciences — M.Sc. Data Science Capstone Project (2026/2027)**  
**Student:** Mohammed Abrar (Matriculation No. 97193111)  
**Supervisor:** Dr. Humera Noor  

---

## 📌 Project Overview
This repository contains the end-to-end research, engineering, and predictive maintenance platform for estimating the **Remaining Useful Life (RUL)** of turbofan aircraft engines from multivariate sensor time-series data (NASA C-MAPSS benchmark). The system pairs high-accuracy deep temporal prognostic models (1D-CNN and LSTM) with a cost-aware decision layer designed to balance the severe operational cost of unplanned in-flight failure ($C_{fail}$) against the cost of premature component replacement ($C_{pm}$).

---

## 🚀 Repository Structure

```
├── Sprint_1_Deliverables.md        # Master literature review, EDA, preprocessing, and Report 1
├── Sprint_2_Deliverables.md        # Master metrics, baselines, deep models, benchmarks, and Report 2
├── reports/                        # Submission-ready milestone reports
│   ├── Report_1_Data_Understanding_Preprocessing.md
│   └── Report_2_Baselines_and_Deep_Models.md
├── scripts/
│   ├── download_cmapss.py          # Automated dataset downloader & synthetic generator
│   └── run_pipeline.py             # CLI runner for end-to-end preprocessing & benchmarking
├── src/
│   ├── eda.py                      # Sensor variance audit & degradation diagnostics
│   ├── utils/
│   │   ├── data_loader.py          # C-MAPSS dataset parser & schema validator
│   │   └── metrics.py              # RMSE and NASA PHM 2008 Asymmetric Score
│   ├── preprocessing/
│   │   └── preprocess.py           # Invariant sensor pruning, normalisation, sliding windows
│   └── models/
│       ├── baselines.py            # Random Forest & XGBoost regressors
│       └── deep_models.py          # PyTorch LSTM & 1D-CNN architectures
├── requirements.txt                # Pinned production dependencies
└── .gitignore                      # Standard clean gitignore
```

---

## 🛠 Quickstart Guide

### 1. Environment Setup
Create and activate an isolated Python 3.10 or 3.11 virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download and Verify Data (Sprint 1)
Download the C-MAPSS dataset (or generate synthetic validation data if offline):
```bash
python3 scripts/download_cmapss.py --dataset FD001
```

### 3. Run Exploratory Data Analysis (Sprint 1)
Analyze engine lifetimes and detect constant sensors:
```bash
python3 src/eda.py --dataset FD001
```

### 4. Execute Preprocessing Pipeline (Sprint 1)
Prune flatline sensors, apply MinMax normalisation, and generate 30-cycle sliding windows:
```bash
python3 src/preprocessing/preprocess.py --dataset FD001 --window-size 30 --rul-max 125
```

### 5. Run Baselines and Deep Learning Benchmarks (Sprint 2)
Train and evaluate Random Forest, XGBoost, LSTM, and 1D-CNN on FD001:
```bash
python3 src/models/baselines.py --dataset FD001
python3 src/models/deep_models.py --model cnn1d --dataset FD001
python3 src/models/deep_models.py --model lstm --dataset FD001
```

---

## 📊 Summary of Benchmark Results (FD001 Test Set)

| Model Architecture | Model Class | Feature Space | Test RMSE (Cycles) | NASA Score ($S$) | Inference Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | Tabular Bagging | Window Statistics | 17.84 | 542.1 | 18 ms |
| **XGBoost** | Tabular Boosting | Window Statistics | 16.42 | 438.7 | 12 ms |
| **PyTorch LSTM** | Recurrent Deep Net | 3D Sequence $(30, 14)$ | 14.86 | 328.4 | 24 ms |
| **PyTorch 1D-CNN** | Temporal ConvNet | 3D Sequence $(14, 30)$ | **13.98** | **289.2** | **9 ms** |

---

## 📖 Milestone Deliverables
- **Report 1 (9 Nov 2026):** [Report 1: Data Understanding & Preprocessing Pipeline](reports/Report_1_Data_Understanding_Preprocessing.md)
- **Report 2 (23 Nov 2026):** [Report 2: Baselines & First Deep Models](reports/Report_2_Baselines_and_Deep_Models.md)
- **Master Sprint 1 Document:** [Sprint 1 Deliverables](Sprint_1_Deliverables.md)
- **Master Sprint 2 Document:** [Sprint 2 Deliverables](Sprint_2_Deliverables.md)
