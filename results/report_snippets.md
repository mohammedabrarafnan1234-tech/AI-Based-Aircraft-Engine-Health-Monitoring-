# Capstone Experiment Outputs & Report Snippets
**Execution Timestamp:** 2026-11-21 14:15:35
**Target Dataset:** C-MAPSS FD001 (Full 20,631 Train Rows, 100 Test Engines)

---

## 1. Terminal Output for Report 1 (Sprint 1 Preprocessing & EDA)
```
$ python3 src/eda.py --dataset FD001
1. ENGINE LIFETIME PROFILE (TRAINING SET, N=100):
   Total recorded flight cycles: 20,631
   Lifetime min / max / mean: 128 / 362 / 206.31 cycles (std 46.34)
3. SUMMARY OF PRUNED CHANNELS (7 total):
   sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
   Retained channels (14 total): sensor_2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21

$ python3 src/preprocessing/preprocess.py --dataset FD001
OK Train 3D sequences shape: 17731 samples, window: 30x14
OK Test 3D sequences shape:  100 samples, window: 30x14
OK Tabular baseline feature shape: 17731x84
OK Preprocessing pipeline passed validation.
```

---

## 2. Terminal Output for Report 2 (Sprint 2 Benchmark Results)
```
$ python3 capstone_run.py --data-dir data/raw --out results
Preprocessed training samples (3D): 17731 windows of shape (30, 14)
Preprocessed test engines (3D):    100 test engines

Model Architecture        RMSE (Cycles)   NASA Score      Late Pred %
Random Forest             16.45           630.6           57.0        %
XGBoost                   16.53           541.6           56.0        %
PyTorch Stacked LSTM      13.73           402.7           47.0        %
PyTorch 1D-CNN            16.33           684.9           57.0        %
```

---

## 3. Quantitative Markdown Summary Table
| Model Architecture | Input Features | Test RMSE (Cycles) | NASA Score ($S$) | Late Error % | MAE (Cycles) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | 84 Window Statistics | **16.45** | **630.6** | 57.0% | 12.2 |
| **XGBoost** | 84 Window Statistics | **16.53** | **541.6** | 56.0% | 12.2 |
| **PyTorch Stacked LSTM** | 3D Sequence $(30, 14)$ | **13.73** | **402.7** | 47.0% | 10.21 |
| **PyTorch 1D-CNN** | 3D Sequence $(14, 30)$ | **16.33** | **684.9** | 57.0% | 12.06 |
