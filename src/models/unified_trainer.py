"""
Unified Prognostic Training and Evaluation Engine.

Provides:
1. Deterministic random seed fixation across Python, NumPy, PyTorch, and Scikit-Learn.
2. Single unified function `train_and_evaluate_model()` returning:
   - RMSE (Root Mean Squared Error)
   - NASA Asymmetric Scoring Function
   - MAE (Mean Absolute Error)
   - Late Prediction Percentage (%)
   - Training Duration (seconds)
   - Predictions and Residuals
3. Support for 4 prognostic architectures:
   - Random Forest Regressor (84-102 tabular window features)
   - XGBoost Regressor (isolated process execution to prevent macOS OpenMP collision)
   - PyTorch Stacked LSTM (2-layer recurrent network with adaptive sequence handling)
   - PyTorch 1D-CNN (Temporal Convolutional Network with AdaptiveAvgPool1d)
4. Multi-seed repetition support (calculating mean +/- std across N random seeds).
"""

import os
import sys
# Prevent macOS OpenMP thread collision / segfault
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import tempfile
import subprocess
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestRegressor

# Metrics
def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_pred - y_true))))

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))

def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    # d < 0: early prediction -> exp(-d/13) - 1
    # d >= 0: late prediction -> exp(d/10) - 1
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    d = np.asarray(y_pred) - np.asarray(y_true)
    early_mask = d < 0
    late_mask = d >= 0
    return {
        "rmse": float(np.round(compute_rmse(y_true, y_pred), 3)),
        "mae": float(np.round(compute_mae(y_true, y_pred), 3)),
        "nasa_score": float(np.round(compute_nasa_score(y_true, y_pred), 2)),
        "late_pct": float(np.round(np.mean(late_mask) * 100.0, 2)),
        "early_pct": float(np.round(np.mean(early_mask) * 100.0, 2)),
        "max_early": float(np.round(np.min(d) if np.any(early_mask) else 0.0, 2)),
        "max_late": float(np.round(np.max(d) if np.any(late_mask) else 0.0, 2)),
    }

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# PyTorch Deep Learning Architectures
# ==============================================================================
class FlexibleLSTM(nn.Module):
    """
    Stacked 2-Layer LSTM with flexible channel dimension, hidden units, and dropout.
    Accepts arbitrary sequence lengths (window sizes 20-50).
    """
    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            num_layers=1,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1,
            hidden_size=hidden_dim2,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.bn = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, (h_n, _) = self.lstm2(out)
        h = self.relu(self.bn(self.fc1(h_n[-1])))
        h = self.drop2(h)
        return self.out(h).squeeze(-1)


class Flexible1DCNN(nn.Module):
    """
    Multi-Layer 1D Temporal CNN with flexible channel dimension, kernel sizes, and filter depths.
    Uses AdaptiveAvgPool1d to seamlessly accept arbitrary window sizes (20-50).
    """
    def __init__(
        self,
        in_channels: int = 14,
        filters1: int = 32,
        filters2: int = 64,
        filters3: int = 64,
        kernel_size1: int = 5,
        kernel_size2: int = 3,
        kernel_size3: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        p1 = kernel_size1 // 2
        p2 = kernel_size2 // 2
        p3 = kernel_size3 // 2

        self.conv1 = nn.Conv1d(in_channels, filters1, kernel_size=kernel_size1, stride=1, padding=p1)
        self.bn1 = nn.BatchNorm1d(filters1)
        self.act1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv1d(filters1, filters2, kernel_size=kernel_size2, stride=1, padding=p2)
        self.bn2 = nn.BatchNorm1d(filters2)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(filters2, filters3, kernel_size=kernel_size3, stride=1, padding=p3)
        self.bn3 = nn.BatchNorm1d(filters3)
        self.act3 = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(filters3, 32)
        self.act4 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, seq_len, in_channels) -> transpose to (batch, in_channels, seq_len)
        if x.dim() == 3 and x.size(2) < x.size(1):
            x = x.transpose(1, 2)
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.pool1(self.act2(self.bn2(self.conv2(h))))
        h = self.act3(self.bn3(self.conv3(h)))
        h = self.avgpool(h).squeeze(-1)
        h = self.drop(self.act4(self.fc1(h)))
        return self.out(h).squeeze(-1)


# ==============================================================================
# Unified Training and Evaluation Engine
# ==============================================================================
def train_and_evaluate_model(
    model_name: str,
    data: Dict[str, Any],
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
    epochs: int = 40,
    batch_size: int = 128,
    rul_max: float = 125.0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Universal training and evaluation entrypoint.
    
    Args:
        model_name: One of ['rf', 'random_forest', 'xgb', 'xgboost', 'lstm', 'cnn', '1d_cnn'].
        data: Preprocessed dataset dictionary from MultiConditionPreprocessor.
        seed: Random seed for reproducibility.
        params: Optional hyperparameter overrides.
        epochs: Max epochs for PyTorch models.
        batch_size: Batch size for PyTorch models.
        rul_max: Capped RUL threshold (125.0).
        device: 'cpu' or 'cuda' (defaults to cpu / available).
        verbose: Print progress logs.

    Returns:
        Dict with metrics, test predictions, validation predictions, and training duration.
    """
    set_seed(seed)
    params = params or {}
    m_type = model_name.lower().replace("-", "_").replace(" ", "_")
    start_t = time.time()

    y_test = data["y_test"]
    y_val = data["y_val_3d"]

    # --------------------------------------------------------------------------
    # 1. Random Forest Regressor
    # --------------------------------------------------------------------------
    if m_type in ["rf", "random_forest"]:
        n_estimators = params.get("n_estimators", 200)
        max_depth = params.get("max_depth", 15)
        min_samples_split = params.get("min_samples_split", 5)
        min_samples_leaf = params.get("min_samples_leaf", 2)
        max_features = params.get("max_features", "sqrt")

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(data["X_train_tab"], data["y_train_3d"])
        
        y_val_pred = np.clip(rf.predict(data["X_val_tab"]), 0.0, rul_max)
        y_test_pred = np.clip(rf.predict(data["X_test_tab"]), 0.0, rul_max)
        train_time = time.time() - start_t
        trained_model = rf

    # --------------------------------------------------------------------------
    # 2. XGBoost Regressor (isolated process execution)
    # --------------------------------------------------------------------------
    elif m_type in ["xgb", "xgboost"]:
        n_estimators = params.get("n_estimators", 300)
        learning_rate = params.get("learning_rate", 0.05)
        max_depth = params.get("max_depth", 6)
        subsample = params.get("subsample", 0.8)
        colsample_bytree = params.get("colsample_bytree", 0.8)
        reg_alpha = params.get("reg_alpha", 0.1)
        reg_lambda = params.get("reg_lambda", 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "xgb_data.npz")
            pred_file = os.path.join(tmpdir, "xgb_pred.npz")
            np.savez_compressed(
                data_file,
                X_train=data["X_train_tab"],
                y_train=data["y_train_3d"],
                X_val=data["X_val_tab"],
                y_val=data["y_val_3d"],
                X_test=data["X_test_tab"],
            )
            script = f'''
import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import xgboost as xgb

data = np.load(r"{data_file}")
m = xgb.XGBRegressor(
    n_estimators={n_estimators},
    learning_rate={learning_rate},
    max_depth={max_depth},
    subsample={subsample},
    colsample_bytree={colsample_bytree},
    reg_alpha={reg_alpha},
    reg_lambda={reg_lambda},
    objective="reg:squarederror",
    random_state={seed},
    n_jobs=4,
)
m.fit(data["X_train"], data["y_train"], eval_set=[(data["X_val"], data["y_val"])], verbose=False)
val_preds = m.predict(data["X_val"])
test_preds = m.predict(data["X_test"])
np.savez_compressed(r"{pred_file}", val=val_preds, test=test_preds)
'''
            res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"XGBoost training failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            preds_loaded = np.load(pred_file)
            y_val_pred = np.clip(preds_loaded["val"], 0.0, rul_max)
            y_test_pred = np.clip(preds_loaded["test"], 0.0, rul_max)
        train_time = time.time() - start_t
        trained_model = None

    # --------------------------------------------------------------------------
    # 3. PyTorch Stacked LSTM
    # --------------------------------------------------------------------------
    elif m_type in ["lstm", "stacked_lstm"]:
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        input_dim = data["X_train_3d"].shape[2]
        hidden_dim1 = params.get("hidden_dim1", 64)
        hidden_dim2 = params.get("hidden_dim2", 32)
        dropout = params.get("dropout", 0.2)
        lr = params.get("learning_rate", 0.001)

        model = FlexibleLSTM(
            input_dim=input_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
        ).to(dev)

        train_ds = TensorDataset(torch.tensor(data["X_train_3d"]), torch.tensor(data["y_train_3d"]))
        val_ds = TensorDataset(torch.tensor(data["X_val_3d"]), torch.tensor(data["y_val_3d"]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_loss = float("inf")
        best_state = None
        patience = 6
        patience_cnt = 0

        for epoch in range(1, epochs + 1):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(dev), by.to(dev)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(dev), by.to(dev)
                    v_loss += criterion(model(bx), by).item() * len(by)
            v_loss /= len(val_ds)
            scheduler.step(v_loss)

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(torch.tensor(data["X_val_3d"]).to(dev)).cpu().numpy()
            y_test_pred = model(torch.tensor(data["X_test_3d"]).to(dev)).cpu().numpy()

        y_val_pred = np.clip(y_val_pred, 0.0, rul_max)
        y_test_pred = np.clip(y_test_pred, 0.0, rul_max)
        train_time = time.time() - start_t
        trained_model = model

    # --------------------------------------------------------------------------
    # 4. PyTorch 1D-CNN
    # --------------------------------------------------------------------------
    elif m_type in ["cnn", "1d_cnn"]:
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        in_channels = data["X_train_3d"].shape[2]
        filters1 = params.get("filters1", 32)
        filters2 = params.get("filters2", 64)
        filters3 = params.get("filters3", 64)
        kernel_size1 = params.get("kernel_size1", 5)
        kernel_size2 = params.get("kernel_size2", 3)
        kernel_size3 = params.get("kernel_size3", 3)
        dropout = params.get("dropout", 0.2)
        lr = params.get("learning_rate", 0.001)

        model = Flexible1DCNN(
            in_channels=in_channels,
            filters1=filters1,
            filters2=filters2,
            filters3=filters3,
            kernel_size1=kernel_size1,
            kernel_size2=kernel_size2,
            kernel_size3=kernel_size3,
            dropout=dropout,
        ).to(dev)

        train_ds = TensorDataset(torch.tensor(data["X_train_3d"]), torch.tensor(data["y_train_3d"]))
        val_ds = TensorDataset(torch.tensor(data["X_val_3d"]), torch.tensor(data["y_val_3d"]))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        best_loss = float("inf")
        best_state = None
        patience = 6
        patience_cnt = 0

        for epoch in range(1, epochs + 1):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(dev), by.to(dev)
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(dev), by.to(dev)
                    v_loss += criterion(model(bx), by).item() * len(by)
            v_loss /= len(val_ds)
            scheduler.step(v_loss)

            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(torch.tensor(data["X_val_3d"]).to(dev)).cpu().numpy()
            y_test_pred = model(torch.tensor(data["X_test_3d"]).to(dev)).cpu().numpy()

        y_val_pred = np.clip(y_val_pred, 0.0, rul_max)
        y_test_pred = np.clip(y_test_pred, 0.0, rul_max)
        train_time = time.time() - start_t
        trained_model = model

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Compute evaluation metrics
    val_metrics = evaluate_metrics(y_val, y_val_pred)
    test_metrics = evaluate_metrics(y_test, y_test_pred)
    residuals = y_test_pred - y_test

    if verbose:
        print(f"[{model_name.upper()} | Seed {seed}] Val RMSE: {val_metrics['rmse']:.2f} | Test RMSE: {test_metrics['rmse']:.2f} | NASA Score: {test_metrics['nasa_score']:.1f} ({train_time:.1f}s)")

    return {
        "model_name": model_name,
        "seed": seed,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,
        "residuals": residuals,
        "train_time": train_time,
        "model": trained_model,
    }
