from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_FIGURE_DIR = CONTROLLER_ROOT / "artifacts" / "report_figures" / "energy_prediction_multidevice"
DEFAULT_MODEL_OUTPUT_DIR = CONTROLLER_ROOT / "artifacts"

FAMILY_PATTERNS = [
    "convnextv2",
    "efficientformer",
    "efficientvit",
    "mobilenet",
    "efficientnet",
    "convnext",
    "coatnet",
    "shufflenet",
    "squeezenet",
    "ghostnet",
    "mnasnet",
    "densenet",
    "regnet",
    "resnet",
    "seresnet",
    "deit",
    "beit",
    "crossvit",
    "convit",
    "maxvit",
    "fastvit",
    "mobilevit",
    "edgenext",
    "coat",
    "cait",
    "xcit",
    "swin",
    "vit",
    "vgg",
    "inception",
    "rexnet",
    "fbnetv",
    "lcnet",
    "tinynet",
    "hrnet",
    "darknet",
    "dla",
    "dpn",
    "movenet",
    "yolo",
    "cnn",
]

TRANSFORMER_TOKENS = {
    "vit",
    "deit",
    "beit",
    "swin",
    "xcit",
    "crossvit",
    "convit",
    "cait",
    "maxvit",
    "coatnet",
    "coat",
}
MOBILE_EDGE_TOKENS = {
    "mobilenet",
    "mobilevit",
    "ghostnet",
    "mnasnet",
    "shufflenet",
    "fbnetv",
    "rexnet",
    "lcnet",
    "movenet",
}
EFFICIENT_EDGE_TOKENS = {
    "efficientnet",
    "efficientformer",
    "efficientvit",
    "tinynet",
    "edgenext",
    "fastvit",
}
CNN_BACKBONE_TOKENS = {
    "resnet",
    "seresnet",
    "regnet",
    "densenet",
    "vgg",
    "inception",
    "convnext",
    "convnextv2",
    "hrnet",
    "darknet",
    "dla",
    "dpn",
    "cnn",
}

EXCLUDED_COLUMNS = {
    "model",
    "energy_avg_mwh",
    "energy_std_mwh",
    "energy_run1_mwh",
    "energy_run2_mwh",
    "energy_run3_mwh",
}


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_model_family(model_name: Any) -> str:
    name = str(model_name or "").strip().lower()
    if not name:
        return "unknown"
    for pattern in FAMILY_PATTERNS:
        if pattern in name:
            return pattern
    prefix = "".join(ch for ch in name.split("_")[0] if ch.isalpha())
    return prefix or "other"


def infer_function_group(model_name: Any) -> str:
    name = str(model_name or "").strip().lower()
    if any(token in name for token in TRANSFORMER_TOKENS):
        return "vision_transformer"
    if any(token in name for token in MOBILE_EDGE_TOKENS):
        return "mobile_edge"
    if any(token in name for token in EFFICIENT_EDGE_TOKENS):
        return "efficient_edge"
    if "yolo" in name:
        return "detection_backbone"
    if any(token in name for token in CNN_BACKBONE_TOKENS):
        return "cnn_backbone"
    return "general_vision"


def parse_resolution(value: Any) -> Tuple[float, float]:
    text = str(value or "").strip().lower()
    if not text:
        return (np.nan, np.nan)
    text = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    text = text.replace(",", "x").replace(" ", "")
    parts = [part for part in text.split("x") if part]
    if len(parts) < 2:
        return (np.nan, np.nan)
    try:
        return (float(parts[-2]), float(parts[-1]))
    except ValueError:
        return (np.nan, np.nan)


def _safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series:
    denominator_arr = np.asarray(denominator, dtype=float)
    numerator_arr = np.asarray(numerator, dtype=float)
    values = numerator_arr / np.where(np.abs(denominator_arr) < 1e-9, np.nan, denominator_arr)
    return pd.Series(values)


def standardize_benchmark_dataset(path: Path | str, device_name: str, device_type: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["device_name"] = device_name
    df["device_type"] = device_type
    df["dataset_source"] = Path(path).name
    df["model_family"] = df["model"].map(infer_model_family)
    df["function_group"] = df["model"].map(infer_function_group)

    widths, heights = zip(*df["input_resolution_actual"].map(parse_resolution))
    df["input_width"] = widths
    df["input_height"] = heights
    df["input_pixels_mp"] = (df["input_width"] * df["input_height"]) / 1_000_000.0
    df["aspect_ratio"] = _safe_divide(df["input_width"], df["input_height"])

    for column in ("duration_run1_s", "duration_run2_s", "duration_run3_s"):
        if column not in df.columns:
            df[column] = np.nan

    duration_columns = [col for col in df.columns if col.startswith("duration_run")]
    df["duration_mean_s"] = df[duration_columns].mean(axis=1)
    df["duration_std_s"] = df[duration_columns].std(axis=1, ddof=0)

    df["params_per_mb"] = _safe_divide(df["params_m"], df["size_mb"] + 1e-9)
    df["gflops_per_param"] = _safe_divide(df["gflops"], df["params_m"] + 1e-9)
    df["gmacs_per_param"] = _safe_divide(df["gmacs"], df["params_m"] + 1e-9)
    df["throughput_per_param"] = _safe_divide(df["throughput_iter_per_s"], df["params_m"] + 1e-9)
    df["latency_per_param"] = _safe_divide(df["latency_avg_s"], df["params_m"] + 1e-9)
    df["latency_per_gflop"] = _safe_divide(df["latency_avg_s"], df["gflops"] + 1e-9)
    df["compute_to_memory_ratio"] = _safe_divide(df["gflops"], df["size_mb"] + 1e-9)
    df["throughput_x_pixels"] = df["throughput_iter_per_s"] * df["input_pixels_mp"].fillna(0)
    df["latency_x_params"] = df["latency_avg_s"] * df["params_m"]
    df["latency_x_gflops"] = df["latency_avg_s"] * df["gflops"]
    df["complexity_index"] = (
        np.log1p(df["params_m"].clip(lower=0))
        * np.log1p(df["gflops"].clip(lower=0))
        * np.log1p(df["size_mb"].clip(lower=0))
    )
    df["batch_run_time_s"] = df["latency_avg_s"] * df["runs"]
    df["is_transformer_like"] = df["function_group"].eq("vision_transformer").astype(int)
    df["is_mobile_optimized"] = df["function_group"].isin({"mobile_edge", "efficient_edge"}).astype(int)
    return df


def load_and_prepare_datasets(
    rpi5_path: Path | str,
    jetson_path: Path | str,
) -> pd.DataFrame:
    rpi_df = standardize_benchmark_dataset(rpi5_path, "Raspberry Pi 5", "raspberry_pi5")
    jetson_df = standardize_benchmark_dataset(jetson_path, "Jetson Nano", "jetson_nano")
    combined_df = pd.concat([rpi_df, jetson_df], ignore_index=True)
    combined_df["device_family"] = combined_df["device_type"].map(
        {
            "raspberry_pi5": "cpu_edge",
            "jetson_nano": "gpu_edge",
        }
    )
    return combined_df


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidate_columns = [col for col in df.columns if col not in EXCLUDED_COLUMNS]
    numeric_features = [
        col
        for col in candidate_columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in {"energy_avg_mwh"}
    ]
    categorical_features = [
        col
        for col in ("device_type", "device_family", "model_family", "function_group")
        if col in df.columns
    ]
    feature_columns = numeric_features + categorical_features
    return numeric_features, categorical_features, feature_columns


def build_stratify_labels(df: pd.DataFrame, target_col: str = "energy_avg_mwh", bins: int = 6) -> pd.Series:
    effective_bins = min(bins, max(2, df.shape[0] // 20))
    for current_bins in range(effective_bins, 1, -1):
        target_bins = pd.qcut(df[target_col], q=current_bins, duplicates="drop")
        labels = df["device_type"].astype(str) + "__" + target_bins.astype(str)
        counts = labels.value_counts()
        if counts.min() >= 2:
            return labels
    return df["device_type"].astype(str)


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "energy_avg_mwh",
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    stratify_labels = build_stratify_labels(df, target_col=target_col)
    train_indices, test_indices = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    train_df["stratify_label"] = stratify_labels.loc[train_indices].values
    test_df["stratify_label"] = stratify_labels.loc[test_indices].values
    return {
        "train_df": train_df,
        "test_df": test_df,
        "train_indices": np.asarray(train_indices),
        "test_indices": np.asarray(test_indices),
    }


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.where(np.abs(y_true_arr) < 1e-9, 1e-9, np.abs(y_true_arr))
    return float(np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100.0)


def smape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_arr) + np.abs(y_pred_arr) + 1e-9
    return float(np.mean(2.0 * np.abs(y_pred_arr - y_true_arr) / denominator) * 100.0)


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "rmse": rmse(y_true_arr, y_pred_arr),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "mape_pct": mape(y_true_arr, y_pred_arr),
        "smape_pct": smape(y_true_arr, y_pred_arr),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def build_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.5, random_state=random_state),
        "Elastic Net": ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=5_000, random_state=random_state),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=9, weights="distance"),
        "SVR (RBF)": SVR(C=15.0, epsilon=0.05, gamma="scale"),
        "Random Forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=450,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "HistGradient Boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=500,
            random_state=random_state,
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            learning_rate_init=0.002,
            max_iter=700,
            early_stopping=True,
            random_state=random_state,
        ),
    }


def _build_estimator(
    base_model: Any,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> TransformedTargetRegressor:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", base_model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def evaluate_sklearn_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    feature_columns: Sequence[str],
    target_col: str = "energy_avg_mwh",
    random_state: int = 42,
    selected_models: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    candidate_models = build_candidate_models(random_state=random_state)
    if selected_models is not None:
        candidate_models = {name: candidate_models[name] for name in selected_models}

    x_train = train_df[list(feature_columns)]
    y_train = train_df[target_col].values
    x_test = test_df[list(feature_columns)]
    y_test = test_df[target_col].values
    train_labels = train_df["stratify_label"].astype(str).values
    min_count = pd.Series(train_labels).value_counts().min()
    n_splits = max(2, min(5, int(min_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, Dict[str, Any]] = {}

    for model_name, model in candidate_models.items():
        estimator = _build_estimator(model, numeric_features, categorical_features)
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        cv_result = cross_validate(
            estimator,
            x_train,
            y_train,
            cv=cv.split(x_train, train_labels),
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        fitted_estimator = clone(estimator)
        fitted_estimator.fit(x_train, y_train)
        test_predictions = fitted_estimator.predict(x_test)
        metrics = compute_metrics(y_test, test_predictions)
        results_rows.append(
            {
                "model_name": model_name,
                "model_type": "sklearn",
                "cv_rmse_mean": float(-np.mean(cv_result["test_rmse"])),
                "cv_rmse_std": float(np.std(-cv_result["test_rmse"])),
                "cv_mae_mean": float(-np.mean(cv_result["test_mae"])),
                "cv_r2_mean": float(np.mean(cv_result["test_r2"])),
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "test_mape_pct": metrics["mape_pct"],
                "test_smape_pct": metrics["smape_pct"],
                "test_r2": metrics["r2"],
            }
        )
        artifacts[model_name] = {
            "model_name": model_name,
            "model_type": "sklearn",
            "estimator": fitted_estimator,
            "test_predictions": test_predictions,
            "test_truth": y_test,
            "test_index": test_df.index.to_numpy(),
            "feature_columns": list(feature_columns),
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "metrics": metrics,
        }

    results_df = pd.DataFrame(results_rows).sort_values(
        by=["test_rmse", "test_mae", "test_mape_pct"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return results_df, artifacts


def _normalize_adjacency(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    matrix = np.maximum(matrix, matrix.T)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def build_knn_adjacency(features: np.ndarray, n_neighbors: int = 12) -> np.ndarray:
    graph = kneighbors_graph(
        features,
        n_neighbors=min(n_neighbors, max(2, features.shape[0] - 1)),
        mode="connectivity",
        include_self=True,
    )
    adjacency = graph.toarray()
    adjacency = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    return _normalize_adjacency(adjacency)


class GraphEnergyRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2 * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        ax = adjacency @ x
        h1 = torch.cat([x, ax], dim=1)
        h1 = torch.relu(self.bn1(self.fc1(h1)))
        h1 = self.dropout(h1)

        ah1 = adjacency @ h1
        h2 = torch.cat([h1, ah1], dim=1)
        h2 = torch.relu(self.bn2(self.fc2(h2)))
        h2 = self.dropout(h2)

        ah2 = adjacency @ h2
        out = self.out(torch.cat([h2, ah2], dim=1)).squeeze(1)
        return out


def _transform_feature_frame(
    df: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    feature_columns: Sequence[str],
    train_indices: Sequence[int],
) -> Tuple[ColumnTransformer, np.ndarray]:
    preprocessor = make_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(df.loc[list(train_indices), list(feature_columns)])
    dense_matrix = np.asarray(preprocessor.transform(df[list(feature_columns)]), dtype=np.float32)
    return preprocessor, dense_matrix


def _train_graph_model(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    random_state: int = 42,
    max_epochs: int = 250,
    patience: int = 35,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    set_global_seed(random_state)
    device = torch.device("cpu")
    adjacency = build_knn_adjacency(features, n_neighbors=12)
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(np.log1p(targets), dtype=torch.float32, device=device)

    train_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=device)
    train_mask[list(train_indices)] = True
    val_mask[list(validation_indices)] = True

    model = GraphEnergyRegressor(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.2)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_tensor, adjacency_tensor)
        train_loss = criterion(predictions[train_mask], y_tensor[train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(x_tensor, adjacency_tensor)
            validation_loss = criterion(validation_predictions[val_mask], y_tensor[val_mask]).item()
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation_loss),
            }
        )

        if validation_loss + 1e-6 < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        log_predictions = model(x_tensor, adjacency_tensor).cpu().numpy()
    predictions = np.expm1(log_predictions)
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    return {
        "model": model,
        "predictions": predictions,
        "adjacency": adjacency,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_gnn_regressor(
    df: pd.DataFrame,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    feature_columns: Sequence[str],
    target_col: str = "energy_avg_mwh",
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_df = df.loc[list(train_indices)].copy()
    preprocessor, all_features = _transform_feature_frame(
        df,
        numeric_features,
        categorical_features,
        feature_columns,
        train_indices=train_indices,
    )
    targets = df[target_col].to_numpy(dtype=float)
    train_labels = build_stratify_labels(train_df, target_col=target_col)
    min_count = train_labels.value_counts().min()
    n_splits = max(2, min(5, int(min_count)))
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_rmse_scores: List[float] = []
    cv_mae_scores: List[float] = []
    cv_r2_scores: List[float] = []

    train_indices_arr = np.asarray(train_indices)
    for fold_id, (fold_train_pos, fold_val_pos) in enumerate(outer_cv.split(train_indices_arr, train_labels), start=1):
        fold_train_indices = train_indices_arr[fold_train_pos]
        fold_val_indices = train_indices_arr[fold_val_pos]
        fold_result = _train_graph_model(
            features=all_features,
            targets=targets,
            train_indices=fold_train_indices,
            validation_indices=fold_val_indices,
            random_state=random_state + fold_id,
        )
        fold_predictions = fold_result["predictions"][fold_val_indices]
        fold_truth = targets[fold_val_indices]
        fold_metrics = compute_metrics(fold_truth, fold_predictions)
        cv_rmse_scores.append(fold_metrics["rmse"])
        cv_mae_scores.append(fold_metrics["mae"])
        cv_r2_scores.append(fold_metrics["r2"])

    inner_train_indices, inner_val_indices = train_test_split(
        train_indices_arr,
        test_size=0.2,
        random_state=random_state,
        stratify=train_labels,
    )
    final_result = _train_graph_model(
        features=all_features,
        targets=targets,
        train_indices=inner_train_indices,
        validation_indices=inner_val_indices,
        random_state=random_state,
    )
    test_predictions = final_result["predictions"][list(test_indices)]
    test_truth = targets[list(test_indices)]
    metrics = compute_metrics(test_truth, test_predictions)

    result_row = {
        "model_name": "Similarity Graph Neural Network",
        "model_type": "gnn",
        "cv_rmse_mean": float(np.mean(cv_rmse_scores)),
        "cv_rmse_std": float(np.std(cv_rmse_scores)),
        "cv_mae_mean": float(np.mean(cv_mae_scores)),
        "cv_r2_mean": float(np.mean(cv_r2_scores)),
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_mape_pct": metrics["mape_pct"],
        "test_smape_pct": metrics["smape_pct"],
        "test_r2": metrics["r2"],
    }

    artifact = {
        "model_name": result_row["model_name"],
        "model_type": "gnn",
        "model_state_dict": {key: value.cpu() for key, value in final_result["model"].state_dict().items()},
        "model_input_dim": int(all_features.shape[1]),
        "preprocessor": preprocessor,
        "adjacency": final_result["adjacency"],
        "all_features": all_features,
        "all_indices": df.index.to_numpy(),
        "feature_columns": list(feature_columns),
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "test_predictions": test_predictions,
        "test_truth": test_truth,
        "test_index": np.asarray(test_indices),
        "metrics": metrics,
        "history": final_result["history"],
    }
    return result_row, artifact


def predict_with_artifact(artifact: Dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    if artifact["model_type"] == "sklearn":
        estimator = artifact["estimator"]
        return estimator.predict(frame[artifact["feature_columns"]])

    preprocessor = artifact["preprocessor"]
    dense_matrix = np.asarray(preprocessor.transform(frame[artifact["feature_columns"]]), dtype=np.float32)
    adjacency = build_knn_adjacency(dense_matrix, n_neighbors=12)
    model = GraphEnergyRegressor(input_dim=artifact["model_input_dim"])
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.tensor(dense_matrix, dtype=torch.float32),
            torch.tensor(adjacency, dtype=torch.float32),
        ).numpy()
    return np.clip(np.expm1(predictions), a_min=0.0, a_max=None)


def compute_artifact_permutation_importance(
    artifact: Dict[str, Any],
    frame: pd.DataFrame,
    target_col: str,
    feature_columns: Sequence[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline_predictions = predict_with_artifact(artifact, frame)
    baseline_rmse = rmse(frame[target_col].values, baseline_predictions)
    rows: List[Dict[str, Any]] = []

    for feature_name in feature_columns:
        deltas: List[float] = []
        for _ in range(n_repeats):
            shuffled_frame = frame.copy()
            shuffled_frame[feature_name] = rng.permutation(shuffled_frame[feature_name].values)
            shuffled_predictions = predict_with_artifact(artifact, shuffled_frame)
            shuffled_rmse = rmse(shuffled_frame[target_col].values, shuffled_predictions)
            deltas.append(shuffled_rmse - baseline_rmse)
        rows.append(
            {
                "feature_name": feature_name,
                "rmse_delta_mean": float(np.mean(deltas)),
                "rmse_delta_std": float(np.std(deltas)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_delta_mean", ascending=False).reset_index(drop=True)


def choose_best_model(
    comparison_df: pd.DataFrame,
    sklearn_artifacts: Dict[str, Dict[str, Any]],
    gnn_artifact: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    best_row = comparison_df.sort_values(
        by=["test_rmse", "test_mae", "test_mape_pct"],
        ascending=[True, True, True],
    ).iloc[0]
    best_name = str(best_row["model_name"])
    if best_name in sklearn_artifacts:
        return best_name, sklearn_artifacts[best_name]
    if gnn_artifact and best_name == gnn_artifact["model_name"]:
        return best_name, gnn_artifact
    raise KeyError(f"Unable to resolve artifact for best model: {best_name}")


def save_best_model(
    model_name: str,
    artifact: Dict[str, Any],
    output_dir: Path | str = DEFAULT_MODEL_OUTPUT_DIR,
    comparison_df: Optional[pd.DataFrame] = None,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")

    if artifact["model_type"] == "sklearn":
        model_path = output_path / f"energy_predictor_multidevice_{safe_name}.joblib"
        joblib.dump(artifact["estimator"], model_path)
    else:
        model_path = output_path / f"energy_predictor_multidevice_{safe_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_type": artifact["model_type"],
                "model_state_dict": artifact["model_state_dict"],
                "model_input_dim": artifact["model_input_dim"],
                "preprocessor": artifact["preprocessor"],
                "feature_columns": artifact["feature_columns"],
                "numeric_features": artifact["numeric_features"],
                "categorical_features": artifact["categorical_features"],
            },
            model_path,
        )

    metadata = {
        "model_name": model_name,
        "model_type": artifact["model_type"],
        "feature_columns": artifact["feature_columns"],
        "numeric_features": artifact["numeric_features"],
        "categorical_features": artifact["categorical_features"],
        "metrics": artifact["metrics"],
        "test_index": [int(idx) for idx in artifact["test_index"]],
        "artifact_path": str(model_path),
    }
    metadata_path = output_path / f"energy_predictor_multidevice_{safe_name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    comparison_path = ""
    if comparison_df is not None:
        comparison_output = output_path / "energy_predictor_multidevice_model_comparison.csv"
        comparison_df.to_csv(comparison_output, index=False)
        comparison_path = str(comparison_output)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "comparison_path": comparison_path,
    }
