#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taylor-root 데이터셋용 가벼운 supervised learning baseline 비교 스크립트

기능
- NPZ train/val/test split 로드
- coeffs -> root0/root1/root2 회귀
- Ridge / KNN / RandomForest / GradientBoosting / SVR / MLPRegressor 비교
- MAE / RMSE / R2 / residual(mean, median, p90) / threshold success(optional) 계산
- CSV 저장

예시
python supervised_learning_review_root0.py \
  --data_dir /path/to/taylor_data_physchem_v4_deg25 \
  --models ridge knn rf gbr svr mlp \
  --target root0 \
  --residual_thr 1e-6 \
  --out_csv results_supervised_review.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


def poly_eval_batch(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """coeffs: (N, D+1), x: (N,) -> P(x)"""
    y = np.zeros_like(x, dtype=np.float64)
    for k in range(coeffs.shape[1] - 1, -1, -1):
        y = y * x + coeffs[:, k]
    return y


def load_split(npz_path: Path, target_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    X = data["coeffs"].astype(np.float64)
    y = data[target_key].reshape(-1).astype(np.float64)
    return X, y, data["coeffs"].astype(np.float64)


def filter_finite(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(y)
    mask &= np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask], mask


def build_models(random_state: int) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models["ridge"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    models["knn"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=7, weights="distance"))
    ])

    models["rf"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    models["gbr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ))
    ])

    models["svr"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"))
    ])

    models["mlp"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=512,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ))
    ])

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, coeffs: np.ndarray, residual_thr: float) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    residual = np.abs(poly_eval_batch(coeffs, y_pred.astype(np.float64)))
    success = float(np.mean(residual <= residual_thr))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
        "success_at_thr": success,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="NPZ split 폴더")
    p.add_argument("--target", type=str, default="root0", choices=["root0", "root1", "root2"])
    p.add_argument("--models", nargs="+", default=["ridge", "knn", "rf", "gbr", "svr", "mlp"])
    p.add_argument("--residual_thr", type=float, default=1e-6)
    p.add_argument("--out_csv", type=str, default="results_supervised_review.csv")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    train_path = next(iter(sorted(data_dir.glob("*_train.npz"))))
    val_path = next(iter(sorted(data_dir.glob("*_val.npz"))))
    test_path = next(iter(sorted(data_dir.glob("*_test.npz"))))

    X_train, y_train, _ = load_split(train_path, args.target)
    X_val, y_val, _ = load_split(val_path, args.target)
    X_test, y_test, coeffs_test = load_split(test_path, args.target)

    X_train, y_train, mask_tr = filter_finite(X_train, y_train)
    X_val, y_val, mask_va = filter_finite(X_val, y_val)
    X_test, y_test, mask_te = filter_finite(X_test, y_test)
    coeffs_test = coeffs_test[mask_te]

    # 간단히 train+val 합쳐서 최종 학습
    X_fit = np.concatenate([X_train, X_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    available_models = build_models(args.random_state)
    selected = {k: available_models[k] for k in args.models if k in available_models}
    if not selected:
        raise ValueError("선택된 모델이 없습니다.")

    rows: List[Dict[str, float]] = []

    print(f"[INFO] train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, target={args.target}")
    print(f"[INFO] residual threshold = {args.residual_thr}")

    for name, model in selected.items():
        print(f"\n[MODEL] {name}")
        model.fit(X_fit, y_fit)
        pred = model.predict(X_test)
        metrics = compute_metrics(y_test, pred, coeffs_test, args.residual_thr)
        row = {"model": name, **metrics}
        rows.append(row)
        print(row)

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "mae", "rmse", "r2", "residual_mean", "residual_median", "residual_p90", "success_at_thr"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n[SAVE] {out_csv.resolve()}")


if __name__ == "__main__":
    main()
