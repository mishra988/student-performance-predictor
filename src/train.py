import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from data_generator import generate_student_dataset
from data_preprocessing import prepare_data

def train_and_benchmark_models(csv_path: str = None, models_dir: str = None) -> Dict[str, Any]:
    """
    Trains multiple regression algorithms, benchmarks test performance,
    selects the champion model, and serializes artifacts.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "raw", "student_data.csv")
    if models_dir is None:
        models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load or generate dataset
    if not os.path.exists(csv_path):
        print(f"[DATA] Dataset not found at {csv_path}. Generating realistic dataset...")
        df = generate_student_dataset(n_samples=2500, output_path=csv_path)
    else:
        print(f"[DATA] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocess & Feature Engineer
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    print(f"Engineered input features: {len(feature_names)}")
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 3. Model Zoo Candidates
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            subsample=0.85,
            random_state=42
        ),
        "Ridge Regression": Ridge(alpha=2.5),
        "Linear Regression": LinearRegression(),
        "Support Vector Regressor": SVR(C=3.0, epsilon=0.15)
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    best_model_obj = None

    print("\n" + "=" * 70)
    print(f"{'MODEL':<26} | {'CV R2 (5-Fold)':<14} | {'TEST R2':<10} | {'RMSE':<8} | {'MAE':<8}")
    print("=" * 70)

    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=2)
        mean_cv_r2 = cv_scores.mean()

        # Fit on full training set
        model.fit(X_train, y_train)

        # Predictions on unseen test set
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[name] = {
            "cv_r2_mean": float(mean_cv_r2),
            "cv_r2_std": float(cv_scores.std()),
            "test_r2": float(test_r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape)
        }

        print(f"{name:<26} | {mean_cv_r2:.4f} (+/-{cv_scores.std():.3f}) | {test_r2:.4f}     | {rmse:.3f}  | {mae:.3f}")

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model_name = name
            best_model_obj = model

    print("=" * 70)
    print(f"[CHAMPION] Best Model: {best_model_name} (Test R2 = {best_r2:.4f})")

    # 4. Save artifacts
    model_path = os.path.join(models_dir, "student_performance_model.joblib")
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model_obj, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    metadata = {
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics": results[best_model_name],
        "all_benchmarks": results,
        "n_samples": int(df.shape[0]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0])
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVE] Saved champion model to: {model_path}")
    print(f"[SAVE] Saved preprocessor to: {preprocessor_path}")
    print(f"[SAVE] Saved metadata to: {metadata_path}\n")

    return {
        "best_model": best_model_obj,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "df": df
    }

if __name__ == "__main__":
    train_and_benchmark_models()
