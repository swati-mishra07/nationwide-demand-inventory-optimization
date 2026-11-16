
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "m5_features.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "reports" / "metrics"
PREDICTION_DIR = ROOT / "reports" / "predictions"

for folder in [MODEL_DIR, RESULT_DIR, PREDICTION_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def calculate_wape(actual, predicted):
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return 0.0

    return float(
        np.abs(actual - predicted).sum() / denominator
    )


def calculate_rmsse(actual, predicted, training_series):
    numerator = np.mean((actual - predicted) ** 2)

    differences = np.diff(training_series)
    denominator = np.mean(differences ** 2)

    if denominator == 0:
        return None

    return float(np.sqrt(numerator / denominator))


def train_lightgbm():
    print("\nLoading engineered features...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    df = df.sort_values(["date", "store_id", "item_id"])
    df = df.dropna(subset=["sales"])

    # Hold out the final 28 calendar days.
    cutoff = df["date"].max() - pd.Timedelta(days=27)

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    excluded = {
        "sales",
        "revenue",
        "date",
        "d",
        "id",
    }

    feature_columns = [
        col for col in df.columns
        if col not in excluded
    ]

    # Convert text columns into categorical features.
    for col in feature_columns:
        if (
            train[col].dtype == "object"
            or isinstance(train[col].dtype, pd.StringDtype)
        ):
            train[col] = train[col].fillna("Unknown").astype("category")
            test[col] = test[col].fillna("Unknown").astype("category")

            # Keep category codes consistent between train and test.
            categories = train[col].cat.categories
            test[col] = pd.Categorical(
                test[col],
                categories=categories,
            )

    # Remove features that contain only missing values.
    feature_columns = [
        col for col in feature_columns
        if train[col].notna().any()
    ]

    X_train = train[feature_columns]
    y_train = train["sales"]

    X_test = test[feature_columns]
    y_test = test["sales"]

    print(f"Training rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")
    print(f"Features: {len(feature_columns)}")
    print(f"Training cutoff: {cutoff.date()}")

    print("\nTraining LightGBM model...")

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)

    actual = y_test.to_numpy()
    wape = calculate_wape(actual, predictions)

    training_series = train["sales"].to_numpy()
    rmsse = calculate_rmsse(
        actual,
        predictions,
        training_series,
    )

    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))

    print("\n======================================")
    print("LIGHTGBM EVALUATION RESULTS")
    print("======================================")
    print(f"WAPE: {wape:.4f} ({wape * 100:.2f}%)")
    print(f"RMSSE: {rmsse:.4f}" if rmsse is not None else "RMSSE: N/A")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("======================================")

    model_path = MODEL_DIR / "lightgbm_model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "categories": {
                col: list(train[col].cat.categories)
                for col in feature_columns
                if isinstance(train[col].dtype, pd.CategoricalDtype)
            },
        },
        model_path,
    )

    output = test[
        ["date", "item_id", "store_id", "sales"]
    ].copy()

    output["predicted_sales"] = predictions
    output.to_csv(
        PREDICTION_DIR / "lightgbm_predictions.csv",
        index=False,
    )

    metrics = {
        "model": "LightGBM",
        "wape": wape,
        "rmsse": rmsse,
        "mae": mae,
        "rmse": rmse,
        "training_rows": len(train),
        "testing_rows": len(test),
        "features": len(feature_columns),
        "forecast_horizon_days": 28,
    }

    with open(RESULT_DIR / "lightgbm_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nModel saved: {model_path}")
    print("Predictions saved: reports/predictions/lightgbm_predictions.csv")
    print("Metrics saved: reports/metrics/lightgbm_metrics.json")

    return model, metrics


if __name__ == "__main__":
    train_lightgbm()
