
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
LIGHTGBM_PATH = ROOT / "reports" / "predictions" / "lightgbm_predictions.csv"

PREDICTION_DIR = ROOT / "reports" / "predictions"
METRIC_DIR = ROOT / "reports" / "metrics"

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


def calculate_wape(actual, predicted):
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return 0.0

    return float(
        np.abs(actual - predicted).sum() / denominator
    )


def evaluate(actual, predicted):
    return {
        "wape": calculate_wape(actual, predicted),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def run_naive_baseline():
    print("\nLoading preprocessed data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    cutoff = df["date"].max() - pd.Timedelta(days=27)

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    # Use the seven calendar days immediately before the test period.
    recent_start = cutoff - pd.Timedelta(days=7)

    recent_sales = train[
        train["date"] >= recent_start
    ]

    print("\nCalculating 7-day average demand...")
    baseline = (
        recent_sales.groupby(["item_id", "store_id"])["sales"]
        .mean()
        .reset_index(name="predicted_sales")
    )

    test = test.merge(
        baseline,
        on=["item_id", "store_id"],
        how="left",
        validate="many_to_one",
    )

    test["predicted_sales"] = test["predicted_sales"].fillna(0)

    actual = test["sales"].to_numpy()
    naive_predictions = test["predicted_sales"].to_numpy()

    naive_metrics = evaluate(actual, naive_predictions)

    naive_output = test[
        ["date", "item_id", "store_id", "sales", "predicted_sales"]
    ].copy()

    naive_output.to_csv(
        PREDICTION_DIR / "naive_predictions.csv",
        index=False,
    )

    print("\nLoading LightGBM predictions...")
    lightgbm = pd.read_csv(
        LIGHTGBM_PATH,
        parse_dates=["date"],
    )

    comparison = test[
        ["date", "item_id", "store_id", "sales"]
    ].merge(
        lightgbm[
            ["date", "item_id", "store_id", "predicted_sales"]
        ].rename(
            columns={"predicted_sales": "lightgbm_prediction"}
        ),
        on=["date", "item_id", "store_id"],
        how="inner",
        validate="one_to_one",
    )

    comparison["naive_prediction"] = naive_predictions

    lightgbm_metrics = evaluate(
        comparison["sales"].to_numpy(),
        comparison["lightgbm_prediction"].to_numpy(),
    )

    naive_metrics = evaluate(
        comparison["sales"].to_numpy(),
        comparison["naive_prediction"].to_numpy(),
    )

    results = {
        "evaluation_period_start": str(test["date"].min().date()),
        "evaluation_period_end": str(test["date"].max().date()),
        "observations": len(comparison),
        "naive_7_day_average": naive_metrics,
        "lightgbm": lightgbm_metrics,
    }

    with open(
        METRIC_DIR / "model_comparison.json",
        "w",
    ) as file:
        json.dump(results, file, indent=4)

    print("\n======================================")
    print("MODEL COMPARISON")
    print("======================================")
    print(f"Evaluation observations: {len(comparison):,}")
    print(
        f"Naive WAPE: "
        f"{naive_metrics['wape'] * 100:.2f}%"
    )
    print(
        f"LightGBM WAPE: "
        f"{lightgbm_metrics['wape'] * 100:.2f}%"
    )
    print(
        f"Naive MAE: {naive_metrics['mae']:.4f}"
    )
    print(
        f"LightGBM MAE: {lightgbm_metrics['mae']:.4f}"
    )
    print(
        f"Naive RMSE: {naive_metrics['rmse']:.4f}"
    )
    print(
        f"LightGBM RMSE: {lightgbm_metrics['rmse']:.4f}"
    )
    print("======================================")

    print("\nSaved:")
    print("reports/predictions/naive_predictions.csv")
    print("reports/metrics/model_comparison.json")


if __name__ == "__main__":
    run_naive_baseline()
