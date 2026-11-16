from pathlib import Path
import json
import logging
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
LIGHTGBM_PATH = ROOT / "reports" / "predictions" / "lightgbm_predictions.csv"

OUTPUT_DIR = ROOT / "reports" / "predictions"
METRICS_DIR = ROOT / "reports" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def build_holiday_calendar(history):
    """Build Prophet holiday data from known M5 calendar events."""

    holiday_frames = []

    for column in ["event_name_1", "event_name_2"]:
        if column not in history.columns:
            continue

        events = history.loc[history[column].notna(), ["date", column]].copy()

        events = events.rename(columns={"date": "ds", column: "holiday"})

        holiday_frames.append(events)

    if not holiday_frames:
        return None

    holidays = pd.concat(holiday_frames, ignore_index=True)
    holidays = holidays.drop_duplicates(subset=["ds", "holiday"])

    holidays["ds"] = pd.to_datetime(holidays["ds"])

    return holidays


def calculate_metrics(actual, predicted, rmsse_values):
    """Calculate forecast evaluation metrics."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    wape = (
        np.abs(errors).sum() / np.abs(actual).sum() if np.abs(actual).sum() > 0 else 0.0
    )

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    return {
        "wape": float(wape),
        "mae": float(mae),
        "rmse": float(rmse),
        "mean_per_series_rmsse": float(np.mean(rmsse_values)) if rmsse_values else None,
        "rmsse_series_count": len(rmsse_values),
        "rmsse_definition": ("Unweighted mean of per-series RMSSE values"),
    }


def train_prophet():
    print("\nLoading historical data...")
    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"])

    print("Loading LightGBM holdout dates...")
    lightgbm = pd.read_csv(LIGHTGBM_PATH, parse_dates=["date"])

    forecast_start = lightgbm["date"].min()

    training_history = history[history["date"] < forecast_start].copy()

    holidays = build_holiday_calendar(history)

    print("\n======================================")
    print("PROPHET FORECASTING")
    print("======================================")
    print(f"Training cutoff : {forecast_start.date()}")
    print(f"Training rows   : {len(training_history):,}")
    print(f"Holdout rows    : {len(lightgbm):,}")
    print(f"Holiday events  : {len(holidays) if holidays is not None else 0}")
    print("======================================")

    prediction_frames = []
    rmsse_values = []

    combinations = lightgbm[["item_id", "store_id"]].drop_duplicates()

    total_combinations = len(combinations)

    for number, (_, combination) in enumerate(combinations.iterrows(), start=1):
        item = combination["item_id"]
        store = combination["store_id"]

        train = training_history[
            (training_history["item_id"] == item)
            & (training_history["store_id"] == store)
        ][["date", "sales"]].copy()

        test = lightgbm[
            (lightgbm["item_id"] == item) & (lightgbm["store_id"] == store)
        ][["date", "sales"]].copy()

        if train.empty or test.empty:
            continue

        train = train.rename(columns={"date": "ds", "sales": "y"})

        train = train.sort_values("ds")
        test = test.sort_values("date")

        train["y"] = pd.to_numeric(train["y"], errors="coerce")

        train = train.dropna(subset=["y"])

        if len(train) < 60:
            continue

        model = Prophet(
            holidays=holidays,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            n_changepoints=10,
            uncertainty_samples=0,
        )

        model.fit(train)

        future = pd.DataFrame({"ds": test["date"].values})

        forecast = model.predict(future)

        result = pd.DataFrame(
            {
                "date": test["date"].values,
                "item_id": item,
                "store_id": store,
                "sales": test["sales"].values,
                "predicted_sales": np.maximum(0, forecast["yhat"].values),
            }
        )

        prediction_frames.append(result)

        # Training-series scale for RMSSE.
        differences = train["y"].diff().dropna()

        scale = np.mean(differences**2)

        if scale > 0:
            mse = np.mean(
                (result["sales"].values - result["predicted_sales"].values) ** 2
            )

            rmsse_values.append(float(np.sqrt(mse / scale)))

        if number % 20 == 0 or number == total_combinations:
            print(f"Completed {number}/{total_combinations} " f"item-store models")

    if not prediction_frames:
        raise RuntimeError("No Prophet predictions were generated.")

    predictions = pd.concat(prediction_frames, ignore_index=True)

    predictions = predictions.sort_values(["store_id", "item_id", "date"])

    metrics = calculate_metrics(
        predictions["sales"],
        predictions["predicted_sales"],
        rmsse_values,
    )

    prediction_path = OUTPUT_DIR / "prophet_predictions.csv"
    metrics_path = METRICS_DIR / "prophet_metrics.json"

    predictions.to_csv(prediction_path, index=False)

    summary = {
        "model": "Prophet",
        "training_cutoff": str(forecast_start.date()),
        "holdout_days": int(predictions["date"].nunique()),
        "series_trained": int(
            predictions[["item_id", "store_id"]].drop_duplicates().shape[0]
        ),
        "prediction_rows": int(len(predictions)),
        "metrics": metrics,
        "features": [
            "Weekly seasonality",
            "Yearly seasonality",
            "Known M5 holiday and event dates",
        ],
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n======================================")
    print("PROPHET TRAINING COMPLETED")
    print("======================================")
    print(f"Models trained : {summary['series_trained']}")
    print(f"Forecast rows  : {len(predictions):,}")
    print(f"WAPE           : {metrics['wape']:.4f}")
    print(f"MAE            : {metrics['mae']:.4f}")
    print(f"RMSE           : {metrics['rmse']:.4f}")

    if metrics["mean_per_series_rmsse"] is not None:
        print("Mean series RMSSE : " f"{metrics['mean_per_series_rmsse']:.4f}")

    print(f"\nPredictions saved: {prediction_path}")
    print(f"Metrics saved    : {metrics_path}")
    print("======================================")


if __name__ == "__main__":
    train_prophet()
