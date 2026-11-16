from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
FORECAST_PATH = ROOT / "reports" / "predictions" / "lightgbm_predictions.csv"

OUTPUT_DIR = ROOT / "reports" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

Z_VALUE = 1.645


def create_prediction_intervals():
    print("\nLoading historical demand...")
    history = pd.read_csv(
        HISTORY_PATH,
        parse_dates=["date"],
    )

    print("Loading LightGBM predictions...")
    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    forecast_start = forecasts["date"].min()

    training_history = history[history["date"] < forecast_start].copy()

    print("\nEstimating historical demand variability...")

    variability = (
        training_history.groupby(["item_id", "store_id"])["sales"]
        .std()
        .reset_index(name="demand_std")
    )

    variability["demand_std"] = variability["demand_std"].fillna(0)

    result = forecasts.merge(
        variability,
        on=["item_id", "store_id"],
        how="left",
        validate="many_to_one",
    )

    result["demand_std"] = result["demand_std"].fillna(0)

    result["uncertainty_margin"] = Z_VALUE * result["demand_std"]

    result["lower_bound"] = np.maximum(
        0,
        result["predicted_sales"] - result["uncertainty_margin"],
    )

    result["upper_bound"] = result["predicted_sales"] + result["uncertainty_margin"]

    result["interval_level"] = "Approximate 90% variability band"

    output_path = OUTPUT_DIR / "lightgbm_predictions_with_intervals.csv"

    result.to_csv(output_path, index=False)

    print("\n======================================")
    print("PREDICTION INTERVALS COMPLETED")
    print("======================================")
    print(f"Forecast rows: {len(result):,}")
    print(f"Average predicted demand: " f"{result['predicted_sales'].mean():.3f}")
    print(f"Average lower bound: " f"{result['lower_bound'].mean():.3f}")
    print(f"Average upper bound: " f"{result['upper_bound'].mean():.3f}")
    print(f"Saved: {output_path}")
    print("======================================")

    return result


if __name__ == "__main__":
    create_prediction_intervals()
