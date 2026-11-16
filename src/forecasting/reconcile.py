
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
FORECAST_PATH = (
    ROOT / "reports" / "predictions"
    / "lightgbm_predictions_with_intervals.csv"
)

OUTPUT_DIR = ROOT / "reports" / "reconciliation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reconcile_forecasts():
    print("\nLoading historical hierarchy...")
    history = pd.read_csv(HISTORY_PATH)

    print("Loading item-store forecasts...")
    forecasts = pd.read_csv(FORECAST_PATH)

    required_history = {"item_id", "store_id", "state_id"}
    required_forecast = {
        "item_id", "store_id", "date",
        "predicted_sales", "lower_bound", "upper_bound"
    }

    if not required_history.issubset(history.columns):
        raise ValueError(
            f"History is missing columns: "
            f"{required_history - set(history.columns)}"
        )

    if not required_forecast.issubset(forecasts.columns):
        raise ValueError(
            f"Forecasts are missing columns: "
            f"{required_forecast - set(forecasts.columns)}"
        )

    # Each store must belong to exactly one state.
    store_hierarchy = (
        history[["store_id", "state_id"]]
        .drop_duplicates()
    )

    if store_hierarchy["store_id"].duplicated().any():
        raise ValueError("A store maps to multiple states.")

    print("\nMapping stores to states...")
    forecasts = forecasts.merge(
        store_hierarchy,
        on="store_id",
        how="left",
        validate="many_to_one"
    )

    if forecasts["state_id"].isna().any():
        raise ValueError("Some forecast stores have no state mapping.")

    # Level 1: Item-store forecasts.
    item_store = forecasts[
        [
            "date", "state_id", "store_id", "item_id",
            "predicted_sales", "lower_bound", "upper_bound"
        ]
    ].copy()

    # Level 2: Aggregate all item forecasts within each store.
    store_forecasts = (
        item_store.groupby(
            ["date", "state_id", "store_id"], as_index=False
        )[
            ["predicted_sales", "lower_bound", "upper_bound"]
        ].sum()
    )

    # Level 3: Aggregate all store forecasts within each state.
    state_forecasts = (
        store_forecasts.groupby(
            ["date", "state_id"], as_index=False
        )[
            ["predicted_sales", "lower_bound", "upper_bound"]
        ].sum()
    )

    # Coherence check: state total must equal its store totals.
    store_totals = (
        store_forecasts.groupby(
            ["date", "state_id"]
        )["predicted_sales"].sum()
    )

    state_totals = (
        state_forecasts.set_index(
            ["date", "state_id"]
        )["predicted_sales"]
    )

    difference = (store_totals - state_totals).abs().max()

    item_store.to_csv(
        OUTPUT_DIR / "item_store_forecasts.csv", index=False
    )
    store_forecasts.to_csv(
        OUTPUT_DIR / "store_forecasts.csv", index=False
    )
    state_forecasts.to_csv(
        OUTPUT_DIR / "state_forecasts.csv", index=False
    )

    summary = {
        "method": "Bottom-up aggregation",
        "item_store_forecast_rows": int(len(item_store)),
        "store_forecast_rows": int(len(store_forecasts)),
        "state_forecast_rows": int(len(state_forecasts)),
        "stores": int(store_forecasts["store_id"].nunique()),
        "states": int(state_forecasts["state_id"].nunique()),
        "total_predicted_demand": float(
            item_store["predicted_sales"].sum()
        ),
        "maximum_state_reconciliation_difference": float(difference),
        "coherence_check_passed": bool(difference < 1e-8),
    }

    with open(
        OUTPUT_DIR / "reconciliation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n======================================")
    print("HIERARCHICAL RECONCILIATION COMPLETED")
    print("======================================")
    print(f"Item-store rows : {len(item_store):,}")
    print(f"Store rows      : {len(store_forecasts):,}")
    print(f"State rows      : {len(state_forecasts):,}")
    print(f"Stores          : {summary['stores']}")
    print(f"States          : {summary['states']}")
    print(f"Total demand    : {summary['total_predicted_demand']:.2f}")
    print(f"Max difference  : {difference:.10f}")
    print(f"Coherence check : {summary['coherence_check_passed']}")
    print(f"\nSaved files in: {OUTPUT_DIR}")
    print("======================================")


if __name__ == "__main__":
    reconcile_forecasts()
