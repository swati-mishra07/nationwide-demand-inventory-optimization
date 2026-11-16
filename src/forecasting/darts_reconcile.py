
from pathlib import Path

import json
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers.reconciliation import (
    BottomUpReconciliator,
)

ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
FORECAST_PATH = (
    ROOT / "reports" / "predictions"
    / "lightgbm_predictions_with_intervals.csv"
)
OUTPUT_DIR = ROOT / "reports" / "reconciliation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def reconcile_with_darts():
    print("\nLoading history and forecasts...")

    history = pd.read_csv(HISTORY_PATH)
    forecasts = pd.read_csv(FORECAST_PATH)

    required_history = {"item_id", "store_id", "state_id"}
    required_forecast = {"item_id", "store_id", "date", "predicted_sales"}

    if not required_history.issubset(history.columns):
        raise ValueError("Required hierarchy columns are missing from history.")

    if not required_forecast.issubset(forecasts.columns):
        raise ValueError("Required forecast columns are missing.")

    store_state = history[
        ["store_id", "state_id"]
    ].drop_duplicates()

    if store_state["store_id"].duplicated().any():
        raise ValueError("A store maps to multiple states.")

    forecasts = forecasts.merge(
        store_state,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    if forecasts["state_id"].isna().any():
        raise ValueError("Some stores have no state mapping.")

    forecasts["date"] = pd.to_datetime(forecasts["date"])
    forecasts["predicted_sales"] = (
        forecasts["predicted_sales"].clip(lower=0)
    )

    # Create a unique component name for every item-store series.
    forecasts["leaf"] = (
        forecasts["item_id"].astype(str)
        + "__"
        + forecasts["store_id"].astype(str)
    )

    if forecasts.duplicated(["date", "leaf"]).any():
        raise ValueError("Duplicate date/item/store forecasts found.")

    print("Building item-store, store, state and national hierarchy...")

    leaf_wide = forecasts.pivot(
        index="date",
        columns="leaf",
        values="predicted_sales",
    ).sort_index()

    leaf_to_store = (
        forecasts[["leaf", "store_id"]]
        .drop_duplicates()
        .set_index("leaf")["store_id"]
        .to_dict()
    )

    leaf_columns = list(leaf_wide.columns)
    wide = leaf_wide.copy()

    hierarchy = {}

    # Item-store -> Store
    for leaf in leaf_columns:
        store_component = f"store__{leaf_to_store[leaf]}"
        hierarchy[leaf] = [store_component]

    # Store -> State
    for store_id, state_id in store_state.itertuples(index=False):
        store_component = f"store__{store_id}"
        state_component = f"state__{state_id}"

        store_leaves = [
            leaf for leaf in leaf_columns
            if leaf_to_store[leaf] == store_id
        ]

        wide[store_component] = wide[store_leaves].sum(axis=1)
        hierarchy[store_component] = [state_component]

    # State -> National total
    state_components = []

    for state_id in sorted(store_state["state_id"].unique()):
        state_component = f"state__{state_id}"
        store_components = [
            f"store__{store_id}"
            for store_id in store_state.loc[
                store_state["state_id"] == state_id, "store_id"
            ].unique()
        ]

        wide[state_component] = wide[store_components].sum(axis=1)
        hierarchy[state_component] = ["national_total"]
        state_components.append(state_component)

    wide["national_total"] = wide[state_components].sum(axis=1)

    components = list(wide.columns)

    # Darts requires every hierarchy component to be present in the series.
    series = TimeSeries(
        times=pd.DatetimeIndex(wide.index),
        values=wide[components].to_numpy(dtype=float),
        components=components,
        hierarchy=hierarchy,
    )

    print("Applying Darts BottomUpReconciliator...")

    reconciliator = BottomUpReconciliator()
    reconciled = reconciliator.transform(series)

    result = reconciled.to_dataframe().reset_index()
    result = result.rename(columns={result.columns[0]: "date"})

    output_path = OUTPUT_DIR / "darts_reconciled_forecasts.csv"
    result.to_csv(output_path, index=False)

    # Verify the hierarchy after reconciliation.
    max_difference = 0.0

    for store_id, state_id in store_state.itertuples(index=False):
        store_component = f"store__{store_id}"
        state_component = f"state__{state_id}"

        leaves = [
            leaf for leaf in leaf_columns
            if leaf_to_store[leaf] == store_id
        ]

        difference = (
            result[leaves].sum(axis=1) - result[store_component]
        ).abs().max()

        max_difference = max(max_difference, float(difference))

    state_sum = result[state_components].sum(axis=1)
    national_difference = (
        state_sum - result["national_total"]
    ).abs().max()

    max_difference = max(
        max_difference, float(national_difference)
    )

    summary = {
        "method": "Darts BottomUpReconciliator",
        "darts_version": "0.48.1",
        "item_store_series": len(leaf_columns),
        "stores": int(store_state["store_id"].nunique()),
        "states": int(store_state["state_id"].nunique()),
        "forecast_dates": int(len(result)),
        "maximum_coherence_difference": max_difference,
        "coherence_check_passed": bool(max_difference < 1e-6),
        "output_file": str(output_path.relative_to(ROOT)),
    }

    with open(
        OUTPUT_DIR / "darts_reconciliation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n========== DARTS RECONCILIATION ==========")
    print(f"Item-store series : {summary['item_store_series']}")
    print(f"Stores            : {summary['stores']}")
    print(f"States            : {summary['states']}")
    print(f"Forecast dates    : {summary['forecast_dates']}")
    print(f"Max difference    : {max_difference:.10f}")
    print(f"Coherence passed  : {summary['coherence_check_passed']}")
    print(f"Saved             : {output_path}")
    print("==========================================")


if __name__ == "__main__":
    reconcile_with_darts()
