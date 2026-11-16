
from pathlib import Path
import json
import math

import pandas as pd
from ortools.sat.python import cp_model


ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
FORECAST_PATH = ROOT / "reports" / "predictions" / "lightgbm_predictions.csv"

OUTPUT_DIR = ROOT / "reports" / "inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configurable planning assumptions.
LEAD_TIME_DAYS = 7
SERVICE_LEVEL_Z = 1.65

OPENING_INVENTORY = 0
PURCHASE_COST = 1.00
HOLDING_COST = 0.20
STOCKOUT_COST = 20.00

CAPACITY_MULTIPLIER = 1.20


def optimize_inventory():
    print("\nLoading historical demand...")
    history = pd.read_csv(
        HISTORY_PATH,
        parse_dates=["date"],
    )

    print("Loading LightGBM forecasts...")
    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    forecast_start = forecasts["date"].min()

    # Use only historical observations before the forecast period.
    training_history = history[
        history["date"] < forecast_start
    ].copy()

    # Forecast total demand over the 28-day planning horizon.
    demand = (
        forecasts.groupby(["item_id", "store_id"])["predicted_sales"]
        .sum()
        .reset_index(name="forecast_demand")
    )

    # Estimate daily demand variability from historical observations.
    variability = (
        training_history.groupby(["item_id", "store_id"])["sales"]
        .std()
        .reset_index(name="daily_demand_std")
    )

    planning = demand.merge(
        variability,
        on=["item_id", "store_id"],
        how="left",
        validate="one_to_one",
    )

    planning["daily_demand_std"] = (
        planning["daily_demand_std"].fillna(0)
    )

    planning["average_daily_forecast"] = (
        planning["forecast_demand"] / forecasts["date"].nunique()
    )

    planning["safety_stock"] = (
        SERVICE_LEVEL_Z
        * planning["daily_demand_std"]
        * math.sqrt(LEAD_TIME_DAYS)
    ).apply(math.ceil)

    planning["forecast_demand_units"] = (
        planning["forecast_demand"].apply(math.ceil)
    )

    planning["target_demand"] = (
        planning["forecast_demand_units"]
        + planning["safety_stock"]
    )

    planning["reorder_point"] = (
        planning["average_daily_forecast"] * LEAD_TIME_DAYS
        + planning["safety_stock"]
    ).apply(math.ceil)

    print("\nBuilding OR-Tools optimization model...")

    model = cp_model.CpModel()

    order_vars = []
    fulfilled_vars = []
    ending_vars = []
    shortage_vars = []

    for index, row in planning.iterrows():
        target = int(row["target_demand"])

        # Capacity limit: maximum 120% of the calculated target.
        capacity = max(1, math.ceil(target * CAPACITY_MULTIPLIER))

        order = model.NewIntVar(
            0, capacity, f"order_{index}"
        )
        fulfilled = model.NewIntVar(
            0, target, f"fulfilled_{index}"
        )
        ending = model.NewIntVar(
            0, capacity + OPENING_INVENTORY, f"ending_{index}"
        )
        shortage = model.NewIntVar(
            0, target, f"shortage_{index}"
        )

        # Inventory balance.
        model.Add(
            order + OPENING_INVENTORY == fulfilled + ending
        )

        # Demand must be fulfilled or recorded as shortage.
        model.Add(fulfilled + shortage == target)

        # Store variables for the objective and output.
        order_vars.append(order)
        fulfilled_vars.append(fulfilled)
        ending_vars.append(ending)
        shortage_vars.append(shortage)

    # Scale costs to integers for CP-SAT.
    purchase_weight = int(PURCHASE_COST * 100)
    holding_weight = int(HOLDING_COST * 100)
    shortage_weight = int(STOCKOUT_COST * 100)

    model.Minimize(
        sum(purchase_weight * var for var in order_vars)
        + sum(holding_weight * var for var in ending_vars)
        + sum(shortage_weight * var for var in shortage_vars)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"Inventory optimization failed. Solver status: "
            f"{solver.StatusName(status)}"
        )

    print(f"Solver status: {solver.StatusName(status)}")

    planning["recommended_order_qty"] = [
        solver.Value(var) for var in order_vars
    ]
    planning["fulfilled_demand"] = [
        solver.Value(var) for var in fulfilled_vars
    ]
    planning["ending_inventory"] = [
        solver.Value(var) for var in ending_vars
    ]
    planning["expected_shortage"] = [
        solver.Value(var) for var in shortage_vars
    ]

    planning["estimated_cost"] = (
        planning["recommended_order_qty"] * PURCHASE_COST
        + planning["ending_inventory"] * HOLDING_COST
        + planning["expected_shortage"] * STOCKOUT_COST
    )

    planning["opening_inventory_assumption"] = OPENING_INVENTORY
    planning["lead_time_days"] = LEAD_TIME_DAYS

    output_path = OUTPUT_DIR / "inventory_recommendations.csv"
    planning.to_csv(output_path, index=False)

    summary = {
        "solver_status": solver.StatusName(status),
        "items_and_stores": len(planning),
        "forecast_horizon_days": int(forecasts["date"].nunique()),
        "lead_time_days": LEAD_TIME_DAYS,
        "assumed_opening_inventory": OPENING_INVENTORY,
        "total_forecast_demand": float(
            planning["forecast_demand"].sum()
        ),
        "total_recommended_order": int(
            planning["recommended_order_qty"].sum()
        ),
        "total_expected_shortage": int(
            planning["expected_shortage"].sum()
        ),
        "total_estimated_cost": float(
            planning["estimated_cost"].sum()
        ),
        "purchase_cost_per_unit": PURCHASE_COST,
        "holding_cost_per_unit": HOLDING_COST,
        "stockout_cost_per_unit": STOCKOUT_COST,
    }

    with open(
        OUTPUT_DIR / "optimization_summary.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n======================================")
    print("INVENTORY OPTIMIZATION COMPLETED")
    print("======================================")
    print(f"Item-store combinations: {len(planning)}")
    print(
        f"Forecast demand: "
        f"{summary['total_forecast_demand']:,.0f}"
    )
    print(
        f"Recommended order: "
        f"{summary['total_recommended_order']:,}"
    )
    print(
        f"Expected shortage: "
        f"{summary['total_expected_shortage']:,}"
    )
    print(
        f"Estimated cost: "
        f"{summary['total_estimated_cost']:,.2f}"
    )
    print(f"Saved: {output_path}")
    print("======================================")

    return planning


if __name__ == "__main__":
    optimize_inventory()
