
from pathlib import Path

import pandas as pd

from src.data.load_data import load_m5_data


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_m5_data(
    n_items: int = 100,
    n_stores: int = 2,
) -> pd.DataFrame:
    """Convert a manageable M5 sample into a daily time-series dataset."""

    print("\nLoading source data...")
    data = load_m5_data()

    sales = data["sales"]
    calendar = data["calendar"]
    prices = data["prices"]

    print("\nSelecting item and store sample...")
    item_ids = (
        sales["item_id"]
        .drop_duplicates()
        .sample(
            n=min(n_items, sales["item_id"].nunique()),
            random_state=42,
        )
    )
    store_ids = sales["store_id"].drop_duplicates().head(n_stores)

    sample = sales[
        sales["item_id"].isin(item_ids)
        & sales["store_id"].isin(store_ids)
    ].copy()

    daily_columns = [
        col for col in sample.columns if col.startswith("d_")
    ]

    print(
        f"Selected {sample['item_id'].nunique()} items, "
        f"{sample['store_id'].nunique()} stores, "
        f"{len(daily_columns)} days."
    )

    print("\nConverting daily columns into rows...")
    long_sales = sample.melt(
        id_vars=[
            "id",
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ],
        value_vars=daily_columns,
        var_name="d",
        value_name="sales",
    )

    long_sales["sales"] = (
        pd.to_numeric(long_sales["sales"], errors="coerce")
        .fillna(0)
        .astype("int16")
    )

    print("\nJoining calendar information...")
    calendar_columns = [
        col
        for col in [
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
        if col in calendar.columns
    ]

    result = long_sales.merge(
        calendar[calendar_columns],
        on="d",
        how="left",
        validate="many_to_one",
    )

    print("\nJoining selling prices...")
    result = result.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    # Capture missing prices before filling them.
    missing_prices = result["sell_price"].isna().sum()

    result["sell_price"] = result["sell_price"].fillna(0)
    result["revenue"] = result["sales"] * result["sell_price"]

    result = result.sort_values(
        ["store_id", "item_id", "date"]
    ).reset_index(drop=True)

    result["price_change_pct"] = (
        result.groupby(["store_id", "item_id"])["sell_price"]
        .pct_change()
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    output_path = OUTPUT_DIR / "m5_sample.csv"
    result.to_csv(output_path, index=False)

    print("\n======================================")
    print("PREPROCESSING COMPLETED")
    print("======================================")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns)}")
    print(f"Items: {result['item_id'].nunique()}")
    print(f"Stores: {result['store_id'].nunique()}")
    print(
        f"Date range: {result['date'].min()} "
        f"to {result['date'].max()}"
    )
    print(f"Missing prices before filling: {missing_prices:,}")
    print(f"Saved: {output_path}")
    print("======================================")

    return result


if __name__ == "__main__":
    preprocess_m5_data()
