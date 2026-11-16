
"""
M5 Dataset Loader
-----------------
Loads sales, calendar, and selling price datasets
with memory-conscious data types.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "m5"


def load_m5_data(sales_file="sales_train_evaluation.csv"):
    """
    Load the M5 Walmart forecasting datasets.

    Parameters
    ----------
    sales_file : str
        Sales file to load. Choose either:
        sales_train_evaluation.csv
        sales_train_validation.csv

    Returns
    -------
    dict
        Dictionary containing sales, calendar, and prices.
    """

    sales_path = DATA_DIR / sales_file
    calendar_path = DATA_DIR / "calendar.csv"
    prices_path = DATA_DIR / "sell_prices.csv"

    # Step 1: Verify required files
    required_files = [
        sales_path,
        calendar_path,
        prices_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    print("\nLoading M5 Walmart dataset...")

    # Step 2: Inspect sales columns before loading
    columns = pd.read_csv(sales_path, nrows=0).columns.tolist()

    metadata_columns = [
        "id", "item_id", "dept_id",
        "cat_id", "store_id", "state_id"
    ]

    missing_columns = set(metadata_columns) - set(columns)

    if missing_columns:
        raise ValueError(
            f"Missing sales metadata columns: {missing_columns}"
        )

    daily_columns = [
        column for column in columns
        if column.startswith("d_")
    ]

    # Step 3: Set memory-efficient data types
    sales_dtypes = {
        column: "category"
        for column in metadata_columns
    }

    sales_dtypes.update({
        column: "int16"
        for column in daily_columns
    })

    # Step 4: Load sales
    print("Loading sales history...")

    sales = pd.read_csv(
        sales_path,
        dtype=sales_dtypes
    )

    # Step 5: Load calendar
    print("Loading calendar...")

    calendar = pd.read_csv(
        calendar_path,
        parse_dates=["date"]
    )

    # Step 6: Load selling prices
    print("Loading selling prices...")

    prices = pd.read_csv(
        prices_path,
        dtype={
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int32",
            "sell_price": "float32"
        }
    )

    # Step 7: Return datasets
    datasets = {
        "sales": sales,
        "calendar": calendar,
        "prices": prices
    }

    print("\n" + "=" * 45)
    print("M5 DATASET LOADED SUCCESSFULLY")
    print("=" * 45)

    for name, dataframe in datasets.items():
        memory_mb = (
            dataframe.memory_usage(deep=True).sum()
            / (1024 ** 2)
        )

        print(
            f"{name.upper():<12} "
            f"Rows: {len(dataframe):>10,} | "
            f"Columns: {len(dataframe.columns):>5,} | "
            f"Memory: {memory_mb:>8.2f} MB"
        )

    print("=" * 45)

    return datasets


if __name__ == "__main__":

    data = load_m5_data()

    sales = data["sales"]

    print("\nHierarchy information:")
    print("States:", sales["state_id"].nunique())
    print("Stores:", sales["store_id"].nunique())
    print("Items:", sales["item_id"].nunique())

    print("\nData loading completed.")
