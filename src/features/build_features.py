
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "m5_sample.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "m5_features.csv"


def build_features() -> pd.DataFrame:
    """Build leakage-safe time-series features for M5 demand forecasting."""

    print("\nLoading preprocessed M5 data...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    df = df.sort_values(
        ["store_id", "item_id", "date"]
    ).reset_index(drop=True)

    group = df.groupby(["store_id", "item_id"])["sales"]

    print("Creating lag features...")
    for lag in [7, 28, 365]:
        df[f"lag_{lag}"] = group.shift(lag)

    print("Creating rolling demand features...")
    shifted_sales = df.groupby(
        ["store_id", "item_id"]
    )["sales"].shift(1)

    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            shifted_sales.groupby(
                [df["store_id"], df["item_id"]]
            )
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    print("Creating calendar features...")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    for col in ["event_type_1", "event_type_2"]:
        if col in df.columns:
            df[f"{col}_present"] = (
                df[col].notna()
            ).astype("int8")

    df = df.dropna(
        subset=["lag_7", "lag_28"]
    ).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\n======================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("======================================")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Features saved: {OUTPUT_PATH}")
    print("======================================")

    return df


if __name__ == "__main__":
    build_features()
