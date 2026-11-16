
from pathlib import Path
import csv

import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "m5"

EXPECTED_FILES = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]


# --------------------------------------------------
# 2. Count records without loading the whole CSV
# --------------------------------------------------

def count_csv_rows(file_path: Path) -> int:
    """Count CSV data rows, excluding the header."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        return max(sum(1 for _ in csv.reader(file)) - 1, 0)


# --------------------------------------------------
# 3. Inspect one CSV
# --------------------------------------------------

def inspect_csv(file_path: Path) -> None:
    print("\n" + "=" * 75)
    print(f"FILE: {file_path.name}")
    print("=" * 75)

    # Read only the header and first five records
    sample = pd.read_csv(file_path, nrows=5)

    # Count actual records using streaming
    row_count = count_csv_rows(file_path)

    print(f"Total records : {row_count:,}")
    print(f"Total columns : {len(sample.columns):,}")
    print(f"File size     : {file_path.stat().st_size / (1024 ** 2):.2f} MB")

    print("\nColumn names:")
    print(sample.columns.tolist())

    print("\nData types in sample:")
    print(sample.dtypes.to_string())

    print("\nFirst five records:")
    print(sample.head().to_string(index=False))

    print("\nMissing values in first five records only:")
    print(sample.isnull().sum().to_string())

    # Special inspection for the M5 sales files
    if file_path.name.startswith("sales_train"):
        day_columns = [
            column for column in sample.columns
            if column.startswith("d_")
        ]

        metadata_columns = [
            column for column in sample.columns
            if not column.startswith("d_")
        ]

        print(f"\nDaily sales columns: {len(day_columns):,}")
        print(f"Metadata columns   : {metadata_columns}")

    # Special inspection for calendar
    if file_path.name == "calendar.csv":
        print("\nCalendar date range:")
        print(f"First date in sample: {sample['date'].iloc[0]}")
        print(f"Calendar columns: {sample.columns.tolist()}")

    # Special inspection for selling prices
    if file_path.name == "sell_prices.csv":
        print("\nPrice data columns:")
        print(sample.columns.tolist())


# --------------------------------------------------
# 4. Main execution
# --------------------------------------------------

def main() -> None:
    print("\nM5 DATASET INSPECTION")
    print(f"Dataset directory: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"M5 dataset directory not found: {DATA_DIR}"
        )

    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (DATA_DIR / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing dataset files: {missing_files}"
        )

    print("\nAll five expected dataset files are present.")

    for filename in EXPECTED_FILES:
        inspect_csv(DATA_DIR / filename)

    print("\n" + "=" * 75)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
