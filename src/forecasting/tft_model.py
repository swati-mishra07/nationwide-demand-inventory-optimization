import json
import os
import warnings

import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/processed/m5_sample.csv"
PREDICTION_PATH = "reports/predictions/tft_predictions.csv"
METRICS_PATH = "reports/metrics/tft_metrics.json"
MODEL_PATH = "models/tft_model"

MAX_ENCODER_LENGTH = 56
MAX_PREDICTION_LENGTH = 28

MAX_SERIES = 20
BATCH_SIZE = 64
MAX_EPOCHS = 3
RANDOM_SEED = 42


# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------

def calculate_metrics(predictions):

    actual = predictions["actual_sales"].to_numpy(dtype=float)
    predicted = predictions["predicted_sales"].to_numpy(dtype=float)

    errors = actual - predicted

    wape = (
        np.abs(errors).sum() / np.abs(actual).sum()
        if np.abs(actual).sum() > 0
        else 0.0
    )

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Mean per-series RMSSE using training history.
    rmsse_values = []

    for _, group in predictions.groupby(["item_id", "store_id"]):

        history = np.asarray(
            group["training_history"].iloc[0],
            dtype=float,
        )

        scale = np.mean(np.diff(history) ** 2)

        if scale > 0:

            series_errors = (
                group["actual_sales"].to_numpy(dtype=float)
                - group["predicted_sales"].to_numpy(dtype=float)
            )

            series_rmse = np.sqrt(np.mean(series_errors ** 2))

            rmsse_values.append(series_rmse / np.sqrt(scale))

    rmsse = float(np.mean(rmsse_values)) if rmsse_values else None

    return {
        "model": "Temporal Fusion Transformer",
        "series_count": int(
            predictions.groupby(["item_id", "store_id"]).ngroups
        ),
        "observations": int(len(predictions)),
        "forecast_horizon_days": MAX_PREDICTION_LENGTH,
        "wape": round(float(wape), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mean_series_rmsse": (
            round(rmsse, 4) if rmsse is not None else None
        ),
    }


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main():

    warnings.filterwarnings("ignore")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "=" * 55)
    print("TEMPORAL FUSION TRANSFORMER - M5 FORECASTING")
    print("=" * 55)

    # 1. Load data
    print("\nLoading processed M5 data...")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # Convert d_1, d_2, ... into numerical time indices.
    df["time_idx"] = (
        df["d"].str.extract(r"(\d+)")[0].astype(int) - 1
    )

    # Fill categorical missing values.
    categorical_columns = [
        "item_id",
        "store_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    for column in categorical_columns:
        df[column] = (
            df[column].fillna("Unknown").astype(str)
        )

    # Fill missing numerical feature values.
    numeric_columns = [
        "month",
        "wday",
        "year",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sell_price",
        "price_change_pct",
    ]

    df[numeric_columns] = df[numeric_columns].fillna(0)

    df = df.sort_values(
        ["item_id", "store_id", "time_idx"]
    ).reset_index(drop=True)

    print("Dataset shape:", df.shape)

    print(
        "Total item-store series:",
        df.groupby(["item_id", "store_id"]).ngroups,
    )

    # 2. Select a manageable number of series for CPU training.
    selected_series = (
        df[["item_id", "store_id"]]
        .drop_duplicates()
        .sort_values(["item_id", "store_id"])
        .head(MAX_SERIES)
    )

    df = df.merge(
        selected_series,
        on=["item_id", "store_id"],
        how="inner",
    )

    print("\nSelected series:", len(selected_series))
    print("Selected rows:", len(df))

    # 3. Create temporal training/holdout split.
    max_time_idx = int(df["time_idx"].max())

    training_cutoff = max_time_idx - MAX_PREDICTION_LENGTH

    train_df = df[
        df["time_idx"] <= training_cutoff
    ].copy()

    test_df = df[
        df["time_idx"] > training_cutoff
    ].copy()

    print("\nTraining cutoff:", training_cutoff)
    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))

    # Store historical training data for RMSSE scaling.
    training_histories = {}

    for key, group in train_df.groupby(["item_id", "store_id"]):
        training_histories[key] = (
            group["sales"].to_numpy(dtype=float)
        )

    # 4. Create TimeSeriesDataSet.
    print("\nBuilding TFT training dataset...")

    training_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="sales",
        group_ids=["item_id", "store_id"],

        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,

        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,

        static_categoricals=[
            "item_id",
            "store_id",
        ],

        time_varying_known_categoricals=[
            "weekday",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ],

        time_varying_known_reals=[
            "time_idx",
            "month",
            "wday",
            "year",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "sell_price",
            "price_change_pct",
        ],

        time_varying_unknown_reals=[
            "sales",
        ],

        target_normalizer=GroupNormalizer(
            groups=["item_id", "store_id"]
        ),

        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # 5. Create prediction dataset for the final 28 days.
    print("Building prediction dataset...")

    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df,
        predict=True,
        stop_randomization=True,
    )

    prediction_loader = prediction_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # 6. Configure TFT.
    print("\nInitializing Temporal Fusion Transformer...")

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.001,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        loss=QuantileLoss(),
        optimizer="adam",
        log_interval=-1,
    )

    print("Model parameters:", f"{tft.size() / 1e3:.1f}K")
    print("Training device: CPU")
    print("Maximum epochs:", MAX_EPOCHS)

    # 7. Train model.
    print("\nStarting TFT training...")

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        log_every_n_steps=20,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_loader,
    )

    # 8. Generate predictions.
    print("\nGenerating 28-day forecasts...")

    result = tft.predict(
        prediction_loader,
        mode="prediction",
        return_index=True,
    )

    # PyTorch Forecasting 1.8.0 returns a Prediction object.
    prediction_values = result.output
    prediction_index = result.index

    if prediction_index is None:
        raise RuntimeError(
            "TFT did not return prediction indices."
        )

    if isinstance(prediction_values, torch.Tensor):
        prediction_values = (
            prediction_values.detach().cpu().numpy()
        )

    prediction_values = np.asarray(prediction_values)

    # Validate the output dimensions.
    if prediction_values.ndim == 3:
        if prediction_values.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected TFT prediction shape: "
                f"{prediction_values.shape}"
            )

        prediction_values = prediction_values[..., 0]

    if prediction_values.shape != (
        len(prediction_index),
        MAX_PREDICTION_LENGTH,
    ):
        raise RuntimeError(
            "Unexpected prediction dimensions. "
            f"Received {prediction_values.shape}; "
            f"expected ({len(prediction_index)}, "
            f"{MAX_PREDICTION_LENGTH})."
        )

    # Build one row per item-store-day.
    output_rows = []

    prediction_index = prediction_index.reset_index(drop=True)

    for position, index_row in prediction_index.iterrows():

        item_id = index_row["item_id"]
        store_id = index_row["store_id"]

        first_forecast_idx = int(index_row["time_idx"])

        for offset, forecast in enumerate(
            prediction_values[position]
        ):

            time_idx = first_forecast_idx + offset

            output_rows.append({
                "item_id": item_id,
                "store_id": store_id,
                "time_idx": time_idx,
                "predicted_sales": max(
                    float(forecast),
                    0.0,
                ),
            })

    predictions = pd.DataFrame(output_rows)

    # 9. Attach actual test observations.
    actual_df = test_df[
        ["item_id", "store_id", "time_idx", "date", "sales"]
    ].rename(
        columns={"sales": "actual_sales"}
    )

    predictions = predictions.merge(
        actual_df,
        on=["item_id", "store_id", "time_idx"],
        how="inner",
    )

    # Check forecast coverage.
    expected_rows = (
        len(selected_series) * MAX_PREDICTION_LENGTH
    )

    if len(predictions) != expected_rows:
        raise RuntimeError(
            f"Forecast coverage mismatch: got {len(predictions)} "
            f"rows; expected {expected_rows}."
        )

    # Attach training histories for RMSSE calculation.
    predictions["training_history"] = predictions.apply(
        lambda row: training_histories[
            (row["item_id"], row["store_id"])
        ],
        axis=1,
    )

    if predictions.empty:
        raise RuntimeError(
            "No predictions matched actual test observations."
        )

    # 10. Evaluate.
    metrics = calculate_metrics(predictions)

    predictions = predictions.drop(
        columns=["training_history"]
    )

    print("\n" + "=" * 55)
    print("TFT EVALUATION RESULTS")
    print("=" * 55)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    # 11. Save outputs.
    os.makedirs("reports/predictions", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    predictions.to_csv(
        PREDICTION_PATH,
        index=False,
    )

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    trainer.save_checkpoint(MODEL_PATH + ".ckpt")

    print("\nSaved predictions:", PREDICTION_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Saved model:", MODEL_PATH + ".ckpt")

    print("\nTFT pipeline completed successfully.")


if __name__ == "__main__":
    main()
