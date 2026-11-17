# Data and Dashboard Documentation

## 1. Project Data Source

This project uses the **M5 Forecasting dataset**, published for the Walmart demand forecasting competition on Kaggle.

Dataset: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

The project uses three primary input files:

| File | Purpose |
|---|---|
| `sales_train_evaluation.csv` | Historical daily unit sales |
| `calendar.csv` | Dates, weekdays, events, and SNAP indicators |
| `sell_prices.csv` | Item prices by store and week |

The raw files are stored locally under `data/raw/m5/` and are not committed to GitHub because of their size.

## 2. Data Hierarchy

The M5 dataset supports a retail hierarchy:

- Item: individual product
- Store: retail store
- State: geographic state
- National: aggregation across selected stores and states

The current prototype selects 100 items from two California stores: CA_1 and CA_2.

| Level | Current sample |
|---|---:|
| Items | 100 |
| Stores | 2 |
| States | 1 |
| Item-store series | 200 |
| Historical dates | 1,941 |

This is a limited sample used to demonstrate the forecasting and reconciliation pipeline. It does not represent all Walmart stores nationwide.

## 3. Data Processing Pipeline

### Loading

`src/data/load_data.py` loads the M5 sales, calendar, and selling-price files.

### Preprocessing

`src/data/preprocess.py`:

- Selects the sample of items and stores.
- Converts daily sales columns into a long time-series structure.
- Joins calendar and price information.
- Handles missing selling-price values.
- Produces the processed dataset.

Output: `data/processed/m5_sample.csv`

### Feature Engineering

`src/features/build_features.py` generates:

- Historical demand lag features
- Rolling demand averages
- Calendar features
- Weekday and month information
- Event and holiday indicators
- SNAP indicators
- Price-related features

Output: `data/processed/m5_features.csv`

**Promotion data limitation:** The M5 files used in this project do not provide a dedicated promotion campaign indicator. Price changes are used as price-related signals, not as confirmed promotion labels. Future work could incorporate external promotion data.

## 4. Forecasting Inputs and Outputs

The forecasting modules use the processed time-series features.

| Module | Purpose | Main output |
|---|---|---|
| `naive.py` | Baseline forecast | `naive_predictions.csv` |
| `prophet_model.py` | Statistical forecasting | `prophet_predictions.csv` |
| `lightgbm_model.py` | Gradient boosting forecast | `lightgbm_predictions.csv` |
| `tft_model.py` | Deep-learning time-series forecast | `tft_predictions.csv` |
| `uncertainty.py` | Approximate uncertainty bands | `lightgbm_predictions_with_intervals.csv` |

Model metrics are stored under `reports/metrics/`.

## 5. Hierarchical Reconciliation

`src/forecasting/darts_reconcile.py` uses Darts' `BottomUpReconciliator` to produce coherent forecasts across the hierarchy.

Output:

`reports/reconciliation/darts_reconciled_forecasts.csv`

The reconciliation summary records the number of item-store series, stores, states, forecast dates, and maximum aggregation difference.

## 6. Inventory Optimization Interface

`src/inventory/optimize_orders.py` uses Google OR-Tools CP-SAT to calculate order recommendations.

Inputs include:

- Forecast demand
- Historical demand variability
- Assumed lead time
- Assumed holding and shortage costs
- Assumed ordering capacity

Outputs:

- `reports/inventory/inventory_recommendations.csv`
- `reports/inventory/optimization_summary.json`

The current implementation uses illustrative business assumptions because actual stock-on-hand, supplier, and cost information was not available.

## 7. Dashboard Interface

Run the dashboard from the repository root:

```powershell
python -m app.dashboard
```

Open:

`http://127.0.0.1:8050/`

The dashboard provides:

- Forecast model performance comparison
- Model selection
- Store and item selection
- Actual-versus-predicted demand charts
- Approximate LightGBM uncertainty bands
- Inventory order recommendations
- Darts hierarchy and coherence information

The dashboard is a local Plotly Dash application. This project does not currently expose a separate REST API.

## 8. Important Evaluation Notes

- Forecast evaluation uses a 28-day holdout.
- TFT uses 20 item-store series; the other model runs use 200.
- RMSSE calculations are implementation-specific and are not official M5 competition scores.
- LightGBM uses observed historical lag values during the holdout for one-step-ahead predictions.
- Prediction intervals are approximate and have not been calibrated for statistical coverage.

These limitations should be considered when interpreting model results or using the inventory recommendations.