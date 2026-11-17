# Test Results and Model Performance

## 1. Test Environment

- Operating system: Windows 11
- Python: 3.11.9
- Environment: Python virtual environment
- Dashboard: Plotly Dash
- Forecasting and optimization: Prophet, LightGBM, PyTorch Forecasting, Darts, OR-Tools

## 2. Implementation Verification

| Component | Verification | Result |
|---|---|---|
| Python source files | `python -m compileall -q src app` | Passed |
| Data loading | `python -m src.data.load_data` | Executed successfully |
| Preprocessing | `python -m src.data.preprocess` | Executed successfully |
| Feature engineering | `python -m src.features.build_features` | Executed successfully |
| Naive forecasting | `python -m src.forecasting.naive` | Executed successfully |
| Prophet forecasting | `python -m src.forecasting.prophet_model` | Executed successfully |
| LightGBM forecasting | `python -m src.forecasting.lightgbm_model` | Executed successfully |
| TFT forecasting | `python -m src.forecasting.tft_model` | Executed successfully |
| Darts reconciliation | `python -m src.forecasting.darts_reconcile` | Passed coherence check |
| Inventory optimization | `python -m src.inventory.optimize_orders` | Solver returned OPTIMAL |
| Dashboard | `python -m app.dashboard` | Launched successfully; HTTP 200 verified |

## 3. Forecasting Performance

All model runs use a 28-day holdout, with the dataset scope and model-specific differences described below.

| Model | WAPE | MAE | RMSE |
|---|---:|---:|---:|
| Naive | 81.44% | 1.1623 | 1.9907 |
| Prophet | 82.65% | 1.1795 | 1.9808 |
| LightGBM | 75.32% | 1.0750 | 1.7893 |
| TFT | 92.18% | 0.9350 | 1.5281 |

LightGBM achieved the lowest WAPE among the reported runs.

TFT was evaluated on 20 item-store series, while the other models used 200. Therefore, the metrics are not a directly comparable ranking. RMSSE values use implementation-specific calculations and are not official M5 competition scores.

## 4. Hierarchical Reconciliation

Darts reconciliation results:

| Check | Result |
|---|---:|
| Item-store series | 200 |
| Stores | 2 |
| States | 1 |
| Forecast dates | 28 |
| Maximum aggregation difference | 0.0000000000 |
| Coherence check | Passed |

Output: `reports/reconciliation/darts_reconciled_forecasts.csv`

## 5. Inventory Optimization

The OR-Tools CP-SAT solver returned `OPTIMAL` under the configured prototype assumptions.

| Output | Result |
|---|---:|
| Item-store combinations | 200 |
| Forecast demand | 7,745 |
| Recommended order quantity | 9,184 |
| Expected shortage | 0 |
| Solver status | OPTIMAL |

These results depend on illustrative costs, lead time, capacity, and zero opening inventory. They are not validated operational purchasing recommendations.

## 6. Dashboard Verification

The Plotly Dash application launched at `http://127.0.0.1:8050/`.

Verified functionality included:

- Model metric comparison
- Item and store selection
- Forecast visualization
- Inventory recommendation table
- Darts hierarchy and coherence information
- Scope and assumption notices

The local dashboard returned HTTP 200 during verification.

## 7. Limitations

- The current sample contains 100 items across two California stores.
- No explicit promotion campaign variable is available in the selected M5 input files.
- Prediction intervals are approximate and not calibrated.
- LightGBM holdout predictions use observed historical lag values for one-step-ahead evaluation.
- TFT and other model results use different sample sizes.

## 8. Reproduction

Run the pipeline commands from the repository root after installing dependencies and downloading the M5 data. See `README.md` and `docs/data_api.md` for setup, inputs, outputs, and execution instructions.