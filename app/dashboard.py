
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "predictions"
METRICS = ROOT / "reports" / "metrics"
REPORTS = ROOT / "reports"

FILES = {
    "Naive": PREDICTIONS / "naive_predictions.csv",
    "Prophet": PREDICTIONS / "prophet_predictions.csv",
    "LightGBM": PREDICTIONS / "lightgbm_predictions_with_intervals.csv",
    "TFT": PREDICTIONS / "tft_predictions.csv",
}

METRIC_FILES = {
    "Naive": METRICS / "model_comparison.json",
    "Prophet": METRICS / "prophet_metrics.json",
    "LightGBM": METRICS / "lightgbm_metrics.json",
    "TFT": METRICS / "tft_metrics.json",
}

INVENTORY_PATH = REPORTS / "inventory" / "inventory_recommendations.csv"
DARTS_PATH = REPORTS / "reconciliation" / "darts_reconciled_forecasts.csv"
DARTS_SUMMARY_PATH = (
    REPORTS / "reconciliation" / "darts_reconciliation_summary.json"
)


def read_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_model_metrics():
    comparison = read_json(METRIC_FILES["Naive"])
    prophet = read_json(METRIC_FILES["Prophet"])
    lightgbm = read_json(METRIC_FILES["LightGBM"])
    tft = read_json(METRIC_FILES["TFT"])

    return {
        "Naive": comparison.get("naive_7_day_average", {}),
        "Prophet": prophet.get("metrics", {}),
        "LightGBM": lightgbm,
        "TFT": tft,
    }


model_data = {
    name: read_csv(path)
    for name, path in FILES.items()
}

for name, df in model_data.items():
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

inventory = read_csv(INVENTORY_PATH)
darts_forecasts = read_csv(DARTS_PATH)
darts_summary = read_json(DARTS_SUMMARY_PATH)
metrics = get_model_metrics()

# Use LightGBM as the default forecast source.
base_forecast = model_data["LightGBM"]

if base_forecast.empty:
    raise FileNotFoundError("LightGBM prediction file is missing.")

stores = sorted(base_forecast["store_id"].dropna().astype(str).unique())
items = sorted(base_forecast["item_id"].dropna().astype(str).unique())

app = Dash(__name__)
app.title = "Demand Forecasting and Inventory Intelligence"


def card(title, value, subtitle=""):
    return html.Div(
        [
            html.Div(title, style={"color": "#64748b", "fontSize": "13px"}),
            html.H2(value, style={"margin": "8px 0", "color": "#0f172a"}),
            html.Div(subtitle, style={"fontSize": "12px", "color": "#64748b"}),
        ],
        style={
            "background": "white",
            "padding": "18px",
            "borderRadius": "10px",
            "flex": "1",
            "minWidth": "150px",
            "boxShadow": "0 1px 5px #00000012",
        },
    )


def metric_table():
    rows = []
    for name, values in metrics.items():
        rows.append({
            "Model": name,
            "WAPE (%)": (
                round(values["wape"] * 100, 2)
                if values.get("wape") is not None else "N/A"
            ),
            "MAE": round(values["mae"], 4)
            if values.get("mae") is not None else "N/A",
            "RMSE": round(values["rmse"], 4)
            if values.get("rmse") is not None else "N/A",
            "RMSSE": round(
                values.get("rmsse", values.get("mean_per_series_rmsse",
                values.get("mean_series_rmsse"))), 4
            ) if values.get("rmsse", values.get(
                "mean_per_series_rmsse", values.get("mean_series_rmsse")
            )) is not None else "N/A",
        })
    return rows


def section_title(text):
    return html.H2(
        text,
        style={"color": "#172554", "marginTop": "28px"}
    )


app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "Demand Forecasting & Inventory Intelligence",
                    style={"marginBottom": "6px"},
                ),
                html.P(
                    "M5 retail demand forecasting, hierarchical "
                    "reconciliation and inventory optimization",
                    style={"color": "#cbd5e1", "margin": "0"},
                ),
            ],
            style={
                "background": "#172554",
                "color": "white",
                "padding": "26px",
                "borderRadius": "12px",
            },
        ),

        section_title("Forecasting model comparison"),
        dash_table.DataTable(
            data=metric_table(),
            columns=[{"name": c, "id": c} for c in [
                "Model", "WAPE (%)", "MAE", "RMSE", "RMSSE"
            ]],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "12px", "textAlign": "center"},
            style_header={
                "backgroundColor": "#dbeafe",
                "fontWeight": "bold",
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": '{Model} = "LightGBM"'},
                    "backgroundColor": "#eff6ff",
                    "fontWeight": "bold",
                }
            ],
        ),
        html.P(
            "Evaluation caveat: TFT uses 20 item-store series; the other "
            "models use 200. Compare their metrics cautiously. "
            "RMSSE definitions also differ across model implementations.",
            style={"fontSize": "12px", "color": "#64748b"},
        ),

        section_title("Interactive forecast explorer"),
        html.Div(
            [
                html.Div([
                    html.Label("Forecasting model"),
                    dcc.Dropdown(
                        id="model-filter",
                        options=[
                            {"label": name, "value": name}
                            for name in FILES
                            if not model_data[name].empty
                        ],
                        value="LightGBM",
                        clearable=False,
                    ),
                ], style={"flex": "1"}),

                html.Div([
                    html.Label("Store"),
                    dcc.Dropdown(
                        id="store-filter",
                        options=[{"label": x, "value": x} for x in stores],
                        value=stores[0],
                        clearable=False,
                    ),
                ], style={"flex": "1"}),

                html.Div([
                    html.Label("Item"),
                    dcc.Dropdown(
                        id="item-filter",
                        options=[{"label": x, "value": x} for x in items],
                        value=items[0],
                        clearable=False,
                    ),
                ], style={"flex": "1"}),
            ],
            style={
                "display": "flex",
                "gap": "15px",
                "flexWrap": "wrap",
                "marginBottom": "15px",
            },
        ),

        dcc.Graph(id="forecast-chart"),

        section_title("Inventory optimization — OR-Tools"),
        html.P(
            "Recommended order quantities and estimated costs "
            "from the current inventory optimization scenario."
        ),
        dash_table.DataTable(
            data=inventory.head(20).to_dict("records")
            if not inventory.empty else [],
            columns=[
                {"name": c, "id": c} for c in inventory.columns
            ],
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
            style_cell={
                "padding": "8px",
                "fontSize": "12px",
                "textAlign": "left",
                "maxWidth": "200px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
            },
            style_header={
                "backgroundColor": "#dbeafe",
                "fontWeight": "bold",
            },
        ),

        section_title("Darts hierarchical reconciliation"),
        html.Div(
            [
                card(
                    "Item-store series",
                    str(darts_summary.get("item_store_series", "N/A")),
                ),
                card(
                    "Stores",
                    str(darts_summary.get("stores", "N/A")),
                ),
                card(
                    "States",
                    str(darts_summary.get("states", "N/A")),
                ),
                card(
                    "Coherence",
                    "Passed" if darts_summary.get(
                        "coherence_check_passed"
                    ) else "Not verified",
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),

        html.Div(
            [
                card(
                    "Forecast dates",
                    str(darts_summary.get("forecast_dates", "N/A")),
                ),
                card(
                    "Maximum coherence difference",
                    f"{darts_summary.get('maximum_coherence_difference', 0):.10f}"
                    if darts_summary else "N/A",
                ),
                card(
                    "National forecast total",
                    f"{darts_forecasts['national_total'].sum():,.2f}"
                    if "national_total" in darts_forecasts.columns
                    else "N/A",
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "flexWrap": "wrap",
                "marginTop": "12px",
            },
        ),

        html.Div(
            "Scope and assumptions: This prototype uses 100 items across "
            "2 California stores and 1 state. Inventory costs, lead times "
            "and opening stock are illustrative assumptions. Uncertainty "
            "bands are approximate, not calibrated prediction intervals. "
            "M5 does not provide an explicit promotion campaign flag.",
            style={
                "background": "#fff7ed",
                "padding": "15px",
                "borderRadius": "8px",
                "marginTop": "28px",
                "fontSize": "12px",
                "color": "#7c2d12",
            },
        ),
    ],
    style={
        "background": "#f1f5f9",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "minHeight": "100vh",
    },
)


@app.callback(
    Output("forecast-chart", "figure"),
    Input("model-filter", "value"),
    Input("store-filter", "value"),
    Input("item-filter", "value"),
)
def update_forecast_chart(selected_model, selected_store, selected_item):
    df = model_data.get(selected_model, pd.DataFrame())
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="No prediction file available")
        return fig

    filtered = df[
        (df["store_id"].astype(str) == str(selected_store))
        & (df["item_id"].astype(str) == str(selected_item))
    ].sort_values("date")

    if filtered.empty:
        fig.update_layout(
            title=f"{selected_model} has no predictions for this item/store. "
                  "TFT currently covers only 20 series."
        )
        return fig

    if selected_model == "LightGBM" and {
        "lower_bound", "upper_bound"
    }.issubset(filtered.columns):
        fig.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered["upper_bound"],
            line={"width": 0},
            showlegend=False,
            name="Upper bound",
        ))
        fig.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered["lower_bound"],
            fill="tonexty",
            fillcolor="rgba(59,130,246,0.15)",
            line={"width": 0},
            name="Approximate uncertainty band",
        ))

    fig.add_trace(go.Scatter(
        x=filtered["date"],
        y=filtered["predicted_sales"],
        mode="lines",
        name=f"{selected_model} forecast",
        line={"width": 2},
    ))

    actual_column = (
        "actual_sales" if "actual_sales" in filtered.columns
        else "sales" if "sales" in filtered.columns
        else None
    )

    if actual_column:
        fig.add_trace(go.Scatter(
            x=filtered["date"],
            y=filtered[actual_column],
            mode="lines",
            name="Actual demand",
            line={"dash": "dash", "width": 2},
        ))

    fig.update_layout(
        title=f"{selected_model}: {selected_item} / {selected_store}",
        xaxis_title="Date",
        yaxis_title="Daily sales",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )

    return fig


if __name__ == "__main__":
    app.run(debug=False)
