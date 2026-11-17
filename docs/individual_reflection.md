# Individual Project Reflection
## Nationwide Demand Forecasting and Inventory Optimization

Working on this project helped me understand how demand forecasting can be connected to practical inventory decisions. My objective was not only to predict future sales but also to explore how those predictions could support better stock planning and reduce the risk of shortages.

I worked with the M5 Walmart dataset and built a forecasting pipeline using historical sales, calendar events, holidays, prices, and demand-related features. I implemented baseline forecasting with a naive approach and Prophet, and explored more advanced models using LightGBM and Temporal Fusion Transformer (TFT).

One important learning was that a more complex model does not automatically produce better forecasts. In my experiments, LightGBM achieved the best WAPE among the reported model results, while TFT required substantially more computational resources and was evaluated on a smaller sample of time series. This helped me appreciate the importance of fair model comparison, data scope, and evaluation methodology.

I also explored hierarchical forecast reconciliation using Darts. This helped me understand how item-store forecasts can be aggregated into higher-level totals while maintaining consistency. I then used Google OR-Tools to build an inventory-order optimization prototype that considers demand, safety stock, holding costs, and shortage costs.

Building the Plotly Dash dashboard helped me bring the different components together. It provides model comparisons, forecast visualizations, uncertainty bands, hierarchy information, and inventory recommendations in one interface.

The project also had limitations. The current prototype uses 100 items across two California stores rather than the complete nationwide dataset. The inventory cost and stock assumptions are illustrative, and the prediction intervals are approximate rather than statistically calibrated. The current evaluation also uses a limited holdout setup, so further work is needed before using the system for real operational decisions.

If I continue developing this project, I would expand the dataset coverage, improve multi-step forecasting evaluation, calibrate prediction intervals, incorporate actual inventory and supplier information, and test the optimization model under realistic warehouse constraints.

Overall, this project strengthened my understanding of the end-to-end data science workflow—from data preparation and feature engineering to model evaluation, hierarchical forecasting, optimization, and dashboard development. It also taught me to communicate model results honestly and clearly distinguish between a working prototype and a production-ready system.