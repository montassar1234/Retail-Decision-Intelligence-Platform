# Project Presentation Guide

This document is a speaking guide for presenting the project confidently. It explains the role of each file, each function, each major code block, the business logic behind the choices, and the questions a professor is most likely to ask.

## 1. One-Sentence Project Story

This project converts raw Olist e-commerce transactions into a decision-support platform that combines business intelligence, machine learning, and rule-based recommendations inside a Streamlit dashboard.

## 2. End-To-End Flow

```text
Raw merged dataset
-> grain normalisation
-> cleaning and feature engineering
-> BI summary tables
-> forecasting / anomalies / segmentation
-> KPI JSON + CSV outputs
-> Streamlit dashboard
-> management recommendations
```

## 3. What You Must Explain Early

Before a professor asks, explain this:

- The raw merged file repeats some order items across payment records.
- Because of that, the pipeline first converts the dataset to one row per order item.
- Order payments are then aggregated correctly and allocated back to item rows.
- This prevents double-counting and makes all later KPIs more reliable.

That single explanation removes the biggest logical weakness from the old version of the project.

## 4. File-By-File Walkthrough

## `app.py`

Purpose:

- runs the Streamlit dashboard
- loads processed outputs
- applies filters
- renders charts, KPIs, tables, and recommendation cards

### Import and path block

What it does:

- imports Python libraries and Streamlit tools
- inserts `src/` into `sys.path` so the dashboard can import project modules cleanly
- imports `run_pipeline` so the dashboard can rebuild outputs automatically if files are missing

Why it matters:

- this makes the app self-contained
- the dashboard does not depend on manual preprocessing each time

### `format_money(value)`

What it does:

- converts raw numeric values into readable strings such as `$9.46M` or `$125.4K`

Why it matters:

- KPI cards become presentation-friendly

### `load_or_build()`

What it does:

- checks whether all required processed output files already exist
- if any are missing, it runs the full pipeline
- loads all required CSV and JSON outputs into memory

Why it matters:

- the dashboard can recover itself
- presentation becomes smoother because you are less likely to face missing-file errors

### `build_filtered_views(clean_retail, region_filter, category_filter, state_filter)`

What it does:

- filters the cleaned dataset using the sidebar choices
- rebuilds an order-level view from the filtered item-level table

Important detail:

- the cleaned base table is item-level
- the function groups back to order level when order metrics are needed
- `is_late` is aggregated with `max`, so if any item in an order is late, the order is counted as late

Why it matters:

- item-level rows are correct for product and category analysis
- order-level rows are correct for order KPIs

### Dashboard setup block

What it does:

- loads data
- fixes any legacy product-segment naming leftovers
- shows title and sidebar filters
- computes the filtered views used by the rest of the app

### KPI block

What it does:

- shows revenue, orders, customers, average review, late-order rate, and forecasted next 30-day revenue

Logic note:

- revenue now uses the filtered order-level aggregation
- late-order rate uses order-level status, not raw item-row averaging

### `Executive BI` tab

What it does:

- trend chart: revenue and orders over time
- top states bar chart
- category revenue and review chart
- state performance scatterplot comparing delivery speed, revenue, and lateness

How to explain it:

- this tab answers “what is happening in the business right now?”

### `Data Preparation` tab

What it does:

- shows rows before cleaning, rows after cleaning, duplicates removed, and noise removed
- visualises missing values before and after processing
- shows scaling summary and PCA projection
- lists engineered feature examples

How to explain it:

- this tab proves the data engineering work, not only the final dashboard

### `Forecasting` tab

What it does:

- compares actual vs predicted historical revenue
- shows future forecast
- presents forecast error metrics
- displays anomaly detection results

How to explain it:

- this is the predictive layer

### `Segmentation` tab

What it does:

- shows product clusters on a revenue-volatility map
- shows customer segment revenue
- includes a searchable product table

How to explain it:

- products are segmented with KMeans
- customers are segmented with RFM logic

### `Recommendations` tab

What it does:

- displays the top state and top category
- shows recommendation cards with priority, theme, explanation, and evidence

How to explain it:

- this is the management action layer
- it converts analysis into business decisions

## `main.py`

Purpose:

- simple command-line entry point for running the full pipeline without the dashboard

### Main block

What it does:

- calls `run_pipeline()`
- prints a success message
- prints processed-file location
- prints the 30-day forecast total

Why it matters:

- useful for batch generation and quick validation

## `src/bi_ai_retail/config.py`

Purpose:

- centralizes file paths and constant settings

### Constants block

What it does:

- defines root folders for data, models, and reports
- stores the dataset URL
- stores the training cutoff date and forecast horizon

Why it matters:

- avoids hardcoding the same values in multiple places

## `src/bi_ai_retail/data_pipeline.py`

Purpose:

- this is the core ETL and analytics orchestration file
- it contains the most important project logic

### `ensure_directories()`

What it does:

- creates all required folders if they do not already exist

Why it matters:

- prevents file-writing failures later in the pipeline

### `download_dataset(force=False)`

What it does:

- downloads the raw Olist dataset if it is missing

Why it matters:

- makes the project portable

### `classify_customer_segment(recency_score, frequency_score, monetary_score)`

What it does:

- maps RFM scores into business labels:
- `Champions`
- `Loyal`
- `Promising`
- `Needs attention`
- `At risk`

Why it matters:

- turns raw scores into presentation-friendly business groups

### `load_raw_olist()`

What it does:

- ensures the dataset exists
- reads the raw CSV

### `collapse_order_item_payment_duplicates(raw_df)`

This is one of the most important functions in the entire project.

What problem it solves:

- the merged dataset can repeat the same order item across many payment rows
- if left untreated, the project would overstate revenue, overstate item counts, and distort product/category analysis

What it does step by step:

1. Defines item keys that identify a unique order item.
2. Defines payment keys that identify a unique payment record.
3. Drops duplicate item rows so only one row per order item remains.
4. Aggregates unique payment rows to obtain the true order payment total.
5. Computes item-level value from `total_order_value` or from `price + freight_value`.
6. Allocates the correct order payment total back to item rows proportionally to item value.
7. Returns the corrected item-level table and the number of duplicated join rows removed.

How to defend it:

- it is a normalization step required by the grain of the merged file
- it preserves total order revenue while preventing item-level double-counting

### `impute_and_engineer(raw_df)`

This is the main preprocessing function.

What it does step by step:

1. Calls the grain-normalisation function above.
2. Converts datetime columns safely.
3. Measures missing values before treatment.
4. Removes exact duplicates.
5. Fills missing text fields with `"Unknown"`.
6. Converts numeric columns and imputes missing values with medians.
7. Filters impossible financial records such as non-positive prices.
8. Removes noise with a 3-IQR rule on key numeric columns.
9. Creates time features such as order date, month, quarter, and weekday.
10. Maps Brazilian states to regions.
11. Engineers business features such as:
    - `review_text_length`
    - `product_volume_cm3`
    - `product_density_g_cm3`
    - `freight_per_weight`
    - `payment_per_installment`
    - `approval_lag_hours`
    - `delivery_delay_days`
    - `price_to_freight_ratio`
    - `order_value_per_item`
12. Applies MinMax scaling and Standard scaling.
13. Applies PCA to produce a 2D projection for visualization.
14. Builds a missing-value report and preprocessing summary.

Why it matters:

- this function transforms a raw transactional table into a modeling-ready analytical table

### `build_order_summary(clean_df)`

What it does:

- aggregates item-level rows into one row per order
- computes items, unique products, sellers, revenue, product value, freight, review score, delivery quality, and estimated profit

Important assumption:

- `estimated_cost = product_value * 0.72`
- therefore profit is estimated, not observed directly from the dataset

How to explain it:

- the dataset does not contain supplier cost
- a fixed ratio is used to create an approximate managerial margin view

### `build_daily_sales(order_summary)`

What it does:

- aggregates orders by day
- computes revenue, profit, quantity, order count, active customers, average order value, and profit margin

Why it matters:

- creates the time series used by forecasting and anomaly detection

### `build_customer_summary(order_summary, max_order_date)`

What it does:

- aggregates to one row per customer
- computes first order date, last order date, order count, revenue, profit, average review, and recency
- creates RFM scores using quintiles
- assigns an interpretable customer segment label

Why it matters:

- this supports customer lifetime and retention analysis

### `build_product_segments(clean_df)`

What it does:

- computes monthly revenue by product
- measures product revenue volatility
- creates a product feature table
- sends that feature table to the clustering function

Why it matters:

- builds the basis for product portfolio segmentation

### `build_aggregates(clean_df, order_summary, customer_summary, product_segments)`

What it does:

- creates state performance
- creates category performance
- creates monthly performance
- creates customer segment summary
- creates product segment summary
- extracts top products

Why it matters:

- these tables feed both the dashboard and the recommendation engine

### `build_semantic_outputs(clean_df, order_summary, customer_summary, product_segments)`

What it does:

- creates fact and dimension style tables:
- `fact_sales`
- `fact_orders`
- `dim_customer`
- `dim_product`
- `dim_date`

Why it matters:

- useful if the project is later connected to Power BI, SQL, or a star-schema style BI model

### `build_kpis(...)`

What it does:

- computes final headline KPIs used in the dashboard and summary report
- examples:
  - total revenue
  - estimated profit
  - order count
  - customer count
  - repeat customer rate
  - latest growth
  - 30-day forecast total
  - forecast accuracy proxy
  - late-order rate
  - top state
  - top category

Important note:

- forecast accuracy proxy is not a formal academic metric like RMSE
- it is a presentation-friendly summary derived from absolute error

### `run_pipeline()`

This is the orchestration function.

What it does:

1. Loads raw data.
2. Cleans and engineers the canonical item-level table.
3. Builds order, daily, customer, and product summary tables.
4. Trains the forecasting model.
5. Detects anomalies.
6. Builds aggregate tables.
7. Builds KPIs.
8. Builds management recommendations.
9. Builds semantic BI outputs.
10. Writes all CSV, JSON, and markdown outputs.
11. Saves the trained forecasting model.
12. Returns everything in a dictionary for further use.

How to explain it:

- this is the backbone of the project
- everything else depends on this function

## `src/bi_ai_retail/modeling.py`

Purpose:

- contains machine-learning and analytical modeling logic

### `ForecastArtifacts`

What it does:

- stores the forecast model, validation results, future forecast, and error metrics in one object

Why it matters:

- keeps the forecasting outputs organized

### `create_time_features(df)`

What it does:

- creates day-of-week, month, quarter, lag features, and rolling statistics from daily revenue

Why it matters:

- transforms a simple time series into a supervised learning feature set

### `train_forecasting_model(daily_sales, train_cutoff_date, forecast_horizon_days)`

What it does:

1. builds time features
2. splits the dataset into train and test by date
3. trains a `GradientBoostingRegressor`
4. predicts the historical test window
5. calculates MAE, RMSE, and MAPE
6. rolls forward recursively to forecast the next 30 days

Why Gradient Boosting was reasonable here:

- handles nonlinear patterns better than a plain linear model
- works well on tabular engineered features
- easy to explain in an academic context

Important limitation:

- recursive forecasting can compound error over future days

### `detect_anomalies(daily_sales)`

What it does:

- adds rolling statistics and percentage change
- fits `IsolationForest`
- also computes a z-score
- flags a day as anomalous if the isolation model or the z-score rule says it is unusual
- labels anomalies as `Spike` or `Drop`

Why it matters:

- combines machine learning and statistical intuition

### `segment_products(product_frame, n_clusters=3)`

What it does:

- standardizes product features
- runs `KMeans`
- profiles clusters by revenue and volatility
- assigns business-friendly labels:
  - `High performers`
  - `Seasonal opportunities`
  - `At-risk products`

Why it matters:

- raw cluster IDs are hard to present
- labels make the result meaningful to non-technical audiences

## `src/bi_ai_retail/recommendations.py`

Purpose:

- converts analytics into actions

### `build_recommendations(...)`

What it does:

- compares forecast to recent revenue
- checks repeated anomaly drops
- finds the weakest stable review category
- finds at-risk products
- checks the number of at-risk customers
- identifies the top state by revenue

Why “stable review category” matters:

- the code now ignores very tiny categories when choosing the lowest-review category if better-supported categories exist
- this avoids making weak recommendations based on an almost empty category

How to explain it:

- this is a rule-based decision layer
- it is meant to be interpretable and actionable rather than black-box

## `src/bi_ai_retail/reporting.py`

Purpose:

- writes final report outputs

### `save_json(path, payload)`

What it does:

- writes dictionaries to JSON files cleanly

### `build_markdown_report(path, summary, model_metrics, recommendations)`

What it does:

- writes a human-readable project summary
- includes KPI headline values, forecasting metrics, and recommendations

Why it matters:

- useful for submission and quick review outside the dashboard

## 5. The Most Important Logic Choices

These are the design decisions you should be ready to justify.

### Choice 1: Normalising the merged dataset

Defense:

- the raw merged file is not already at a clean analytical grain
- payment joins can multiply item rows
- normalising to one row per order item is the correct modeling choice for downstream analytics

### Choice 2: Using item-level allocation of payment totals

Defense:

- it preserves order-level payment totals
- it allows product/category analysis without revenue inflation
- proportional allocation is a reasonable method when direct per-item payment attribution is unavailable

### Choice 3: Using estimated profit

Defense:

- supplier cost is not present in the dataset
- estimated profit is used only as a proxy decision metric
- the README and report clearly label it as estimated

### Choice 4: Using Gradient Boosting instead of deep learning

Defense:

- for this dataset size and tabular feature style, gradient boosting is easier to justify and interpret
- the goal is reliable academic modeling, not unnecessary complexity

### Choice 5: Rule-based recommendations

Defense:

- recommendations need to be explainable to decision-makers
- rule-based logic is transparent and easy to audit

## 6. Questions Your Professor May Ask

### “What is the grain of your cleaned dataset?”

Answer:

The cleaned analytical table is one row per order item. I explicitly normalised the raw merged file because payment joins repeated some items across payment records.

### “Why did you normalise payments back to items?”

Answer:

The merged dataset stores order payments separately from items, which can create row multiplication after merging. I first recovered the true order payment total, then allocated it back to items proportionally to item value so item-level and order-level revenue stay consistent.

### “Is profit real or estimated?”

Answer:

It is estimated. The dataset does not include supplier cost, so I used a fixed cost ratio to create a margin proxy. I clearly label the result as estimated profit and estimated gross margin.

### “Why did you remove outliers?”

Answer:

I removed extreme numeric noise using an IQR-based rule to reduce distortion in the aggregates, clustering, and forecast features. I also reported how many rows were removed so the preprocessing remains transparent.

### “Why these algorithms?”

Answer:

- Forecasting: Gradient Boosting because the data is tabular and nonlinear.
- Anomaly detection: Isolation Forest because it works well for unusual-pattern detection without labels.
- Product segmentation: KMeans because it is easy to explain and interpret after standardization.

### “How reliable is the forecast?”

Answer:

I report MAE, RMSE, and MAPE, and I also show a forecast-accuracy proxy in the dashboard for an executive summary view. The model is useful for directional planning, but it is not a perfect future guarantee.

### “Why are recommendations rule-based instead of AI-generated text?”

Answer:

Because interpretability matters. Each recommendation is tied directly to measurable evidence from the data, which makes it more defensible in an academic project.

### “What are the main limitations?”

Answer:

- the project depends on one dataset and one historical period
- some business metrics such as profit must be estimated
- the recommendation engine is deterministic, not causal
- future promotions and external events are not modeled explicitly

## 7. Suggested Oral Walkthrough

Use this order:

1. Business problem
2. Dataset and grain issue
3. Preprocessing and feature engineering
4. BI layer
5. Forecasting and anomalies
6. Segmentation
7. Recommendations
8. Limitations and next steps

## 8. Short Closing Script

You can close with this:

“This project shows how raw e-commerce data can be transformed into a practical decision-support system. The BI layer explains past performance, the AI layer supports forward-looking analysis, and the recommendation layer translates both into business actions. I also corrected the merged-data grain issue so the KPIs and segment results remain analytically defensible.”
