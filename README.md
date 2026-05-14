# Retail Decision Intelligence Platform

Retail Decision Intelligence Platform is a BI + AI academic project built on the Olist Brazilian e-commerce dataset. It combines data preparation, KPI reporting, forecasting, anomaly detection, customer segmentation, product segmentation, and recommendation generation inside one Streamlit dashboard.

## What The Project Does

The project answers practical retail questions such as:

- Which states and product categories generate the most revenue?
- How do delivery quality and review scores affect performance?
- What revenue can be expected over the next 30 days?
- Which dates show abnormal sales behavior?
- Which customers and products need management attention?

## Core Business Value

- BI layer: explains what happened in the business
- AI layer: estimates what may happen next and where management should act
- Decision layer: converts analytics into clear operational recommendations

## Important Data Modeling Note

The source file is a merged transactional dataset. In that merged table, some order items are repeated across payment rows. The pipeline now normalises the data to a canonical grain of **one row per order item** before KPI calculation.

Why this matters:

- It prevents double-counting of item value and revenue
- It keeps product and category analysis consistent
- It makes order-level and item-level summaries defensible during presentation

Payment totals are reconstructed at order level and then allocated back to item rows proportionally to item value so that:

- item-level analysis remains possible
- order revenue still matches the payment totals
- the dashboard uses one coherent revenue definition

## Project Architecture

```text
BI+AI Project/
|-- app.py
|-- main.py
|-- README.md
|-- PROJECT_PRESENTATION_GUIDE.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- reports/
`-- src/
    `-- bi_ai_retail/
        |-- config.py
        |-- data_pipeline.py
        |-- modeling.py
        |-- recommendations.py
        `-- reporting.py
```

## Pipeline Stages

### 1. Data Ingestion

- reads the merged Olist dataset
- creates required folders automatically
- downloads the dataset if it is missing

### 2. Grain Normalisation

- collapses duplicated joins caused by payment records
- keeps one canonical row per order item
- rebuilds valid item-level revenue values from order payments

### 3. Cleaning And Feature Engineering

- converts text, numeric, and datetime columns
- imputes missing values
- removes exact duplicates
- removes outliers with an IQR rule
- creates derived features such as delivery delay, freight per weight, and product density
- applies MinMax scaling, Standard scaling, and PCA

### 4. BI Summaries

- order summary
- daily sales summary
- customer RFM summary
- product performance summary
- state and category performance tables

### 5. AI Models

- Gradient Boosting daily revenue forecast
- Isolation Forest anomaly detection
- KMeans product clustering

### 6. Recommendation Engine

- translates outputs into management actions for operations, portfolio, retention, and customer experience

## Main Methods Used

- `Pandas` for transformation and aggregation
- `NumPy` for numerical logic
- `scikit-learn` for forecasting, clustering, anomaly detection, scaling, and PCA
- `Plotly` for interactive visuals
- `Streamlit` for the dashboard interface
- `joblib` for model export

## Current Outputs

The pipeline currently produces:

- cleaned retail table
- order summary
- customer summary
- daily sales table
- future forecast
- forecast validation table
- anomaly table
- product segments
- customer segment summary
- state and category performance tables
- scaling and PCA outputs
- KPI JSON files
- project summary report

## Current Result Snapshot

These values come from the latest regenerated outputs in `data/processed`:

- Total revenue: `$9.46M`
- Estimated profit: `$2.22M`
- Estimated gross margin: `23.49%`
- Orders analysed: `80,520`
- Customers analysed: `78,096`
- States analysed: `27`
- Repeat customer rate: `2.8%`
- Forecasted next 30 days revenue: `$363,053`
- Forecast accuracy proxy: `82.31%`
- Detected anomaly days: `19`
- Rows after cleaning: `92,047`

## Assumptions You Should Mention In Presentation

These are not weaknesses if you explain them clearly:

- Profit is **estimated**, not observed directly
- Estimated cost is derived from product value using a fixed ratio
- Forecasting uses historical daily patterns and engineered lag features
- Product segmentation labels are business-friendly names mapped from KMeans clusters
- Revenue is normalised to item grain after resolving duplicated payment joins

## Limitations

- The project uses one dataset and one time period only
- Forecasting is based on historical behavior and does not include promotions or seasonality calendars outside the observed data
- Profit is estimated because supplier cost is not available in the dataset
- Recommendations are rule-based, not generated by a reinforcement or causal model

## How To Run

Install dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

Run the pipeline:

```powershell
py -3 main.py
```

Launch the dashboard:

```powershell
py -3 -m streamlit run app.py
```

## Suggested Presentation Flow

1. Start with the business problem: retail teams need both monitoring and forward-looking support.
2. Explain the dataset and the grain-normalisation step.
3. Walk through preprocessing and engineered features.
4. Show the BI dashboard tabs.
5. Explain the forecast, anomalies, customer segmentation, and product segmentation.
6. End with the recommendation engine and managerial actions.

## Presentation Support File

For a function-by-function and code-block explanation, use:

- [PROJECT_PRESENTATION_GUIDE.md](C:\Users\a\Desktop\BI+AI Project\PROJECT_PRESENTATION_GUIDE.md)
