# Retail Decision Intelligence Platform

This repository contains a reworked **BI + AI retail analytics solution** built to feel closer to a professional BI delivery than a classroom prototype. The project combines:

- a **BI semantic layer** with fact and dimension outputs
- an **executive dashboard** for performance monitoring and drill-down analysis
- an **AI layer** for forecasting, anomaly detection, segmentation, and action recommendations

## Business goal

Retail teams need more than historical reporting. They need a decision-support platform that can answer:

- Which markets, categories, and products drive value?
- Where is margin quality weakening?
- Which customers are loyal, promising, or at risk?
- What demand should we expect next month?
- Which abnormal sales patterns need investigation?

## Senior-level project upgrades

This reworked version adds a stronger BI backbone:

- order-level and line-level fact outputs
- date, customer, and product dimensions
- customer RFM segmentation
- product ABC classification
- richer KPI set including margin, repeat rate, forecast delta, and forecast accuracy proxy
- executive dashboard filters for region, category, and country

## Dataset

- Source: [Online Retail dataset](https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/online_retail.csv)
- Period covered: `2010-12-01` to `2011-12-09`
- Type: transactional e-commerce retail data

## Architecture

### Data and semantic layer

- `fact_sales.csv`
- `fact_orders.csv`
- `dim_date.csv`
- `dim_customer.csv`
- `dim_product.csv`

### BI outputs

- executive KPI summary
- market and category performance tables
- weekday and monthly performance
- top-product tables
- customer and product segment summaries

### AI outputs

- daily revenue forecasting
- anomaly detection on sales behavior
- product clustering
- recommendation engine

## Repository structure

```text
BI+AI Project/
|-- app.py
|-- main.py
|-- requirements.txt
|-- data/
|-- models/
|-- presentation/
|-- reports/
`-- src/bi_ai_retail/
```

## How to run

Install dependencies:

```powershell
py -3.14 -m pip install -r requirements.txt
```

Build or refresh outputs:

```powershell
py -3.14 main.py
```

Launch the dashboard:

```powershell
py -3.14 -m streamlit run app.py
```

## Main generated outputs

All outputs are written to `data/processed/`.

Key files include:

- `kpis.json`
- `model_metrics.json`
- `order_summary.csv`
- `customer_summary.csv`
- `daily_sales.csv`
- `product_segments.csv`
- `customer_segment_summary.csv`
- `recommendations.csv`
- `fact_sales.csv`
- `fact_orders.csv`
- `dim_date.csv`
- `dim_customer.csv`
- `dim_product.csv`

## Current headline results

- Revenue: `$10.64M`
- Profit: `$3.41M`
- Gross margin: `32.0%`
- Orders: `19,960`
- Customers: `4,339`
- Repeat customer rate: `65.6%`
- Forecast next 30 days: `$1.16M`
- Anomaly days detected: `11`

## Why this is strong for BI + AI

This project demonstrates:

- data engineering and semantic modeling
- KPI design for decision-making
- interactive BI dashboard development
- predictive analytics
- anomaly detection
- customer and product segmentation
- business recommendation logic

## Presentation summary

> This project delivers a retail decision-intelligence platform that combines BI reporting with AI-driven forecasting and pattern detection.  
> It supports executives, analysts, and operations teams with a shared view of performance, risk, and next-step actions.
