# Retail Decision Intelligence Platform

Retail Decision Intelligence Platform is a BI + AI project that transforms retail transaction data into an interactive decision-support system. It combines business intelligence reporting with machine learning to monitor performance, forecast sales, detect anomalies, segment customers and products, and generate actionable recommendations.

## Project Overview

This project was developed as an academic BI + AI solution using open-source retail data. The objective is to move beyond static reporting by combining descriptive analytics and predictive analytics in one platform.

The system helps answer questions such as:

- What are the main revenue and profit drivers?
- Which countries and product categories perform best?
- What sales level can be expected in the next 30 days?
- Which unusual sales patterns should be investigated?
- Which customers and products need strategic attention?

## Objectives

- Prepare and clean retail transaction data for analysis
- Build key business indicators for performance monitoring
- Forecast future sales using machine learning
- Detect anomalies in historical sales behavior
- Segment customers and products for decision support
- Deliver an interactive dashboard for analysis and presentation

## Dataset

- Source: [Online Retail dataset](https://raw.githubusercontent.com/dbdmg/data-science-lab/master/datasets/online_retail.csv)
- Type: transactional e-commerce retail data
- Period covered: `2010-12-01` to `2011-12-09`

The dataset includes invoice information, product descriptions, quantities, unit prices, customer identifiers, and country information.

## Methodology

### 1. Data Preparation

The raw dataset was cleaned by:

- removing duplicates
- excluding invalid transactions
- converting date fields
- creating calculated fields such as revenue, estimated cost, profit, and profit margin
- deriving business dimensions such as category, region, month, and day of week

### 2. Business Intelligence Layer

The BI component provides:

- revenue, profit, and margin KPIs
- average order value and repeat customer rate
- country and regional performance analysis
- category and product performance analysis
- monthly and weekday trend monitoring

### 3. Artificial Intelligence Layer

The AI component includes:

- sales forecasting using a machine learning regression model
- anomaly detection for unusual spikes and drops in sales
- customer segmentation using RFM analysis
- product segmentation using clustering
- recommendation logic based on analytical findings

### 4. Dashboard

The final dashboard was developed with Streamlit and includes:

- executive overview
- commercial performance analysis
- forecasting and anomaly views
- customer and product segment insights
- filter-driven reporting
- interactive world map selection

## Tools and Technologies

- Python
- Pandas
- NumPy
- scikit-learn
- Plotly
- Streamlit
- Joblib
- python-pptx

## Project Structure

```text
Retail-Decision-Intelligence-Platform/
|-- app.py
|-- main.py
|-- requirements.txt
|-- dashboard/
|-- notebooks/
|-- presentation/
|-- reports/
`-- src/bi_ai_retail/
```

## How to Run

Install dependencies:

```powershell
py -3.14 -m pip install -r requirements.txt
```

Run the data pipeline:

```powershell
py -3.14 main.py
```

Launch the dashboard:

```powershell
py -3.14 -m streamlit run app.py
```

## Main Outputs

The project generates:

- cleaned and transformed retail data
- KPI summaries
- forecast results
- anomaly detection outputs
- customer and product segmentation outputs
- business recommendations
- dashboard views for analysis and presentation

## Results Summary

- Revenue analysed: `$10.64M`
- Profit analysed: `$3.41M`
- Gross margin: `32.0%`
- Orders analysed: `19,960`
- Customers analysed: `4,339`
- Countries analysed: `38`
- Repeat customer rate: `65.6%`
- Forecasted revenue for the next 30 days: `$1.16M`
- Detected anomaly days: `11`

## Academic Value

This project demonstrates practical skills in:

- data cleaning and transformation
- KPI design
- business intelligence reporting
- machine learning for forecasting
- anomaly detection
- segmentation
- dashboard development
- business-oriented data storytelling

## Conclusion

Retail Decision Intelligence Platform shows how BI and AI can be combined in a realistic retail use case. The BI component explains what has happened in the business, while the AI component provides predictive and analytical support for future decisions.
