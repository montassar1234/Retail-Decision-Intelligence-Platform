from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, FORECAST_HORIZON_DAYS, MODELS_DIR, PROCESSED_DIR, RAW_DATA_PATH, RAW_DATA_URL, RAW_DIR, REPORTS_DIR, TRAIN_CUTOFF_DATE
from .modeling import detect_anomalies, segment_products, train_forecasting_model
from .recommendations import build_recommendations
from .reporting import build_markdown_report, save_json


REGION_MAP = {
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Portugal": "Europe",
    "EIRE": "Europe",
    "Switzerland": "Europe",
    "Norway": "Europe",
    "Italy": "Europe",
    "Cyprus": "Europe",
    "Austria": "Europe",
    "Finland": "Europe",
    "Denmark": "Europe",
    "Poland": "Europe",
    "Sweden": "Europe",
    "Iceland": "Europe",
    "Channel Islands": "Europe",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Japan": "Asia",
    "Singapore": "Asia",
    "Hong Kong": "Asia",
    "Israel": "Asia",
    "Lebanon": "Asia",
    "United Arab Emirates": "Asia",
    "Saudi Arabia": "Asia",
    "USA": "North America",
    "Canada": "North America",
    "Brazil": "South America",
    "Bahrain": "Asia",
    "South Africa": "Africa",
}

CATEGORY_RULES = {
    "Home Decor": ["LANTERN", "HEART", "CANDLE", "T-LIGHT", "HOLDER", "FRAME", "HOOK", "DECORATION", "VASE"],
    "Kitchen & Dining": ["MUG", "BOWL", "PLATE", "JAR", "CUP", "TEA", "SPOON", "GLASS", "DINNER", "TRAY"],
    "Seasonal & Gifts": ["CHRISTMAS", "ADVENT", "GIFT", "CARD", "PARTY", "BIRTHDAY", "WRAP", "PRESENT"],
    "Accessories": ["BAG", "PURSE", "NECKLACE", "BRACELET", "RING", "BROOCH", "CHARM"],
    "Kids & Toys": ["DOLL", "TOY", "CHILD", "BABY", "KIDS", "PLAY", "TEDDY"],
    "Storage & Organization": ["BOX", "DRAWER", "BASKET", "CABINET", "TIN", "STORAGE"],
    "Stationery": ["PEN", "NOTEBOOK", "PAPER", "DIARY", "ENVELOPE", "LABEL", "STICKER"],
}


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def download_dataset(force: bool = False) -> Path:
    ensure_directories()
    if force or not RAW_DATA_PATH.exists():
        urlretrieve(RAW_DATA_URL, RAW_DATA_PATH)
    return RAW_DATA_PATH


def classify_product(description: str) -> str:
    text = (description or "").upper()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "General Merchandise"


def classify_customer_segment(recency_score: int, frequency_score: int, monetary_score: int) -> str:
    if recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4:
        return "Champions"
    if recency_score >= 3 and frequency_score >= 3:
        return "Loyal"
    if recency_score >= 4 and frequency_score <= 2:
        return "Promising"
    if recency_score <= 2 and frequency_score >= 3:
        return "Needs attention"
    return "At risk"


def load_and_clean_data() -> pd.DataFrame:
    download_dataset()
    df = pd.read_csv(RAW_DATA_PATH, encoding="ISO-8859-1")
    df = df.drop_duplicates().copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M", errors="coerce")
    df = df[df["InvoiceDate"].notna()].copy()
    df = df[df["Quantity"] > 0].copy()
    df = df[df["UnitPrice"] > 0].copy()
    df = df[df["Description"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int).astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["EstimatedCost"] = df["Revenue"] * 0.68
    df["Profit"] = df["Revenue"] - df["EstimatedCost"]
    df["ProfitMarginPct"] = np.where(df["Revenue"] == 0, 0, (df["Profit"] / df["Revenue"]) * 100)
    df["OrderDate"] = df["InvoiceDate"].dt.floor("D")
    df["OrderMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["MonthName"] = df["InvoiceDate"].dt.month_name()
    df["Year"] = df["InvoiceDate"].dt.year
    df["Quarter"] = "Q" + df["InvoiceDate"].dt.quarter.astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["WeekOfYear"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    df["Category"] = df["Description"].apply(classify_product)
    df["Region"] = df["Country"].map(REGION_MAP).fillna("Other")
    df["OrderLineKey"] = np.arange(1, len(df) + 1)
    return df


def build_order_summary(clean_df: pd.DataFrame) -> pd.DataFrame:
    order_summary = (
        clean_df.groupby("InvoiceNo")
        .agg(
            order_date=("OrderDate", "min"),
            invoice_timestamp=("InvoiceDate", "min"),
            customer_id=("CustomerID", "first"),
            country=("Country", "first"),
            region=("Region", "first"),
            lines=("OrderLineKey", "count"),
            unique_products=("StockCode", "nunique"),
            total_quantity=("Quantity", "sum"),
            revenue=("Revenue", "sum"),
            profit=("Profit", "sum"),
        )
        .reset_index()
    )
    order_summary["avg_item_price"] = order_summary["revenue"] / order_summary["total_quantity"]
    order_summary["profit_margin_pct"] = np.where(order_summary["revenue"] == 0, 0, (order_summary["profit"] / order_summary["revenue"]) * 100)
    return order_summary


def build_daily_sales(order_summary: pd.DataFrame) -> pd.DataFrame:
    daily_sales = (
        order_summary.groupby("order_date")
        .agg(
            daily_revenue=("revenue", "sum"),
            daily_profit=("profit", "sum"),
            daily_quantity=("total_quantity", "sum"),
            order_count=("InvoiceNo", "nunique"),
            active_customers=("customer_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    daily_sales["avg_order_value"] = daily_sales["daily_revenue"] / daily_sales["order_count"]
    daily_sales["profit_margin_pct"] = np.where(daily_sales["daily_revenue"] == 0, 0, (daily_sales["daily_profit"] / daily_sales["daily_revenue"]) * 100)
    return daily_sales


def build_customer_summary(order_summary: pd.DataFrame, max_order_date: pd.Timestamp) -> pd.DataFrame:
    customer_summary = (
        order_summary.groupby("customer_id")
        .agg(
            first_order_date=("order_date", "min"),
            last_order_date=("order_date", "max"),
            country=("country", "first"),
            region=("region", "first"),
            order_count=("InvoiceNo", "nunique"),
            total_revenue=("revenue", "sum"),
            total_profit=("profit", "sum"),
            quantity=("total_quantity", "sum"),
        )
        .reset_index()
    )
    customer_summary["recency_days"] = (max_order_date - customer_summary["last_order_date"]).dt.days
    customer_summary["avg_order_value"] = customer_summary["total_revenue"] / customer_summary["order_count"]
    customer_summary["profit_margin_pct"] = np.where(
        customer_summary["total_revenue"] == 0,
        0,
        (customer_summary["total_profit"] / customer_summary["total_revenue"]) * 100,
    )

    customer_summary["recency_score"] = pd.qcut(
        customer_summary["recency_days"].rank(method="first", ascending=True),
        5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)
    customer_summary["frequency_score"] = pd.qcut(
        customer_summary["order_count"].rank(method="first", ascending=True),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    customer_summary["monetary_score"] = pd.qcut(
        customer_summary["total_revenue"].rank(method="first", ascending=True),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    customer_summary["rfm_segment"] = customer_summary.apply(
        lambda row: classify_customer_segment(row["recency_score"], row["frequency_score"], row["monetary_score"]),
        axis=1,
    )
    return customer_summary


def build_product_segments(clean_df: pd.DataFrame) -> pd.DataFrame:
    monthly_revenue = (
        clean_df.groupby(["StockCode", "Description", "OrderMonth"])["Revenue"]
        .sum()
        .reset_index()
    )
    variability = monthly_revenue.groupby(["StockCode", "Description"])["Revenue"].std().fillna(0).reset_index(name="revenue_volatility")

    product_frame = (
        clean_df.groupby(["StockCode", "Description", "Category"])
        .agg(
            total_revenue=("Revenue", "sum"),
            total_profit=("Profit", "sum"),
            total_quantity=("Quantity", "sum"),
            avg_unit_price=("UnitPrice", "mean"),
            transaction_count=("InvoiceNo", "nunique"),
            active_months=("OrderMonth", "nunique"),
        )
        .reset_index()
    )
    product_frame["avg_monthly_revenue"] = product_frame["total_revenue"] / product_frame["active_months"]
    product_frame["profit_margin_pct"] = np.where(
        product_frame["total_revenue"] == 0,
        0,
        (product_frame["total_profit"] / product_frame["total_revenue"]) * 100,
    )
    product_frame = product_frame.merge(variability, on=["StockCode", "Description"], how="left")
    product_frame["revenue_volatility"] = product_frame["revenue_volatility"].fillna(0)
    product_frame["revenue_share_pct"] = (product_frame["total_revenue"] / product_frame["total_revenue"].sum()) * 100
    product_frame["revenue_rank_pct"] = product_frame["total_revenue"].rank(pct=True, ascending=False)
    product_frame["abc_class"] = np.select(
        [product_frame["revenue_rank_pct"] <= 0.2, product_frame["revenue_rank_pct"] <= 0.5],
        ["A", "B"],
        default="C",
    )
    return segment_products(product_frame)


def build_date_dimension(clean_df: pd.DataFrame) -> pd.DataFrame:
    date_dim = pd.DataFrame({"date": pd.date_range(clean_df["OrderDate"].min(), clean_df["OrderDate"].max(), freq="D")})
    date_dim["date_key"] = date_dim["date"].dt.strftime("%Y%m%d").astype(int)
    date_dim["year"] = date_dim["date"].dt.year
    date_dim["quarter"] = "Q" + date_dim["date"].dt.quarter.astype(str)
    date_dim["month"] = date_dim["date"].dt.month
    date_dim["month_name"] = date_dim["date"].dt.month_name()
    date_dim["year_month"] = date_dim["date"].dt.to_period("M").astype(str)
    date_dim["week_of_year"] = date_dim["date"].dt.isocalendar().week.astype(int)
    date_dim["day_of_week"] = date_dim["date"].dt.day_name()
    date_dim["is_weekend"] = date_dim["date"].dt.dayofweek.isin([5, 6]).astype(int)
    return date_dim


def build_aggregates(clean_df: pd.DataFrame, order_summary: pd.DataFrame, customer_summary: pd.DataFrame, product_segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    country_perf = (
        order_summary.groupby(["country", "region"])
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"country": "Country", "region": "Region"})
        .sort_values("revenue", ascending=False)
    )
    country_perf["profit_margin_pct"] = np.where(country_perf["revenue"] == 0, 0, (country_perf["profit"] / country_perf["revenue"]) * 100)
    country_perf["avg_order_value"] = country_perf["revenue"] / country_perf["orders"]

    category_perf = (
        clean_df.groupby("Category")
        .agg(
            revenue=("Revenue", "sum"),
            profit=("Profit", "sum"),
            quantity=("Quantity", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    category_perf["profit_margin_pct"] = np.where(category_perf["revenue"] == 0, 0, (category_perf["profit"] / category_perf["revenue"]) * 100)
    category_perf["avg_order_value"] = category_perf["revenue"] / category_perf["orders"]

    monthly_perf = (
        order_summary.assign(OrderMonth=order_summary["order_date"].dt.to_period("M").astype(str))
        .groupby("OrderMonth")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"OrderMonth": "Month"})
    )
    monthly_perf["month_date"] = pd.to_datetime(monthly_perf["Month"] + "-01")
    monthly_perf["profit_margin_pct"] = np.where(monthly_perf["revenue"] == 0, 0, (monthly_perf["profit"] / monthly_perf["revenue"]) * 100)

    region_monthly = (
        clean_df.groupby(["OrderMonth", "Region"])["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"OrderMonth": "Month", "Revenue": "revenue"})
    )
    region_monthly["month_date"] = pd.to_datetime(region_monthly["Month"] + "-01")

    weekday_performance = (
        order_summary.assign(DayOfWeek=order_summary["order_date"].dt.day_name())
        .groupby("DayOfWeek")
        .agg(
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_performance["DayOfWeek"] = pd.Categorical(weekday_performance["DayOfWeek"], categories=weekday_order, ordered=True)
    weekday_performance = weekday_performance.sort_values("DayOfWeek")

    top_products = product_segments.sort_values("total_revenue", ascending=False).head(25).copy()

    customer_segment_summary = (
        customer_summary.groupby("rfm_segment")
        .agg(
            customers=("customer_id", "nunique"),
            revenue=("total_revenue", "sum"),
            avg_order_value=("avg_order_value", "mean"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    product_segment_summary = (
        product_segments.groupby("segment_name")
        .agg(
            products=("StockCode", "nunique"),
            revenue=("total_revenue", "sum"),
            profit=("total_profit", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    return {
        "country_performance": country_perf,
        "category_performance": category_perf,
        "monthly_performance": monthly_perf.sort_values("month_date"),
        "region_monthly_performance": region_monthly.sort_values("month_date"),
        "weekday_performance": weekday_performance,
        "top_products": top_products,
        "customer_segment_summary": customer_segment_summary,
        "product_segment_summary": product_segment_summary,
    }


def build_semantic_outputs(clean_df: pd.DataFrame, order_summary: pd.DataFrame, customer_summary: pd.DataFrame, product_segments: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fact_sales = clean_df[
        [
            "OrderLineKey",
            "InvoiceNo",
            "InvoiceDate",
            "OrderDate",
            "CustomerID",
            "StockCode",
            "Description",
            "Country",
            "Region",
            "Category",
            "Quantity",
            "UnitPrice",
            "Revenue",
            "EstimatedCost",
            "Profit",
        ]
    ].copy()

    dim_date = build_date_dimension(clean_df)
    dim_customer = customer_summary.rename(
        columns={
            "customer_id": "CustomerID",
            "country": "Country",
            "region": "Region",
        }
    )
    dim_product = product_segments.rename(columns={"profit_margin_pct": "product_margin_pct"})
    fact_orders = order_summary.rename(
        columns={
            "customer_id": "CustomerID",
            "country": "Country",
            "region": "Region",
        }
    )
    return {
        "fact_sales": fact_sales,
        "fact_orders": fact_orders,
        "dim_date": dim_date,
        "dim_customer": dim_customer,
        "dim_product": dim_product,
    }


def build_kpis(
    clean_df: pd.DataFrame,
    order_summary: pd.DataFrame,
    customer_summary: pd.DataFrame,
    daily_sales: pd.DataFrame,
    forecast_actuals: pd.DataFrame,
    forecast_future: pd.DataFrame,
    anomalies: pd.DataFrame,
    category_perf: pd.DataFrame,
    country_perf: pd.DataFrame,
) -> dict[str, float | int | str]:
    total_revenue = float(order_summary["revenue"].sum())
    total_profit = float(order_summary["profit"].sum())
    orders_count = int(order_summary["InvoiceNo"].nunique())
    customers_count = int(customer_summary["customer_id"].nunique())
    countries_count = int(clean_df["Country"].nunique())
    average_order_value = float(total_revenue / max(orders_count, 1))
    average_items_per_order = float(order_summary["total_quantity"].sum() / max(orders_count, 1))
    gross_margin_pct = float((total_profit / max(total_revenue, 1)) * 100)
    repeat_customer_rate = float((customer_summary["order_count"].gt(1).mean()) * 100)

    latest_30 = daily_sales.tail(30)["daily_revenue"].sum()
    previous_30 = daily_sales.tail(60).head(30)["daily_revenue"].sum()
    sales_growth_pct = float(((latest_30 - previous_30) / max(previous_30, 1)) * 100)
    forecast_30d_revenue = float(forecast_future["predicted_revenue"].sum())
    forecast_delta_pct = float(((forecast_30d_revenue - latest_30) / max(latest_30, 1)) * 100)
    forecast_accuracy_pct = float(100 - forecast_actuals["absolute_error"].sum() / max(forecast_actuals["daily_revenue"].sum(), 1) * 100)

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "orders_count": orders_count,
        "customers_count": customers_count,
        "countries_count": countries_count,
        "average_order_value": average_order_value,
        "average_items_per_order": average_items_per_order,
        "gross_margin_pct": gross_margin_pct,
        "repeat_customer_rate": repeat_customer_rate,
        "sales_growth_pct": sales_growth_pct,
        "forecast_30d_revenue": forecast_30d_revenue,
        "forecast_delta_pct": forecast_delta_pct,
        "forecast_accuracy_pct": forecast_accuracy_pct,
        "anomaly_days": int(len(anomalies)),
        "top_country": str(country_perf.iloc[0]["Country"]),
        "top_category": str(category_perf.iloc[0]["Category"]),
        "date_range_start": clean_df["OrderDate"].min().strftime("%Y-%m-%d"),
        "date_range_end": clean_df["OrderDate"].max().strftime("%Y-%m-%d"),
    }


def run_pipeline() -> dict[str, object]:
    clean_df = load_and_clean_data()
    order_summary = build_order_summary(clean_df)
    daily_sales = build_daily_sales(order_summary)
    customer_summary = build_customer_summary(order_summary, order_summary["order_date"].max())
    product_segments = build_product_segments(clean_df)
    forecast_artifacts = train_forecasting_model(
        daily_sales=daily_sales,
        train_cutoff_date=TRAIN_CUTOFF_DATE,
        forecast_horizon_days=FORECAST_HORIZON_DAYS,
    )
    anomalies = detect_anomalies(daily_sales)
    aggregates = build_aggregates(clean_df, order_summary, customer_summary, product_segments)
    kpis = build_kpis(
        clean_df=clean_df,
        order_summary=order_summary,
        customer_summary=customer_summary,
        daily_sales=daily_sales,
        forecast_actuals=forecast_artifacts.forecast_actuals,
        forecast_future=forecast_artifacts.future_forecast,
        anomalies=anomalies,
        category_perf=aggregates["category_performance"],
        country_perf=aggregates["country_performance"],
    )
    recommendations = build_recommendations(
        kpis=kpis,
        future_forecast=forecast_artifacts.future_forecast,
        recent_daily_sales=daily_sales,
        anomalies=anomalies,
        product_segments=product_segments,
        category_performance=aggregates["category_performance"],
        customer_segments=aggregates["customer_segment_summary"],
        country_performance=aggregates["country_performance"],
    )
    semantic_outputs = build_semantic_outputs(clean_df, order_summary, customer_summary, product_segments)

    clean_df.to_csv(PROCESSED_DIR / "clean_retail.csv", index=False)
    order_summary.to_csv(PROCESSED_DIR / "order_summary.csv", index=False)
    customer_summary.to_csv(PROCESSED_DIR / "customer_summary.csv", index=False)
    daily_sales.to_csv(PROCESSED_DIR / "daily_sales.csv", index=False)
    forecast_artifacts.forecast_actuals.to_csv(PROCESSED_DIR / "forecast_actuals.csv", index=False)
    forecast_artifacts.future_forecast.to_csv(PROCESSED_DIR / "future_forecast.csv", index=False)
    anomalies.to_csv(PROCESSED_DIR / "anomalies.csv", index=False)
    product_segments.to_csv(PROCESSED_DIR / "product_segments.csv", index=False)
    recommendations.to_csv(PROCESSED_DIR / "recommendations.csv", index=False)

    for name, frame in aggregates.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    for name, frame in semantic_outputs.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    save_json(PROCESSED_DIR / "kpis.json", kpis)
    save_json(PROCESSED_DIR / "model_metrics.json", forecast_artifacts.metrics)
    build_markdown_report(REPORTS_DIR / "project_summary.md", kpis, forecast_artifacts.metrics, recommendations)
    joblib.dump(forecast_artifacts.model, MODELS_DIR / "forecast_model.joblib")

    return {
        "clean_df": clean_df,
        "order_summary": order_summary,
        "customer_summary": customer_summary,
        "daily_sales": daily_sales,
        "forecast_actuals": forecast_artifacts.forecast_actuals,
        "future_forecast": forecast_artifacts.future_forecast,
        "anomalies": anomalies,
        "product_segments": product_segments,
        "aggregates": aggregates,
        "recommendations": recommendations,
        "kpis": kpis,
        "model_metrics": forecast_artifacts.metrics,
        "semantic_outputs": semantic_outputs,
    }
