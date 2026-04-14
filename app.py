from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bi_ai_retail.data_pipeline import run_pipeline


st.set_page_config(page_title="Retail Decision Intelligence", layout="wide")


def format_money(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


@st.cache_data(show_spinner=False)
def load_or_build() -> dict[str, pd.DataFrame | dict]:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    required_files = [
        "clean_retail.csv",
        "order_summary.csv",
        "customer_summary.csv",
        "daily_sales.csv",
        "future_forecast.csv",
        "forecast_actuals.csv",
        "anomalies.csv",
        "product_segments.csv",
        "country_performance.csv",
        "category_performance.csv",
        "monthly_performance.csv",
        "region_monthly_performance.csv",
        "weekday_performance.csv",
        "top_products.csv",
        "customer_segment_summary.csv",
        "product_segment_summary.csv",
        "recommendations.csv",
        "kpis.json",
        "model_metrics.json",
    ]
    if not all((processed_dir / filename).exists() for filename in required_files):
        run_pipeline()

    return {
        "clean_retail": pd.read_csv(
            processed_dir / "clean_retail.csv",
            parse_dates=["InvoiceDate", "OrderDate"],
            dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str},
            low_memory=False,
        ),
        "daily_sales": pd.read_csv(processed_dir / "daily_sales.csv", parse_dates=["date"]),
        "future_forecast": pd.read_csv(processed_dir / "future_forecast.csv", parse_dates=["date"]),
        "forecast_actuals": pd.read_csv(processed_dir / "forecast_actuals.csv", parse_dates=["date"]),
        "anomalies": pd.read_csv(processed_dir / "anomalies.csv", parse_dates=["date"]),
        "product_segments": pd.read_csv(processed_dir / "product_segments.csv"),
        "weekday_performance": pd.read_csv(processed_dir / "weekday_performance.csv"),
        "customer_segment_summary": pd.read_csv(processed_dir / "customer_segment_summary.csv"),
        "product_segment_summary": pd.read_csv(processed_dir / "product_segment_summary.csv"),
        "recommendations": pd.read_csv(processed_dir / "recommendations.csv"),
        "kpis": json.loads((processed_dir / "kpis.json").read_text(encoding="utf-8")),
        "model_metrics": json.loads((processed_dir / "model_metrics.json").read_text(encoding="utf-8")),
    }


def build_filtered_views(clean_retail: pd.DataFrame, region_filter: list[str], category_filter: list[str], country_filter: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = clean_retail.copy()
    if region_filter:
        filtered = filtered[filtered["Region"].isin(region_filter)]
    if category_filter:
        filtered = filtered[filtered["Category"].isin(category_filter)]
    if country_filter:
        filtered = filtered[filtered["Country"].isin(country_filter)]

    orders = (
        filtered.groupby("InvoiceNo")
        .agg(
            order_date=("OrderDate", "min"),
            customer_id=("CustomerID", "first"),
            country=("Country", "first"),
            region=("Region", "first"),
            revenue=("Revenue", "sum"),
            profit=("Profit", "sum"),
            quantity=("Quantity", "sum"),
        )
        .reset_index()
    )
    return filtered, orders


def render_metric_cards(filtered_sales: pd.DataFrame, filtered_orders: pd.DataFrame) -> None:
    total_revenue = filtered_sales["Revenue"].sum()
    total_profit = filtered_sales["Profit"].sum()
    total_orders = filtered_orders["InvoiceNo"].nunique()
    total_customers = filtered_sales["CustomerID"].nunique()
    margin = (total_profit / total_revenue * 100) if total_revenue else 0
    aov = (total_revenue / total_orders) if total_orders else 0

    cols = st.columns(6)
    cols[0].metric("Revenue", format_money(total_revenue))
    cols[1].metric("Profit", format_money(total_profit))
    cols[2].metric("Margin", f"{margin:.1f}%")
    cols[3].metric("Orders", f"{total_orders:,}")
    cols[4].metric("Customers", f"{total_customers:,}")
    cols[5].metric("AOV", format_money(aov))


def build_country_report(filtered_sales: pd.DataFrame, selected_country: str | None) -> dict[str, str | float | int]:
    scope_sales = filtered_sales if not selected_country else filtered_sales[filtered_sales["Country"] == selected_country]
    if scope_sales.empty:
        return {
            "title": "No data for selected scope",
            "revenue": 0.0,
            "profit": 0.0,
            "orders": 0,
            "customers": 0,
            "top_category": "N/A",
            "summary": "No transactions are available for the current filters.",
        }

    order_count = scope_sales["InvoiceNo"].nunique()
    customer_count = scope_sales["CustomerID"].nunique()
    total_revenue = float(scope_sales["Revenue"].sum())
    total_profit = float(scope_sales["Profit"].sum())
    category_summary = scope_sales.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
    top_category = category_summary.index[0]
    country_label = selected_country or "Current filtered scope"
    summary = (
        f"{country_label} generated {format_money(total_revenue)} in revenue from {order_count:,} orders "
        f"and {customer_count:,} customers. The leading category is {top_category}."
    )
    return {
        "title": country_label,
        "revenue": total_revenue,
        "profit": total_profit,
        "orders": order_count,
        "customers": customer_count,
        "top_category": top_category,
        "summary": summary,
    }


data = load_or_build()
kpis = data["kpis"]
metrics = data["model_metrics"]
clean_retail = data["clean_retail"]

if "map_country" not in st.session_state:
    st.session_state.map_country = None

st.title("Retail Decision Intelligence Platform")
st.caption("Executive BI plus predictive AI for retail performance, demand planning, and action management")

with st.sidebar:
    st.header("Filters")
    region_filter = st.multiselect("Region", sorted(clean_retail["Region"].dropna().unique()))
    category_filter = st.multiselect("Category", sorted(clean_retail["Category"].dropna().unique()))
    manual_country_filter = st.multiselect("Country", sorted(clean_retail["Country"].dropna().unique()))

    st.divider()
    st.markdown("**Map selection**")
    if st.session_state.map_country:
        st.write(f"Selected on map: `{st.session_state.map_country}`")
    if st.button("Clear map selection", use_container_width=True):
        st.session_state.map_country = None
        st.rerun()

    st.divider()
    st.markdown("**Enterprise KPIs**")
    st.write(f"Forecast next 30 days: {format_money(kpis['forecast_30d_revenue'])}")
    st.write(f"Forecast delta: {kpis['forecast_delta_pct']:.1f}%")
    st.write(f"Repeat customer rate: {kpis['repeat_customer_rate']:.1f}%")
    st.write(f"Forecast accuracy proxy: {kpis['forecast_accuracy_pct']:.1f}%")

effective_country_filter = list(manual_country_filter)
if st.session_state.map_country and st.session_state.map_country not in effective_country_filter:
    effective_country_filter.append(st.session_state.map_country)

filtered_sales, filtered_orders = build_filtered_views(clean_retail, region_filter, category_filter, effective_country_filter)
render_metric_cards(filtered_sales, filtered_orders)

report_scope = build_country_report(filtered_sales, st.session_state.map_country if st.session_state.map_country in filtered_sales["Country"].unique() else None)

overview_tab, commercial_tab, ai_tab, action_tab = st.tabs(
    ["Executive Overview", "Commercial Performance", "AI Insights", "Action Center"]
)

with overview_tab:
    st.subheader("Scope Report")
    report_cols = st.columns([1.3, 1, 1, 1, 1])
    report_cols[0].markdown(f"**{report_scope['title']}**  \n{report_scope['summary']}")
    report_cols[1].metric("Revenue", format_money(float(report_scope["revenue"])))
    report_cols[2].metric("Profit", format_money(float(report_scope["profit"])))
    report_cols[3].metric("Orders", f"{int(report_scope['orders']):,}")
    report_cols[4].metric("Top Category", str(report_scope["top_category"]))

    left, right = st.columns([1.7, 1])
    trend = (
        filtered_sales.groupby("OrderDate")
        .agg(revenue=("Revenue", "sum"), profit=("Profit", "sum"))
        .reset_index()
        .sort_values("OrderDate")
    )
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=trend["OrderDate"], y=trend["revenue"], name="Revenue"))
    trend_fig.add_trace(go.Scatter(x=trend["OrderDate"], y=trend["profit"], name="Profit"))
    trend_fig.update_layout(title="Revenue and Profit Trend", margin=dict(l=20, r=20, t=50, b=20))
    left.plotly_chart(trend_fig, width="stretch")

    country_summary = (
        filtered_sales.groupby(["Country", "Region"])["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(10)
    )
    country_fig = px.bar(country_summary, x="Country", y="Revenue", color="Region", title="Top Markets by Revenue")
    right.plotly_chart(country_fig, width="stretch")

    monthly = (
        filtered_sales.assign(Month=filtered_sales["OrderDate"].dt.to_period("M").astype(str))
        .groupby("Month")
        .agg(revenue=("Revenue", "sum"), orders=("InvoiceNo", "nunique"))
        .reset_index()
    )
    monthly_mix = px.bar(monthly, x="Month", y="revenue", title="Monthly Revenue Run Rate")
    st.plotly_chart(monthly_mix, width="stretch")

    geo_summary = (
        filtered_sales.groupby(["Country", "Region"], as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            profit=("Profit", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
    )
    geo_summary["iso_country"] = geo_summary["Country"].replace(
        {"EIRE": "Ireland", "Channel Islands": "United Kingdom", "USA": "United States"}
    )
    geo_fig = px.choropleth(
        geo_summary,
        locations="iso_country",
        locationmode="country names",
        color="revenue",
        hover_name="Country",
        custom_data=["Country"],
        hover_data={"Region": True, "revenue": ":,.0f", "profit": ":,.0f", "orders": True, "iso_country": False},
        color_continuous_scale="Blues",
        title="Global Revenue Heatmap",
    )
    geo_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    selected_points = plotly_events(geo_fig, click_event=True, hover_event=False, select_event=False, override_height=500, key="world_map")
    if selected_points:
        point = selected_points[0]
        selected_country = None
        if "customdata" in point and point["customdata"]:
            selected_country = point["customdata"][0]
        elif "location" in point:
            reverse_map = {"Ireland": "EIRE", "United Kingdom": "United Kingdom", "United States": "USA"}
            selected_country = reverse_map.get(point["location"], point["location"])
        if selected_country and selected_country != st.session_state.map_country:
            st.session_state.map_country = selected_country
            st.rerun()
    st.caption("Click a country on the map to update the scope report and all filtered visuals.")

with commercial_tab:
    left, right = st.columns(2)
    category_summary = (
        filtered_sales.groupby("Category")
        .agg(revenue=("Revenue", "sum"), profit=("Profit", "sum"), quantity=("Quantity", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    category_summary["profit_margin_pct"] = np.where(
        category_summary["revenue"] == 0,
        0,
        category_summary["profit"] / category_summary["revenue"] * 100,
    )
    category_fig = px.bar(
        category_summary,
        x="Category",
        y="revenue",
        color="profit_margin_pct",
        color_continuous_scale="Tealgrn",
        title="Category Revenue and Margin Quality",
    )
    left.plotly_chart(category_fig, width="stretch")

    weekday_scope = (
        filtered_sales.assign(DayOfWeek=filtered_sales["OrderDate"].dt.day_name())
        .groupby("DayOfWeek")["Revenue"]
        .sum()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .reset_index()
    )
    weekday_fig = px.line(
        weekday_scope,
        x="DayOfWeek",
        y="Revenue",
        markers=True,
        title="Revenue by Day of Week",
    )
    right.plotly_chart(weekday_fig, width="stretch")

    product_table = (
        filtered_sales.groupby(["StockCode", "Description", "Category"])
        .agg(revenue=("Revenue", "sum"), profit=("Profit", "sum"), quantity=("Quantity", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(15)
    )
    product_table["margin_pct"] = np.where(product_table["revenue"] == 0, 0, product_table["profit"] / product_table["revenue"] * 100)
    st.dataframe(product_table, width="stretch", hide_index=True)

with ai_tab:
    left, right = st.columns([1.6, 1])
    combined = data["forecast_actuals"][["date", "daily_revenue", "predicted_revenue"]].copy()
    future = data["future_forecast"].copy()
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=combined["date"], y=combined["daily_revenue"], name="Actual"))
    forecast_fig.add_trace(go.Scatter(x=combined["date"], y=combined["predicted_revenue"], name="Predicted"))
    forecast_fig.add_trace(go.Scatter(x=future["date"], y=future["predicted_revenue"], name="Future forecast"))
    forecast_fig.update_layout(title="Forecast Performance and Forward View", margin=dict(l=20, r=20, t=50, b=20))
    left.plotly_chart(forecast_fig, width="stretch")

    right.metric("MAE", f"{metrics['mae']:,.0f}")
    right.metric("RMSE", f"{metrics['rmse']:,.0f}")
    right.metric("MAPE", f"{metrics['mape']:.1f}%")
    right.metric("Forecast Accuracy Proxy", f"{kpis['forecast_accuracy_pct']:.1f}%")
    right.metric("Anomaly Days", f"{kpis['anomaly_days']}")

    bottom_left, bottom_right = st.columns(2)
    anomaly_scope = data["anomalies"]
    if effective_country_filter:
        anomaly_dates = set(filtered_sales["OrderDate"].dt.normalize())
        anomaly_scope = anomaly_scope[anomaly_scope["date"].dt.normalize().isin(anomaly_dates)]
    anomaly_fig = px.scatter(
        anomaly_scope.assign(size_value=lambda df: df["z_score"].abs().clip(lower=0.1)),
        x="date",
        y="daily_revenue",
        color="anomaly_type",
        size="size_value",
        title="Anomaly Detection Output",
        color_discrete_map={"Spike": "#d9982b", "Drop": "#bf4040"},
    )
    bottom_left.plotly_chart(anomaly_fig, width="stretch")

    segment_scope = data["product_segments"]
    if category_filter:
        segment_scope = segment_scope[segment_scope["Category"].isin(category_filter)]
    segment_fig = px.scatter(
        segment_scope,
        x="total_revenue",
        y="revenue_volatility",
        color="segment_name",
        size="transaction_count",
        hover_data=["Description", "Category", "abc_class"],
        title="Product Segment Map",
    )
    bottom_right.plotly_chart(segment_fig, width="stretch")

with action_tab:
    action_left, action_right = st.columns([1.3, 1])
    action_left.subheader("Recommended Actions")
    for _, row in data["recommendations"].iterrows():
        with action_left.container(border=True):
            st.markdown(f"**{row['priority']} | {row['theme']}**")
            st.write(row["recommendation"])
            st.caption(row["evidence"])

    action_right.subheader("Customer and Product Segments")
    customer_seg_fig = px.bar(
        data["customer_segment_summary"],
        x="rfm_segment",
        y="revenue",
        color="customers",
        title="Customer Segment Revenue",
    )
    action_right.plotly_chart(customer_seg_fig, width="stretch")
    product_seg_fig = px.pie(
        data["product_segment_summary"],
        names="segment_name",
        values="revenue",
        title="Revenue Mix by Product Segment",
    )
    action_right.plotly_chart(product_seg_fig, width="stretch")
