from __future__ import annotations

import pandas as pd


def build_recommendations(
    kpis: dict[str, float | int | str],
    future_forecast: pd.DataFrame,
    recent_daily_sales: pd.DataFrame,
    anomalies: pd.DataFrame,
    product_segments: pd.DataFrame,
    category_performance: pd.DataFrame,
    customer_segments: pd.DataFrame,
    country_performance: pd.DataFrame,
) -> pd.DataFrame:
    recommendations: list[dict[str, str]] = []

    recent_30_day_revenue = recent_daily_sales.tail(30)["daily_revenue"].sum()
    next_month_forecast = future_forecast["predicted_revenue"].sum()
    projected_growth = ((next_month_forecast - recent_30_day_revenue) / max(recent_30_day_revenue, 1)) * 100

    if projected_growth > 8:
        recommendations.append(
            {
                "priority": "High",
                "theme": "Inventory",
                "recommendation": "Raise inventory cover for the highest-revenue A-class products before the next planning cycle.",
                "evidence": f"The next 30-day forecast is {projected_growth:.1f}% above the latest 30-day actual revenue.",
            }
        )

    repeated_drops = anomalies[anomalies["anomaly_type"] == "Drop"]
    if len(repeated_drops) >= 3:
        recommendations.append(
            {
                "priority": "High",
                "theme": "Operations",
                "recommendation": "Audit fulfilment and stock availability for abnormal drop days and top affected regions.",
                "evidence": f"{len(repeated_drops)} statistically abnormal low-sales days were detected.",
            }
        )

    low_margin_category = category_performance.sort_values(["profit_margin_pct", "revenue"], ascending=[True, False]).iloc[0]
    recommendations.append(
        {
            "priority": "Medium",
            "theme": "Pricing",
            "recommendation": f"Review discounting and supplier cost structure for {low_margin_category['Category']} to lift margin quality.",
            "evidence": f"{low_margin_category['Category']} delivers only {low_margin_category['profit_margin_pct']:.1f}% margin.",
        }
    )

    at_risk = product_segments[product_segments["segment_name"] == "At-risk products"].sort_values("total_revenue").head(3)
    if not at_risk.empty:
        recommendations.append(
            {
                "priority": "Medium",
                "theme": "Portfolio",
                "recommendation": "Rationalise low-performing SKUs through bundling, promotion, or discontinuation review.",
                "evidence": f"Representative at-risk products include {', '.join(at_risk['Description'].tolist())}.",
            }
        )

    at_risk_customers = customer_segments[customer_segments["rfm_segment"] == "At risk"]
    if not at_risk_customers.empty:
        recommendations.append(
            {
                "priority": "Medium",
                "theme": "Retention",
                "recommendation": "Launch a targeted win-back campaign for dormant high-value customers.",
                "evidence": f"{int(at_risk_customers['customers'].sum())} customers fall into the at-risk RFM segment.",
            }
        )

    top_country = country_performance.sort_values("revenue", ascending=False).iloc[0]
    recommendations.append(
        {
            "priority": "Low",
            "theme": "Expansion",
            "recommendation": f"Use {top_country['Country']} as the benchmark market for assortment and service-level planning.",
            "evidence": f"It is the largest market by revenue at ${top_country['revenue']:,.0f}.",
        }
    )

    return pd.DataFrame(recommendations)
