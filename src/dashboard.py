import streamlit as st
import pandas as pd
import plotly.express as px
import os

DATA_DIR = "data"
FORECAST_DIR = os.path.join(DATA_DIR, "forecast")

st.set_page_config(page_title="Brand Forecast Dashboard", layout="wide")
st.title("📊 Brand Forecasting Dashboard")

# ================= LOAD DATA =====================
sales_df = pd.read_csv(os.path.join(DATA_DIR, "processed", "processed_sales.csv"))

prophet_path = os.path.join(FORECAST_DIR, "prophet_forecast_results.csv")
lstm_path = os.path.join(FORECAST_DIR, "lstm_forecast_results.csv")
metrics_path = os.path.join(FORECAST_DIR, "prophet_metrics.csv")

prophet_df = pd.read_csv(prophet_path) if os.path.exists(prophet_path) else pd.DataFrame()
lstm_df = pd.read_csv(lstm_path) if os.path.exists(lstm_path) else pd.DataFrame()
metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()

# Convert date columns
sales_df["Date"] = pd.to_datetime(sales_df["Date"])
if not prophet_df.empty:
    prophet_df["Date"] = pd.to_datetime(prophet_df["Date"])
if not lstm_df.empty:
    lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])

# ================= SIDEBAR FILTERS ==================
st.sidebar.header("🔍 Filters")

brand_search = st.sidebar.text_input("Search Brand")
brands = sorted([b for b in sales_df["Brand"].unique() if brand_search.lower() in b.lower()])
selected_brand = st.sidebar.selectbox("Select Brand", brands)

min_date, max_date = sales_df["Date"].min(), sales_df["Date"].max()
date_range = st.sidebar.date_input("Select Date Range", [min_date.date(), max_date.date()])
date_range = pd.to_datetime(date_range)  # convert to Timestamp

model_choice = st.sidebar.multiselect(
    "Select Forecast Model",
    ["Prophet", "LSTM"],
    default=["Prophet", "LSTM"]
)

# If Prophet data is missing, warn and stop (dashboard still shows actuals)
if prophet_df.empty:
    st.warning("No Prophet forecast data found yet. Run the Airflow pipeline to generate forecasts.")
    st.stop()

# ================= TOP SUMMARY CARDS ==================
st.markdown("### Branch Overview")

# This month revenue (sum of Total_Sales for latest month)
this_month = sales_df["Date"].dt.to_period("M").max()
this_month_sales = sales_df[sales_df["Date"].dt.to_period("M") == this_month]["Total_Sales"].sum()

# Next month projected revenue from Prophet
next_month = prophet_df["Date"].dt.to_period("M").max()
next_month_sales = prophet_df[prophet_df["Date"].dt.to_period("M") == next_month]["Predicted_Sales"].sum()

# Fastest growing brand (by Growth from your growth_df)
last_actual_month = sales_df["Date"].dt.to_period("M").max()
last_actual = sales_df[sales_df["Date"].dt.to_period("M") == last_actual_month] \
    .groupby("Brand")["Total_Sales"].sum().reset_index().rename(columns={"Total_Sales": "Last_Month_Sales"})
next_month_forecast = prophet_df[prophet_df["Date"].dt.to_period("M") == next_month]
growth_df_all = next_month_forecast.groupby("Brand")["Predicted_Sales"].sum().reset_index()
growth_df_all = growth_df_all.merge(last_actual, on="Brand", how="left")
growth_df_all["Growth"] = growth_df_all["Predicted_Sales"] - growth_df_all["Last_Month_Sales"]

if not growth_df_all.empty:
    fastest_row = growth_df_all.sort_values("Growth", ascending=False).iloc[0]
    fastest_brand = fastest_row["Brand"]
    fastest_growth = fastest_row["Growth"]
else:
    fastest_brand = "N/A"
    fastest_growth = 0.0

growing_brands_count = (growth_df_all["Growth"] > 0).sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Est. Revenue (This Month)", f"{this_month_sales:,.0f}")
with c2:
    st.metric("Projected Revenue (Next Month)", f"{next_month_sales:,.0f}")
with c3:
    st.metric("Fastest Growing Brand", fastest_brand, f"{fastest_growth:,.0f} Δ")
with c4:
    st.metric("Brands with Positive Growth", int(growing_brands_count))

# ================= CATEGORY LEADERS ==================
st.markdown("### Category Leaders (Best Sellers)")

# total sales per Brand & Category over last actual month
cat_scope = sales_df[sales_df["Date"].dt.to_period("M") == last_actual_month]

if "Category" in cat_scope.columns:
    cat_group = cat_scope.groupby(["Category", "Brand"])["Total_Sales"].sum().reset_index()

    def top_n_for_category(cat, n=3):
        sub = cat_group[cat_group["Category"] == cat].sort_values("Total_Sales", ascending=False).head(n)
        return sub[["Brand", "Total_Sales"]]

    cols_cat = st.columns(4)
    categories = ["Dairy", "Beverages", "Snacks", "Personal Care"]

    for col, cat in zip(cols_cat, categories):
        with col:
            st.markdown(f"**{cat.upper()}**")
            top_cat = top_n_for_category(cat, 3)
            if top_cat.empty:
                st.caption("No data")
            else:
                for _, row in top_cat.iterrows():
                    st.write(f"{row['Brand']}: {row['Total_Sales']:,.0f}")
else:
    st.info("Category column not available in processed data. Re-run preprocessing if you need category leaders.")

# ================= FILTER DATA FOR SELECTED BRAND ==================
actual = sales_df[(sales_df["Brand"] == selected_brand) &
                  (sales_df["Date"].between(date_range[0], date_range[1]))]

prophet = prophet_df[prophet_df["Brand"] == selected_brand]
lstm = lstm_df[lstm_df["Brand"] == selected_brand] if not lstm_df.empty else pd.DataFrame()

# ================= ACTUAL vs FORECAST ==================
st.markdown("### Actual vs Forecast")

fig = px.line(actual, x="Date", y="Total_Sales",
              title=f"Actual vs Forecast — {selected_brand}")

if "Prophet" in model_choice and not prophet.empty:
    fig.add_scatter(x=prophet["Date"], y=prophet["Predicted_Sales"],
                    name="Prophet Forecast")

if "LSTM" in model_choice and not lstm.empty:
    fig.add_scatter(x=lstm["Date"], y=lstm["Predicted_Sales"],
                    name="LSTM Forecast")

st.plotly_chart(fig, use_container_width=True)

# ================= MODEL METRICS ==================
st.markdown("### Model Comparison Metrics")
if not metrics.empty:
    st.dataframe(metrics)
else:
    st.caption("No metrics file found yet.")

# ================= TOP PERFORMING BRANDS NEXT MONTH ==================
st.markdown("### 🏆 Top Performing Brands — Next Month Forecast")

growth_df = growth_df_all.copy()
growth_df["Growth_%"] = (growth_df["Growth"] / growth_df["Last_Month_Sales"]) * 100
growth_df["Trend"] = growth_df["Growth"].apply(lambda x: "📈" if x > 0 else "📉")
growth_df = growth_df.sort_values("Growth", ascending=False)

st.dataframe(growth_df[["Brand", "Predicted_Sales",
                        "Last_Month_Sales", "Growth",
                        "Growth_%", "Trend"]].head(10))

# ================= BAR CHART: Brand Growth Ranking ==================
st.markdown("### 📊 Brand Forecast Growth Ranking")

fig2 = px.bar(
    growth_df.head(10),
    x="Brand",
    y="Growth",
    text="Trend",
    title="Top Growing Brands — Forecast Next Month"
)
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)
