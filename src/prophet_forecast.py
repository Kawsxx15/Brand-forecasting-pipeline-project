import os

# Ensure writable temp directory for CmdStanpy/Prophet in Docker
os.makedirs("/opt/airflow/tmp", exist_ok=True)
os.environ["TMPDIR"] = "/opt/airflow/tmp"

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Configuration
DATA_DIR = "data"
FORECAST_DIR = os.path.join(DATA_DIR, "forecast")
os.makedirs(FORECAST_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(DATA_DIR, "processed", "processed_sales.csv")
FORECAST_FILE = os.path.join(FORECAST_DIR, "prophet_forecast_results.csv")
METRICS_FILE = os.path.join(FORECAST_DIR, "prophet_metrics.csv")

# Additional regressors you want to include if available
extra_regressors = [
    "Competitor_Price", "Category_Trend_Index", "Customer_Growth_Rate",
    "Customer_Retention_Rate", "Stock_Level", "Supply_Delay_Days",
    "Inflation_Rate", "Weather_Score", "Promotion",
    "Discount_Percentage", "Is_Holiday",
]

def forecast_with_prophet():
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError(f"{PROCESSED_FILE} not found. Run preprocessing first.")

    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], dayfirst=True)

    forecast_results, metrics = [], []

    for brand, brand_df in df.groupby("Brand"):
        brand_df = brand_df.sort_values("Date")

        # Aggregate per date to ensure unique date-brand pairs
        agg_dict = {"Total_Sales": "sum", "Online_Popularity": "mean"}
        for reg in extra_regressors:
            if reg in brand_df.columns:
                agg_dict[reg] = "mean"
        brand_df = brand_df.groupby("Date", as_index=False).agg(agg_dict)

        used_regressors = [col for col in extra_regressors if col in brand_df.columns]

        prophet_cols = ["Date", "Total_Sales", "Online_Popularity"] + used_regressors
        prophet_df = brand_df[prophet_cols].rename(
            columns={"Date": "ds", "Total_Sales": "y", "Online_Popularity": "Trend_Score"}
        )

        # Ensure ds is datetime
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        # ====== CLEANING (RELAXED) ======
        initial_length = len(prophet_df)

        # Replace infs with NaN
        prophet_df = prophet_df.replace([np.inf, -np.inf], np.nan)

        # Require only y (target) to be valid and numeric
        prophet_df = prophet_df.dropna(subset=["y"])
        prophet_df = prophet_df[pd.to_numeric(prophet_df["y"], errors="coerce").notnull()]

        dropped_rows = initial_length - len(prophet_df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with invalid y for {brand}")
        print(f"After cleaning, {brand} has {len(prophet_df)} rows")
        # =================================

        print("Data types in train set:")
        print(prophet_df.dtypes)
        print("Describe target (y):")
        print(prophet_df["y"].describe())
        print("Unique sample dates:", prophet_df["ds"].unique()[:10])

        if len(prophet_df) < 3:
            print(f"⚠️ Skipping {brand} — insufficient data ({len(prophet_df)} rows after cleaning).")
            continue

        train_size = int(len(prophet_df) * 0.8)
        train, test = prophet_df[:train_size], prophet_df[train_size:]

        print(f"Trying to fit brand: {brand}")
        print("Train preview:", train.head(5))
        print("NaNs per column in train:", train.isnull().sum())
        print("Number of train rows:", len(train))
        if len(train) < 2 or train.isnull().any().any():
            print(f"⚠️ Skipping {brand}: Not enough valid train data or remaining NaNs.")
            continue
        if test.empty:
            print(f"⚠️ Skipping {brand}: Test set is empty after cleaning.")
            continue

        # Exclude constant regressors
        regressors_to_use = []
        if "Trend_Score" in train.columns and train["Trend_Score"].nunique() > 1:
            regressors_to_use.append("Trend_Score")
        else:
            print(f"Regressor Trend_Score is constant for {brand}; skipping.")
        for reg in used_regressors:
            if reg in train.columns and train[reg].nunique() > 1:
                regressors_to_use.append(reg)
            else:
                print(f"Regressor {reg} is constant for {brand}; skipping.")

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        for reg in regressors_to_use:
            model.add_regressor(reg)

        train_predictors = ["ds", "y"] + regressors_to_use
        test_predictors = ["ds"] + regressors_to_use

        model.fit(train[train_predictors])
        forecast_test = model.predict(test[test_predictors])
        rmse = np.sqrt(mean_squared_error(test["y"], forecast_test["yhat"]))
        mape = mean_absolute_percentage_error(test["y"], forecast_test["yhat"]) * 100
        metrics.append({"Brand": brand, "RMSE": rmse, "MAPE (%)": mape})

        future = model.make_future_dataframe(periods=30)
        # Ensure both sides have datetime ds for merge
        future["ds"] = pd.to_datetime(future["ds"])
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        future = future.merge(prophet_df[["ds"] + regressors_to_use], on="ds", how="left")
        for reg in regressors_to_use:
            future[reg].fillna(method="ffill", inplace=True)
        forecast = model.predict(future)
        forecast["Brand"] = brand
        forecast_results.append(forecast[["ds", "Brand", "yhat", "yhat_lower", "yhat_upper"]])

    if not forecast_results:
        print("⚠️ No forecasts generated. Check data availability per brand.")
        return

    all_forecasts = pd.concat(forecast_results, ignore_index=True)
    all_forecasts.rename(columns={"ds": "Date", "yhat": "Predicted_Sales"}, inplace=True)
    all_forecasts.to_csv(FORECAST_FILE, index=False)
    pd.DataFrame(metrics).to_csv(METRICS_FILE, index=False)

    print(f"✅ Prophet forecast & metrics saved in {FORECAST_DIR}")
    print(pd.DataFrame(metrics).head())
    return metrics

if __name__ == "__main__":
    forecast_with_prophet()
