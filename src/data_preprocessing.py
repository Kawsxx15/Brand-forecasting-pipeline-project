import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# =======================================
# CONFIGURATION
# =======================================
DATA_DIR = "data"
RAW_DATA_FILE = os.path.join(DATA_DIR, "raw", "brand_sales_dataset.csv")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(PROCESSED_DIR, "processed_sales.csv")

def preprocess_data():
    """Preprocess raw data: fill missing, encode categorical, feature engineer, aggregate."""

    if not os.path.exists(RAW_DATA_FILE):
        raise FileNotFoundError(f"{RAW_DATA_FILE} not found. Generate raw data first.")

    df = pd.read_csv(RAW_DATA_FILE)

    print(f"Loaded raw data with {len(df)} rows.")

    # Handle missing values
    df["Online_Popularity"] = df["Online_Popularity"].fillna(df["Online_Popularity"].median())
    df["Total_Sales"] = df["Total_Sales"].fillna(df["Total_Sales"].median())
    df["Quantity_Sold"] = df["Quantity_Sold"].fillna(df["Quantity_Sold"].median())

    numeric_fill_cols = [
        "Competitor_Price", "Category_Trend_Index", "Customer_Growth_Rate",
        "Customer_Retention_Rate", "Stock_Level", "Supply_Delay_Days",
        "Inflation_Rate", "Weather_Score"
    ]
    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "Is_Holiday" in df.columns:
        df["Is_Holiday"] = df["Is_Holiday"].fillna(0)

    # Convert Date to datetime and sort data
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(by=["Brand", "Date"])

    # Encode categorical columns
    label_encoders = {}
    for col in ["Category", "Brand", "Region"]:
        le = LabelEncoder()
        df[col + "_Encoded"] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature engineering: lags and rolling mean for Total_Sales per brand
    df["Sales_Lag_1"] = df.groupby("Brand")["Total_Sales"].shift(1)
    df["Sales_MA_3"] = df.groupby("Brand")["Total_Sales"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Use transform with ffill and bfill to avoid index issues
    df["Sales_Lag_1"] = df.groupby("Brand")["Sales_Lag_1"].transform(lambda x: x.ffill().bfill())

    # For any remaining NaNs in rolling mean, fill with Total_Sales
    df["Sales_MA_3"].fillna(df["Total_Sales"], inplace=True)

    # Normalize selected numeric columns except Total_Sales
    numeric_cols_to_normalize = [
        "Quantity_Sold", "Online_Popularity",
        "Sales_Lag_1", "Sales_MA_3",
        "Competitor_Price", "Category_Trend_Index", "Customer_Growth_Rate",
        "Customer_Retention_Rate", "Stock_Level", "Supply_Delay_Days",
        "Inflation_Rate", "Weather_Score"
    ]
    existing_numeric_cols = [col for col in numeric_cols_to_normalize if col in df.columns]
    df[existing_numeric_cols] = df[existing_numeric_cols].apply(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # Aggregate by Category, Brand and Date to get unique daily summary rows for forecasting
    agg_dict = {
        "Total_Sales": "sum",
        "Quantity_Sold": "sum",
        "Online_Popularity": "mean",
        "Competitor_Price": "mean",
        "Category_Trend_Index": "mean",
        "Customer_Growth_Rate": "mean",
        "Customer_Retention_Rate": "mean",
        "Stock_Level": "mean",
        "Supply_Delay_Days": "mean",
        "Inflation_Rate": "mean",
        "Weather_Score": "mean",
        "Promotion": "mean",
        "Discount_Percentage": "mean",
        "Is_Holiday": "mean",
        "Sales_Lag_1": "mean",
        "Sales_MA_3": "mean",
        "Category_Encoded": "first",
        "Brand_Encoded": "first",
        "Region_Encoded": "first",
        "Category": "first"   # ✅ keep human-readable Category
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    # ✅ group by Category as well so it's preserved
    df_agg = df.groupby(["Category", "Brand", "Date"], as_index=False).agg(agg_dict)

    # Save the processed aggregated data for forecasting
    df_agg.to_csv(PROCESSED_FILE, index=False)
    print(f"Processed data saved with {len(df_agg)} rows to {PROCESSED_FILE}")

    return df_agg

if __name__ == "__main__":
    preprocess_data()
