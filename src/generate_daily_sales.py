import pandas as pd
import random
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import numpy as np
import os
import time
import requests  # optional, for clearer exception messages

# =======================================
# CONFIGURATION
# =======================================
DATA_DIR = "/opt/airflow/data"    # Absolute path for Docker volume mount
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

RAW_FILE = os.path.join(RAW_DIR, "brand_sales_dataset.csv")

DAYS_TO_GENERATE = 1
HISTORICAL_DAYS = 90   # Ensure multi-day historical coverage

REGIONS = ["Maharashtra", "Tamil Nadu", "Karnataka", "Delhi", "Gujarat"]

categories = {
    "Beverages": ["Coca-Cola", "Pepsi", "Fanta", "Thums Up", "Sprite", "Maaza"],
    "Dairy": ["Amul", "Mother Dairy", "Britannia", "Parag", "Gowardhan"],
    "Snacks": ["Lays", "Haldiram", "Bingo", "Kurkure", "Balaji"],
    "Personal Care": ["Dove", "Pond's", "Lifebuoy", "Patanjali", "Himalaya"]
}

# Cache dictionary to store trend scores per brand
trend_cache = {}

def get_trend_score(brand: str) -> float:
    if brand in trend_cache:
        return trend_cache[brand]

    # Initialize pytrends INSIDE function only (no DAG import timeout)
    pytrends = TrendReq(hl="en-US", tz=330, timeout=(10, 25))

    try:
        pytrends.build_payload([brand], timeframe="now 7-d", geo="IN")
        data = pytrends.interest_over_time()
        if not data.empty:
            score = float(data[brand].iloc[-1])
        else:
            score = random.uniform(30, 80)
    except (requests.exceptions.RequestException, Exception) as e:
        print(f"⚠️ Trend fetch failed for {brand}: {e}. Using random score.")
        score = random.uniform(30, 80)

    trend_cache[brand] = score
    return score

def generate_brand_sales(date: str):
    rows = []
    dt = datetime.strptime(date, "%Y-%m-%d")

    holidays = ["01-26", "08-15", "10-02", "12-25"]

    # Create list of all brands with their categories
    all_brands = []
    for category, brand_list in categories.items():
        for brand in brand_list:
            all_brands.append((category, brand))

    repeats = 20  # number of sales entries per brand per day
    all_choices = all_brands * repeats
    random.shuffle(all_choices)

    for category, brand in all_choices:
        trend_score = get_trend_score(brand)  # Cached call per brand
        base_sales = random.randint(1000, 10000)
        seasonal_multiplier = 1 + 0.1 * np.sin(dt.timetuple().tm_yday / 365 * 2 * np.pi)
        trend_multiplier = 0.8 + (trend_score / 100) * 0.4
        total_sales = base_sales * seasonal_multiplier * trend_multiplier * random.uniform(0.8, 1.2)
        quantity_sold = int(total_sales / random.uniform(20, 250))
        region = random.choice(REGIONS)
        promotion = random.choice([0, 1]) if random.random() < 0.15 else 0
        discount_pct = random.choice([0, 5, 10, 15, 20]) if promotion else 0
        competitor_price = random.uniform(50, 500)
        category_trend_index = round(random.uniform(0.8, 1.3), 2)
        customer_growth_rate = round(random.uniform(-0.02, 0.05), 4)
        retention_rate = round(random.uniform(0.65, 0.95), 2)
        stock_level = random.randint(500, 5000)
        supply_delay = random.choice([0, 1, 2, 3])
        inflation_rate = round(random.uniform(4.5, 9.5), 2)
        weather_score = random.uniform(60, 100) if category == "Beverages" else random.uniform(40, 90)
        is_holiday = 1 if dt.strftime("%m-%d") in holidays else 0

        rows.append({
            "Date": date,
            "Category": category,
            "Brand": brand,
            "Region": region,
            "Promotion": promotion,
            "Total_Sales": round(total_sales, 2),
            "Quantity_Sold": quantity_sold,
            "Online_Popularity": round(trend_score, 2),
            "Discount_Percentage": discount_pct,
            "Month": dt.month,
            "Week": dt.isocalendar().week,
            "Is_Holiday": is_holiday,
            "Competitor_Price": round(competitor_price, 2),
            "Category_Trend_Index": category_trend_index,
            "Customer_Growth_Rate": customer_growth_rate,
            "Customer_Retention_Rate": retention_rate,
            "Stock_Level": stock_level,
            "Supply_Delay_Days": supply_delay,
            "Inflation_Rate": inflation_rate,
            "Weather_Score": round(weather_score, 2)
        })

    return rows

def generate_historical_data():
    start_date = datetime.now() - timedelta(days=HISTORICAL_DAYS)
    all_rows = []
    for i in range(HISTORICAL_DAYS):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        print(f"Generating data for {date}")  # Progress log
        rows = generate_brand_sales(date)
        all_rows.extend(rows)
        time.sleep(0.1)  # Reduced sleep to speed up (adjust if necessary)
    df = pd.DataFrame(all_rows)
    df.to_csv(RAW_FILE, mode="w", header=True, index=False)
    print(f"✅ Generated {HISTORICAL_DAYS} days of historical data ({len(df)} rows)")

def validate_data():
    df = pd.read_csv(RAW_FILE, parse_dates=["Date"])
    unique_dates_total = df["Date"].nunique()
    unique_dates_per_brand = df.groupby("Brand")["Date"].nunique()
    print(f"Unique dates in dataset: {unique_dates_total}")
    print("Unique dates per brand:")
    print(unique_dates_per_brand)

if __name__ == "__main__":
    # Generate full historical data (multi-day)
    generate_historical_data()
    # Validate dates coverage before forecasting
    validate_data()
