import pandas as pd
from pytrends.request import TrendReq
import os
import random
from datetime import datetime

# =======================================
# CONFIGURATION
# =======================================
DATA_DIR = os.path.join("data", "raw")
DATA_FILE = os.path.join(DATA_DIR, "brand_sales_dataset.csv")

# =======================================
# GOOGLE TRENDS HELPER
# =======================================

def fetch_trend_score(brand: str) -> float:
    """Fetch the current Google Trends score for a brand."""
    # Initialize pytrends INSIDE the function so Airflow import does not hit network
    pytrends = TrendReq(hl="en-US", tz=330, timeout=(10, 25))
    try:
        pytrends.build_payload([brand], timeframe="now 7-d", geo="IN")
        data = pytrends.interest_over_time()
        if not data.empty:
            return float(data[brand].iloc[-1])
        else:
            return random.uniform(30, 80)
    except Exception as e:
        # Handle timeouts or rate limits gracefully
        print(f"⚠️ Trend fetch failed for {brand}: {e}. Using random score.")
        return random.uniform(30, 80)

# ✅ Function Airflow expects
def fetch_google_trends():
    """
    Reads the existing dataset and updates Online_Popularity
    scores for all brands for the most recent date.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run generate_sales_data first.")

    df = pd.read_csv(DATA_FILE)
    latest_date = df["Date"].max()
    print(f"📊 Updating trend scores for date: {latest_date}")

    latest_df = df[df["Date"] == latest_date].copy()

    updated_scores = []
    for brand in latest_df["Brand"].unique():
        trend_score = fetch_trend_score(brand)
        updated_scores.append({"Brand": brand, "Online_Popularity": trend_score})
        print(f"🔹 {brand}: {trend_score:.2f}")

    scores_df = pd.DataFrame(updated_scores)
    df = df.drop(columns=["Online_Popularity"], errors="ignore")
    df = df.merge(scores_df, on="Brand", how="left")

    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Trend scores updated and saved to {DATA_FILE}")

    return df

# ✅ Backwards compatibility (old name still works)
def get_trend_score():
    return fetch_google_trends()

if __name__ == "__main__":
    fetch_google_trends()
