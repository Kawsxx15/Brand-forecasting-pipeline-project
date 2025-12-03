import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from fpdf import FPDF

# =======================================
# CONFIGURATION
# =======================================
DATA_DIR = "data"
FORECAST_DIR = os.path.join(DATA_DIR, "forecast")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("Set2")

def generate_visualizations():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def safe_load_csv(file_path, name):
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"⚠️ {name} file missing: {file_path}")
            return pd.DataFrame()

    prophet_df = safe_load_csv(os.path.join(FORECAST_DIR, "prophet_forecast_results.csv"), "Prophet Forecast")
    lstm_df = safe_load_csv(os.path.join(FORECAST_DIR, "lstm_forecast_results.csv"), "LSTM Forecast")
    prophet_metrics = safe_load_csv(os.path.join(FORECAST_DIR, "prophet_metrics.csv"), "Prophet Metrics")
    lstm_metrics = safe_load_csv(os.path.join(FORECAST_DIR, "lstm_metrics.csv"), "LSTM Metrics")
    processed_sales = safe_load_csv(os.path.join(DATA_DIR, "processed_sales.csv"), "Processed Sales")

    if processed_sales.empty:
        print("⚠️ Processed sales data is empty. Skipping visualization.")
        return
    if "Brand" not in processed_sales.columns:
        print("⚠️ 'Brand' column missing in processed sales data. Skipping visualization.")
        return

    if not prophet_df.empty:
        prophet_df["Date"] = pd.to_datetime(prophet_df["Date"])
    if not lstm_df.empty:
        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"])

    # BRAND FORECAST COMPARISONS
    if not prophet_df.empty and not lstm_df.empty:
        for brand in prophet_df["Brand"].unique():
            p_df = prophet_df[prophet_df["Brand"] == brand]
            l_df = lstm_df[lstm_df["Brand"] == brand]

            plt.figure(figsize=(10, 5))
            plt.plot(p_df["Date"], p_df["Predicted_Sales"], label="Prophet", linewidth=2)
            plt.plot(l_df["Date"], l_df["Predicted_Sales"], label="LSTM", linewidth=2)
            plt.title(f"Forecast Comparison - {brand}")
            plt.xlabel("Date")
            plt.ylabel("Predicted Sales")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{brand}_comparison.png"), dpi=300)
            plt.close()

    # TOP 5 BRANDS - Prophet vs LSTM
    if not prophet_df.empty and not lstm_df.empty:
        top5_p = prophet_df.groupby("Brand")["Predicted_Sales"].mean().nlargest(5).reset_index()
        top5_l = lstm_df.groupby("Brand")["Predicted_Sales"].mean().nlargest(5).reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(data=top5_p, x="Brand", y="Predicted_Sales", color="skyblue", label="Prophet")
        sns.barplot(data=top5_l, x="Brand", y="Predicted_Sales", color="salmon", alpha=0.7, label="LSTM")
        plt.title("Top 5 Predicted Performing Brands - Prophet vs LSTM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "top5_brands_comparison.png"), dpi=300)
        plt.close()

    # METRICS COMPARISON
    if not prophet_metrics.empty and not lstm_metrics.empty:
        metrics = prophet_metrics.merge(lstm_metrics, on="Brand", suffixes=("_Prophet", "_LSTM"))

        # RMSE
        plt.figure(figsize=(8, 5))
        sns.barplot(data=metrics, x="Brand", y="RMSE_Prophet", color="skyblue", label="Prophet RMSE")
        sns.barplot(data=metrics, x="Brand", y="RMSE_LSTM", color="salmon", alpha=0.7, label="LSTM RMSE")
        plt.title("RMSE Comparison - Prophet vs LSTM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "rmse_comparison.png"), dpi=300)
        plt.close()

        # MAPE
        plt.figure(figsize=(8, 5))
        sns.barplot(data=metrics, x="Brand", y="MAPE (%)_Prophet", color="skyblue", label="Prophet MAPE")
        sns.barplot(data=metrics, x="Brand", y="MAPE (%)_LSTM", color="salmon", alpha=0.7, label="LSTM MAPE")
        plt.title("MAPE (%) Comparison - Prophet vs LSTM")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "mape_comparison.png"), dpi=300)
        plt.close()

    # COMBINE FORECAST RESULTS
    if not prophet_df.empty and not lstm_df.empty:
        next_month_p = prophet_df.groupby("Brand")["Predicted_Sales"].mean()
        next_month_l = lstm_df.groupby("Brand")["Predicted_Sales"].mean()

        combined_scores = (next_month_p + next_month_l) / 2
        best_brand = combined_scores.idxmax()
        best_value = combined_scores.max()

        trending_brand = processed_sales.groupby("Brand")["Online_Popularity"].mean().idxmax()

        summary_text = (
            f"Forecast Summary ({timestamp})\n\n"
            f"Brand expected to perform best next month: {best_brand}\n"
            f"Average Forecasted Sales Score: {round(best_value, 2)}\n\n"
            f"Most Trending Brand Right Now (Social buzz): {trending_brand}"
        )

        with open(os.path.join(FORECAST_DIR, "brand_forecast_summary.txt"), "w") as f:
            f.write(summary_text)

        print("\n✅ Final Forecast Decision")
        print(summary_text)

    # PDF REPORT
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Brand Forecasting Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10,
        "This report compares forecasts from Prophet and a Deep 2-Layer LSTM model.\n"
        "It contains accuracy metrics (RMSE, MAPE) and forecast trend visuals."
    )

    for file in sorted(os.listdir(PLOTS_DIR)):
        if file.endswith(".png"):
            pdf.add_page()
            pdf.image(os.path.join(PLOTS_DIR, file), x=10, y=25, w=180)

    pdf.output(os.path.join(FORECAST_DIR, "forecast_report.pdf"))
    print("📘 Forecast visualization PDF generated.")

    # SEND TO TELEGRAM (added without modifying logic)
    from telegram_alert import send_telegram_message, send_telegram_document

    summary_file = os.path.join(FORECAST_DIR, "brand_forecast_summary.txt")
    report_pdf = os.path.join(FORECAST_DIR, "forecast_report.pdf")

    send_telegram_message("✅ Forecast pipeline completed! Sending report…")

    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            send_telegram_message(f.read())

    send_telegram_document(report_pdf, caption="📊 Brand Forecast Report Ready ✅")
