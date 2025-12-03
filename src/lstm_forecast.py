import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os

# =======================================
# CONFIG
# =======================================
DATA_DIR = "data"
FORECAST_DIR = os.path.join(DATA_DIR, "forecast")
os.makedirs(FORECAST_DIR, exist_ok=True)

PROCESSED_FILE = os.path.join(DATA_DIR, "processed", "processed_sales.csv")
FORECAST_FILE = os.path.join(FORECAST_DIR, "lstm_forecast_results.csv")
METRICS_FILE = os.path.join(FORECAST_DIR, "lstm_metrics.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================================
# LSTM MODEL DEFINITION
# =======================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # output shape: (batch, seq_len, hidden_dim)
        return self.fc(out[:, -1, :])    # use output at last time step

# =======================================
# MAIN FORECAST FUNCTION
# =======================================
def run_lstm_forecast():
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError(f"{PROCESSED_FILE} not found. Did preprocessing run?")

    df = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"], dayfirst=True)

    extra_regressors = [
        "Competitor_Price", "Category_Trend_Index", "Customer_Growth_Rate",
        "Customer_Retention_Rate", "Stock_Level", "Supply_Delay_Days",
        "Inflation_Rate", "Weather_Score", "Is_Holiday"
    ]

    forecasts, metrics = [], []
    seq_length = 7  # Use previous 7 days to predict next day

    for brand, brand_df in df.groupby("Brand"):
        brand_df = brand_df.sort_values("Date")

        # ✅ FIX: Ensure Date column is properly converted to datetime
        brand_df["Date"] = pd.to_datetime(brand_df["Date"])

        # Aggregate multiple entries per date into a single row
        agg_dict = {
            "Total_Sales": "sum",
            "Online_Popularity": "mean"
        }
        for reg in extra_regressors:
            if reg in brand_df.columns:
                agg_dict[reg] = "mean"
        brand_df = brand_df.groupby("Date", as_index=False).agg(agg_dict)

        # Prepare features and verify their existence
        features = ["Total_Sales", "Online_Popularity"] + [reg for reg in extra_regressors if reg in brand_df.columns]

        if len(brand_df) < seq_length + 1:
            print(f"Skipping {brand}: insufficient rows ({len(brand_df)}) to form sequences.")
            continue

        feature_array = brand_df[features].values

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(feature_array)

        X, y = [], []
        for i in range(len(scaled) - seq_length):
            X.append(scaled[i:i+seq_length])
            y.append(scaled[i + seq_length, 0])  # predict scaled Total_Sales

        X, y = np.array(X), np.array(y)

        if len(X) < 20:
            print(f"Skipping {brand}: not enough samples ({len(X)}) after sequence creation.")
            continue

        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

        model = LSTMModel(input_dim=len(features)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(80):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        preds = model(X_test).detach().cpu().numpy()
        y_true = y_test.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = mean_absolute_percentage_error(y_true, preds) * 100
        metrics.append({"Brand": brand, "RMSE": rmse, "MAPE (%)": mape})

        # Forecast 30 days ahead
        last_seq = torch.tensor(scaled[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)
        future_scaled = []
        for _ in range(30):
            pred = model(last_seq).item()
            last_features = scaled[-1].copy()
            next_step = np.array([pred] + list(last_features[1:]))  # update sales prediction only
            future_scaled.append(next_step)
            scaled = np.vstack([scaled, next_step])
            last_seq = torch.tensor(scaled[-seq_length:], dtype=torch.float32).unsqueeze(0).to(device)

        future_scaled = np.array(future_scaled)

        # Build full inverse transform data with repeated features for non-sales columns
        repeated_features = np.tile(brand_df[features[1:]].iloc[-1].values, (30, 1))
        full_inverse_data = np.hstack([future_scaled[:, [0]], repeated_features])

        future_sales = scaler.inverse_transform(full_inverse_data)[:, 0]

        # ✅ FIXED: Now Date column is proper datetime, so this works
        future_dates = pd.date_range(brand_df["Date"].max() + pd.Timedelta(days=1), periods=30)
        forecasts.append(pd.DataFrame({
            "Date": future_dates,
            "Brand": brand,
            "Predicted_Sales": future_sales
        }))

    if forecasts:
        pd.concat(forecasts, ignore_index=True).to_csv(FORECAST_FILE, index=False)
        pd.DataFrame(metrics).to_csv(METRICS_FILE, index=False)
        print(f"✅ LSTM forecast saved at: {FORECAST_FILE}")
        print(f"✅ Metrics saved at: {METRICS_FILE}")
        print(pd.DataFrame(metrics).head())
    else:
        print("⚠️ No forecasts generated. Please check preprocessed data or feature consistency.")

    return metrics

# Export for Airflow
def forecast_with_lstm():
    return run_lstm_forecast()

if __name__ == "__main__":
    run_lstm_forecast()
