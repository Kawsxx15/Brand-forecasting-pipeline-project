# Brand Forecasting Pipeline (Prophet + LSTM + Airflow + Streamlit)

End‑to‑end brand sales forecasting pipeline using synthetic supermarket data, Facebook Prophet, LSTM (PyTorch), Apache Airflow for orchestration, and a Streamlit dashboard for visualization.

# Project Overview

This project simulates daily sales for FMCG brands across categories and regions, enriches them with Google Trends–style popularity scores and business features, and builds two forecasting models:

  * Prophet (with extra regressors)

  * LSTM sequence model (PyTorch)

The pipeline is orchestrated with Apache Airflow and the final outputs are:

  * Forecast CSVs under data/forecast/

  * Model metrics (RMSE, MAPE) for each brand

A Streamlit dashboard showing:

  * Actual vs Prophet/LSTM forecasts per brand

  * Top‑growing brands next month

  * Branch overview KPIs and category leaders

# Repository Structure

.
├─ docker/

│  ├─ Dockerfile.streamlit             # Streamlit dashboard image

│  ├─ requirements-dashboard.txt       # Lightweight dashboard deps

│  └─ requirements.txt                 # Airflow + ML deps (Prophet, LSTM, Airflow, etc.)

├─ dags/

│  └─ brand_forecast_dag.py            # Airflow DAG definition

├─ src/

│  ├─ generate_daily_sales.py          # Synthetic data generation (raw daily sales)

│  ├─ generate_trend_score.py          # Updates Online_Popularity using TrendReq

│  ├─ data_preprocessing.py            # Cleans, encodes, aggregates to processed_sales.csv

│  ├─ prophet_forecast.py              # Prophet training + 30‑day forecast + metrics

│  ├─ lstm_forecast.py                 # LSTM training + 30‑day forecast + metrics

│  ├─ visualization.py                 # Saves static plots / reports (optional)

│  ├─ telegram_alert.py                # Telegram notification helper (optional)

│  └─ dashboard.py                     # Streamlit dashboard app

├─ data/

│  ├─ raw/                             # brand_sales_dataset.csv (generated)

│  ├─ processed/                       # processed_sales.csv (from preprocessing)

│  └─ forecast/                        # prophet_*.csv and lstm_*.csv outputs

├─ logs/                               # Airflow task logs (not required in Git)

├─ docker-compose.yml                  # Brings up Airflow + Streamlit services

└─ README.md

You do not need to commit large generated CSVs and logs; see the Git section below.

# How to Run the Project

# 1. Prerequisites
   
  * Docker and Docker Compose installed

  * Git installed

  * (Optional) Telegram bot token and chat ID if you want alert messages

# 2. Clone the Repository
   
  git clone https://github.com/Kawsxx15/Brand-forecasting-pipeline-project.git

  cd <your-repo>

# 3. Start the Stack (Airflow + Streamlit)
   
* Build images and start services:

Open the cmd in the project root folder and run:

* docker compose build

* docker compose up -d

# Services:

* Airflow Webserver: http://localhost:8080

  * Default: user/password as configured in your docker‑compose (or Airflow docs).

* Streamlit Dashboard: http://localhost:8501

# 4. Run the Airflow Pipeline
   
  * Open Airflow UI at http://localhost:8080.

  * Enable the DAG brand_forecasting_pipeline.

  * Trigger the DAG manually (play ▶ button).


The DAG will execute the following tasks in order:

# Generate_historical_data

  * Uses generate_daily_sales.py to create multi‑day synthetic sales in

  * data/raw/brand_sales_dataset.csv.

# collect_trends

  * Uses generate_trend_score.py to update Online_Popularity based on Google Trends‑style scores.

# Preprocess_data

  * Uses data_preprocessing.py to:

  * Clean and normalize numeric features

  * Encode categorical variables

  * Engineer lags/rolling means

  * Aggregate to daily level

  * Outputs data/processed/processed_sales.csv.

# Forecast_prophet

  * Uses prophet_forecast.py to train a Prophet model per brand and forecast 30 days ahead.

   * Outputs:

        * data/forecast/prophet_forecast_results.csv
        
        * data/forecast/prophet_metrics.csv

# Forecast_lstm

* Uses lstm_forecast.py to train an LSTM per brand and forecast 30 days ahead.

   * Outputs:

       * data/forecast/lstm_forecast_results.csv
         
       * data/forecast/lstm_metrics.csv
         
       * generate_visualizations (optional)

# Notify_success

* Sends a Telegram message if configured.

Wait until all tasks turn green in the Airflow grid.

# 5.View the Streamlit Dashboard
   
 Open:

http://localhost:8501

# You will see:

# Branch Overview cards:

* Estimated revenue (this month)

* Projected revenue (next month, from Prophet)

* Fastest growing brand

* Count of brands with positive growth

# Category Leaders:

* Top 3 brands in Dairy, Beverages, Snacks, Personal Care (by last‑month sales)

# Actual vs Forecast line chart:

* Historical Total_Sales vs Prophet and/or LSTM forecasts for the selected brand and date range.

# Model Comparison Metrics:

* RMSE and MAPE per brand (from Prophet metrics CSV).

# Top Performing Brands — Next Month Forecast:

* Table with predicted sales, last‑month sales, absolute growth, growth%, and trend icon.

# Brand Forecast Growth Ranking:

* Bar chart of top‑growing brands (by absolute growth next month).

# What to Commit vs Ignore

#Recommended to commit:

docker/requirements.txt and docker/requirements-dashboard.txt

docker-compose.yml

dags/brand_forecast_dag.py

All Python source files under src/

Any .env.example or configuration templates

README.md

# Recommended to ignore in .gitignore:

text
# Data

data/raw/

data/processed/

data/forecast/

# Airflow logs and temp

logs/

*/__pycache__/

*.log

*.tmp

.opt/airflow/tmp/

This avoids pushing large CSVs and logs while keeping the pipeline reproducible.

# Final Output Summary

After a successful run:

# Data artifacts:

* data/raw/brand_sales_dataset.csv – synthetic multi‑day sales with rich features.

* data/processed/processed_sales.csv – cleaned, encoded, aggregated data for modeling.

* data/forecast/prophet_forecast_results.csv – daily Prophet predictions (with bounds) for each brand.

* data/forecast/prophet_metrics.csv – Prophet RMSE/MAPE per brand.

* data/forecast/lstm_forecast_results.csv – daily LSTM predictions for each brand.

* data/forecast/lstm_metrics.csv – LSTM RMSE/MAPE per brand.

# Visualization:

* Interactive Streamlit dashboard at localhost:8501 for:

* Comparing Prophet vs LSTM

* Inspecting brand‑level trends

* Seeing next‑month growth leaders and category best‑sellers.
