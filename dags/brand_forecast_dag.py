from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys

# ==============================
# ✅ Auto-detect src folder (works in Docker & local)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# ==============================
# ✅ Import modules from src/
# ==============================
from data_preprocessing import preprocess_data
from get_trend_score import get_trend_score
from generate_daily_sales import generate_historical_data  # Import existing function only
from prophet_forecast import forecast_with_prophet
from lstm_forecast import forecast_with_lstm
from visualization import generate_visualizations
from telegram_alert import send_telegram_message

# ==============================
# ✅ DAG CONFIG
# ==============================
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "brand_forecasting_pipeline",
    default_args=default_args,
    description="End-to-end brand forecasting pipeline (Prophet + LSTM)",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# ==============================
# ✅ Task Wrappers
# ==============================
def task_generate_historical():
    generate_historical_data()

def task_collect_trends():
    get_trend_score()

def task_preprocess():
    preprocess_data()

def task_prophet():
    forecast_with_prophet()

def task_lstm():
    forecast_with_lstm()

def task_visualize():
    generate_visualizations()

def task_notify_success():
    send_telegram_message("✅ DAG completed successfully! Forecast report generated.")

def task_notify_failure(context):
    send_telegram_message("❌ DAG Failed! Check Airflow logs.")

# ==============================
# ✅ Operators
# ==============================
t0 = PythonOperator(
    task_id="generate_historical_data",
    python_callable=task_generate_historical,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t2 = PythonOperator(
    task_id="collect_trends",
    python_callable=task_collect_trends,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t3 = PythonOperator(
    task_id="preprocess_data",
    python_callable=task_preprocess,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t4 = PythonOperator(
    task_id="forecast_prophet",
    python_callable=task_prophet,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t5 = PythonOperator(
    task_id="forecast_lstm",
    python_callable=task_lstm,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t6 = PythonOperator(
    task_id="generate_visualizations",
    python_callable=task_visualize,
    dag=dag,
    on_failure_callback=task_notify_failure,
)

t7 = PythonOperator(
    task_id="notify_success",
    python_callable=task_notify_success,
    dag=dag,
)

# ==============================
# ✅ Workflow dependencies
# ==============================
t0 >> t2 >> t3 >> [t4, t5] >> t6 >> t7
