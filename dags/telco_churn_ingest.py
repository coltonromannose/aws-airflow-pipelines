from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def placeholder_task():
    print("This is a placeholder DAG. Real logic coming soon.")

with DAG(
    dag_id="telco_churn_ingest",
    start_date=datetime(2023, 9, 24),
    schedule="@daily",
    catchup=False,
    tags=["telco_churn", "placeholder"]
):

    start = PythonOperator(
        task_id="placeholder",
        python_callable=placeholder_task
    )
