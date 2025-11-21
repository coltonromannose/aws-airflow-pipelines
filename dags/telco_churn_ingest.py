import boto3
import logging
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


# --- Bucket Configuration: Already validated ---
MWAA_BUCKET = "airflow-server-setup-v-colton"
DATA_BUCKET = "airflow-telco-data-v-colton"

RAW_SOURCE_KEY = "data/Telco-Customer-Churn.csv"
RAW_DEST_KEY = "raw/Telco-Customer-Churn.csv"


def copy_raw_to_data_bucket(**context):
    """Copy the raw CSV from the MWAA bucket into the new data bucket."""
    s3 = boto3.client("s3")

    logging.info(
        f"Starting raw CSV copy:\n"
        f"  FROM: s3://{MWAA_BUCKET}/{RAW_SOURCE_KEY}\n"
        f"  TO:   s3://{DATA_BUCKET}/{RAW_DEST_KEY}"
    )

    s3.copy_object(
        Bucket=DATA_BUCKET,
        CopySource={"Bucket": MWAA_BUCKET, "Key": RAW_SOURCE_KEY},
        Key=RAW_DEST_KEY
    )

    logging.info("Raw CSV successfully copied to new data bucket.")


default_args = {
    "owner": "airflow"
}


with DAG(
    dag_id="telco_churn_ingest",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,     # Airflow 2.6+ recommended replacement for schedule_interval
    catchup=False,
    tags=["telco", "raw-copy"],
) as dag:

    copy_raw = PythonOperator(
        task_id="copy_raw_csv_to_data_bucket",
        python_callable=copy_raw_to_data_bucket
    )

