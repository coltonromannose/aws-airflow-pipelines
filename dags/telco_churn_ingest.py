import boto3
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

MWAA_BUCKET = "airflow-server-setup-v-colton"
DATA_BUCKET = "airflow-telco-data-v-colton"

RAW_SOURCE_KEY = "data/Telco-Customer-Churn.csv"
RAW_DEST_KEY = "raw/Telco-Customer-Churn.csv"

def copy_raw_to_data_bucket(**context):
    """Copies the raw CSV from the MWAA bucket into the new data bucket (raw/)."""
    s3 = boto3.client("s3")

    logging.info(
        f"Copying raw CSV from s3://{MWAA_BUCKET}/{RAW_SOURCE_KEY} "
        f"to s3://{DATA_BUCKET}/{RAW_DEST_KEY}"
    )

    s3.copy_object(
        Bucket=DATA_BUCKET,
        CopySource={"Bucket": MWAA_BUCKET, "Key": RAW_SOURCE_KEY},
        Key=RAW_DEST_KEY
    )

    logging.info("Raw CSV successfully copied to new data bucket.")


default_args = {"owner": "airflow"}

with DAG(
    dag_id="telco_copy_raw",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["telco", "copy", "raw"],
) as dag:

    copy_raw = PythonOperator(
        task_id="copy_raw_csv_to_data_bucket",
        python_callable=copy_raw_to_data_bucket
    )
