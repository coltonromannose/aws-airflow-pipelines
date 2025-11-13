import boto3
import logging
from io import StringIO
from datetime import datetime
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator


# ----------------------------------------------------------------------
# Bucket configuration
# The raw file was copied here by my ingestion DAG.
# This cleaning DAG reads the raw CSV and writes a sanitized version
# into the processed zone of the same bucket.
# ----------------------------------------------------------------------
DATA_BUCKET = "airflow-telco-data-v-colton"
RAW_KEY = "raw/Telco-Customer-Churn.csv"
CLEAN_KEY = "processed/telco_clean.csv"


def clean_telco_data(**context):
    """
    Load the raw Telco Customer Churn CSV from S3, apply the minimal
    cleaning steps needed to make the dataset usable, and write the
    cleaned output back to S3.

    Important: this step is **not** feature engineering — just
    correcting issues in the raw IBM dataset such as empty strings,
    mis-typed numeric fields, and inconsistent whitespace.
    """

    s3 = boto3.client("s3")

    # ------------------------------------------------------------------
    # 1. Load raw CSV from the raw data zone in S3
    # ------------------------------------------------------------------
    logging.info(f"Loading raw CSV from s3://{DATA_BUCKET}/{RAW_KEY}")
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=RAW_KEY)
    df = pd.read_csv(obj["Body"])
    logging.info(f"Raw dataset loaded. Shape: {df.shape}")

    # ------------------------------------------------------------------
    # 2. Clean column names (IBM dataset ships with stray whitespace)
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------------
    # 3. Strip whitespace inside string columns
    # This cleans up values like " Yes" or "No ".
    # ------------------------------------------------------------------
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # ------------------------------------------------------------------
    # 4. Convert numeric fields
    # The Telco dataset stores numbers as strings, and TotalCharges has
    # empty strings that must be treated as missing before conversion.
    # ------------------------------------------------------------------
    df["TotalCharges"] = df["TotalCharges"].replace("", pd.NA)

    numeric_columns = ["MonthlyCharges", "TotalCharges", "tenure", "SeniorCitizen"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 5. Drop rows with invalid or missing critical fields
    # - customerID: required unique identifier
    # - TotalCharges: contains ~11 blank values that cannot be recovered
    # ------------------------------------------------------------------
    df = df.dropna(subset=["customerID", "TotalCharges"]).reset_index(drop=True)

    logging.info(f"Dataset after cleaning: {df.shape}")

    # ------------------------------------------------------------------
    # 6. Write cleaned dataset back to S3 (processed zone)
    # ------------------------------------------------------------------
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    logging.info(f"Writing cleaned CSV to s3://{DATA_BUCKET}/{CLEAN_KEY}")
    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=CLEAN_KEY,
        Body=csv_buffer.getvalue()
    )

    logging.info("Cleaned dataset successfully written to processed folder.")


# ----------------------------------------------------------------------
# DAG definition
# - schedule=None means “manual trigger only”
# - start_date must be static (Airflow requirement)
# ----------------------------------------------------------------------
default_args = {"owner": "airflow"}

with DAG(
    dag_id="telco_cleaning",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["telco", "cleaning"],
) as dag:

    clean_data_task = PythonOperator(
        task_id="clean_telco_dataset",
        python_callable=clean_telco_data
    )
