import boto3
import logging
import csv
from io import StringIO
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator


# ----------------------------------------------------------------------
# S3 Bucket configuration
# ----------------------------------------------------------------------
DATA_BUCKET = "airflow-telco-data-v-colton"
RAW_KEY = "raw/Telco-Customer-Churn.csv"
CLEAN_KEY = "processed/telco_clean.csv"


def clean_telco_data(**context):
    """
    Clean the Telco Customer Churn dataset using only Python's built-in
    CSV and string parsing tools so the DAG runs natively on MWAA
    without requiring external dependencies like pandas.

    Cleaning steps:
    - Trim whitespace in column names and string values
    - Convert numeric fields manually
    - Treat empty strings in TotalCharges as missing
    - Drop rows missing customerID or TotalCharges
    """

    s3 = boto3.client("s3")

    # ------------------------------------------------------------------
    # 1. Load raw CSV into memory
    # ------------------------------------------------------------------
    logging.info(f"Loading raw CSV from s3://{DATA_BUCKET}/{RAW_KEY}")
    obj = s3.get_object(Bucket=DATA_BUCKET, Key=RAW_KEY)
    raw_csv = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(StringIO(raw_csv))

    # Normalize column names (strip whitespace)
    fieldnames = [col.strip() for col in reader.fieldnames]

    cleaned_rows = []

    for row in reader:
        # Create new dict with stripped column names
        clean_row = {}

        for col, val in row.items():
            col_clean = col.strip()

            # Strip whitespace from values as well
            val_clean = val.strip() if isinstance(val, str) else val

            clean_row[col_clean] = val_clean

        # Drop rows missing customerID
        if clean_row.get("customerID") in (None, "", " "):
            continue

        # Convert numeric fields
        def to_float(x):
            if x in (None, "", " "):
                return None
            try:
                return float(x)
            except:
                return None

        clean_row["MonthlyCharges"] = to_float(clean_row.get("MonthlyCharges"))
        clean_row["TotalCharges"] = to_float(clean_row.get("TotalCharges"))

        # tenure + SeniorCitizen as ints
        def to_int(x):
            if x in (None, "", " "):
                return None
            try:
                return int(float(x))
            except:
                return None

        clean_row["tenure"] = to_int(clean_row.get("tenure"))
        clean_row["SeniorCitizen"] = to_int(clean_row.get("SeniorCitizen"))

        # Drop rows missing TotalCharges (dataset contains ~11)
        if clean_row["TotalCharges"] is None:
            continue

        cleaned_rows.append(clean_row)

    logging.info(f"Cleaned dataset size: {len(cleaned_rows)} rows")

    # ------------------------------------------------------------------
    # 2. Write cleaned rows back to S3
    # ------------------------------------------------------------------
    output_buffer = StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cleaned_rows)

    logging.info(f"Writing cleaned CSV to s3://{DATA_BUCKET}/{CLEAN_KEY}")

    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=CLEAN_KEY,
        Body=output_buffer.getvalue()
    )

    logging.info("Cleaned dataset written successfully.")


# ----------------------------------------------------------------------
# DAG Definition
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

