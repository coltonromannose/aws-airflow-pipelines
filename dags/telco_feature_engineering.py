import boto3
import logging
from io import StringIO, BytesIO
from datetime import datetime

import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator


# S3 configuration
# This DAG reads the cleaned dataset from the processed zone and
# generates ML-ready numeric features written as Parquet.
# ----------------------------------------------------------------------

DATA_BUCKET = "airflow-telco-data-v-colton"
CLEANED_KEY = "processed/telco_clean.csv"
FEATURE_KEY = "features/telco_features.parquet"


def engineer_features(**context):
    """
    Load the cleaned Telco dataset from S3, apply feature engineering, and
    write the resulting feature matrix back as a Parquet file.

    This step focuses on:
    - encoding categorical fields (binary + one-hot + ordinal)
    - deriving extra high-signal features (tenure buckets, service count, ratios)
    - producing a compact, ML-friendly feature matrix ready for training
    """

    s3 = boto3.client("s3")

    # ------------------------------------------------------------------
    # 1. Load the cleaned CSV from S3 into Pandas
    # ------------------------------------------------------------------
    logging.info(f"Loading cleaned dataset from s3://{DATA_BUCKET}/{CLEANED_KEY}")

    obj = s3.get_object(Bucket=DATA_BUCKET, Key=CLEANED_KEY)
    df = pd.read_csv(obj["Body"])

    logging.info(f"Loaded cleaned dataset. Shape = {df.shape}")

    # ------------------------------------------------------------------
    # 2. Encode binary Yes/No columns to 1/0
    # ------------------------------------------------------------------
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Churn"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # ------------------------------------------------------------------
    # 3. One-hot encode multi-category fields
    # ------------------------------------------------------------------
    one_hot_cols = ["InternetService", "PaymentMethod", "MultipleLines"]

    df = pd.get_dummies(df, columns=one_hot_cols, prefix=one_hot_cols)

    # ------------------------------------------------------------------
    # 4. Ordinal encode Contract (this category has natural order)
    # ------------------------------------------------------------------
    contract_order = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    }

    df["Contract"] = df["Contract"].map(contract_order)

    # ------------------------------------------------------------------
    # 5. Derived feature: count number of "Yes" services
    #    This is consistently one of the strongest churn predictors.
    # ------------------------------------------------------------------
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df["num_services"] = df[service_cols].sum(axis=1)

    # ------------------------------------------------------------------
    # 6. Derived feature: tenure bucket
    #    A simple but effective way to capture non-linear churn behavior.
    # ------------------------------------------------------------------
    def bucket_tenure(t):
        if t < 12:
            return "0-12"
        elif t < 24:
            return "12-24"
        elif t < 48:
            return "24-48"
        else:
            return "48+"

    df["tenure_bucket"] = df["tenure"].apply(bucket_tenure)
    df = pd.get_dummies(df, columns=["tenure_bucket"], prefix="tenure")

    # ------------------------------------------------------------------
    # 7. Derived feature: MonthlyCharges-to-TotalCharges ratio
    #    Helps separate long-term customers from new customers with
    #    similar monthly payments.
    # ------------------------------------------------------------------
    df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)

    # ------------------------------------------------------------------
    # 8. Drop columns that should not be fed to ML models
    # ------------------------------------------------------------------
    df = df.drop(columns=["customerID"])

    logging.info(f"Final feature matrix shape = {df.shape}")

    # ------------------------------------------------------------------
    # 9. Write the feature matrix to S3 as a Parquet file
    # ------------------------------------------------------------------
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)

    logging.info(f"Writing features to s3://{DATA_BUCKET}/{FEATURE_KEY}")

    s3.put_object(
        Bucket=DATA_BUCKET,
        Key=FEATURE_KEY,
        Body=buffer.getvalue()
    )

    logging.info("Feature engineering complete.")


# ----------------------------------------------------------------------
# DAG Definition
# ----------------------------------------------------------------------

default_args = {"owner": "airflow"}

with DAG(
    dag_id="telco_feature_engineering",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["telco", "feature-engineering"],
) as dag:

    feature_task = PythonOperator(
        task_id="engineer_telco_features",
        python_callable=engineer_features
    )
