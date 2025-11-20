from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import boto3
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import os

# -----------------------------
# S3 configuration
# -----------------------------
BUCKET_NAME = "airflow-telco-data-v-colton"
FEATURES_PATH = "features/telco_features.parquet"
MODEL_OUTPUT_PATH = "model/churn_model.pkl"
METRICS_OUTPUT_PATH = "metrics/churn_metrics.json"

# -----------------------------
# Main training function
# -----------------------------
def train_model(**context):

    # 1. Download features file from S3
    s3 = boto3.client("s3")
    local_features = "/tmp/telco_features.parquet"
    s3.download_file(BUCKET_NAME, FEATURES_PATH, local_features)

    # 2. Load dataset
    df = pd.read_parquet(local_features)

    # ASSUME target column name is "Churn" or similar
    # If the column name is different, update this line!
    target_col = "Churn"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train simple XGBoost classifier
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # 5. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs))
    }

    # 6. Save model locally
    local_model = "/tmp/churn_model.pkl"
    with open(local_model, "wb") as f:
        pickle.dump(model, f)

    # 7. Save metrics locally
    local_metrics = "/tmp/churn_metrics.json"
    with open(local_metrics, "w") as f:
        json.dump(metrics, f)

    # 8. Upload outputs to S3
    s3.upload_file(local_model, BUCKET_NAME, MODEL_OUTPUT_PATH)
    s3.upload_file(local_metrics, BUCKET_NAME, METRICS_OUTPUT_PATH)

    print("Model and metrics uploaded successfully to S3!")
    print(metrics)

# -----------------------------
# Airflow DAG
# -----------------------------
with DAG(
    dag_id="train_churn_model",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # MANUAL TRIGGER ONLY
    catchup=False,
    tags=["telco", "ml", "training"],
) as dag:

    train_task = PythonOperator(
        task_id="train_xgboost_model",
        python_callable=train_model,
        provide_context=True
    )
