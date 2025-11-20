from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import boto3
import json
import pickle
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
# Manual metrics (NO sklearn)
# -----------------------------
def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def f1_score_manual(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(2 * (precision * recall) / (precision + recall + 1e-9))

def roc_auc_manual(y_true, y_prob):
    # Simple and MWAA-safe AUC implementation
    # Rank-based AUC
    order = np.argsort(y_prob)
    y_true_sorted = np.array(y_true)[order]
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return 0.5
    rank_sum = np.sum(np.where(y_true_sorted == 1)[0])
    auc = (rank_sum - n1*(n1+1)/2) / (n1*n0)
    return float(auc)

# -----------------------------
# Training function
# -----------------------------
def train_model(**context):

    s3 = boto3.client("s3")

    # 1. Download features file from S3
    local_features = "/tmp/telco_features.parquet"
    s3.download_file(BUCKET_NAME, FEATURES_PATH, local_features)

    # 2. Load dataset
    df = pd.read_parquet(local_features)

    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Manual train/test split (NO sklearn)
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8

    X_train = X[mask]
    X_test = X[~mask]

    y_train = y[mask]
    y_test = y[~mask]

    # 4. Train XGBoost
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

    # 5. Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # 6. Metrics (manual, no sklearn)
    metrics = {
        "accuracy": accuracy(y_test.values, preds),
        "f1_score": f1_score_manual(y_test.values, preds),
        "roc_auc": roc_auc_manual(y_test.values, probs)
    }

    # 7. Save model
    local_model = "/tmp/churn_model.pkl"
    with open(local_model, "wb") as f:
        pickle.dump(model, f)

    # 8. Save metrics
    local_metrics = "/tmp/churn_metrics.json"
    with open(local_metrics, "w") as f:
        json.dump(metrics, f)

    # 9. Upload to S3
    s3.upload_file(local_model, BUCKET_NAME, MODEL_OUTPUT_PATH)
    s3.upload_file(local_metrics, BUCKET_NAME, METRICS_OUTPUT_PATH)

    print("Model + metrics uploaded to S3")
    print(metrics)

# -----------------------------
# Airflow DAG
# -----------------------------
with DAG(
    dag_id="train_churn_model",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["telco", "ml", "training"]
) as dag:

    train_task = PythonOperator(
        task_id="train_xgboost_model",
        python_callable=train_model,
        provide_context=True
    )
