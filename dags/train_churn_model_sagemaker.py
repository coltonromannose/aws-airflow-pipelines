from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from datetime import datetime
import pandas as pd
import boto3
import io

# -----------------------
# Config
# -----------------------
BUCKET = "airflow-telco-data-v-colton"
FEATURES_PARQUET = "features/telco_features.parquet"
TRAIN_CSV = "sagemaker/train/train.csv"
OUTPUT_PATH = f"s3://{BUCKET}/sagemaker/output/"
REGION = "us-west-2"

# -----------------------
# Step 1 — Convert parquet to CSV with label first
# -----------------------
def prepare_training_data():
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=FEATURES_PARQUET)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))

    # Move label to first column
    cols = ["Churn"] + [c for c in df.columns if c != "Churn"]
    df = df[cols]

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=BUCKET,
        Key=TRAIN_CSV,
        Body=csv_buffer.getvalue().encode("utf-8")
    )
    print("CSV saved to S3:", TRAIN_CSV)

# -----------------------
# Step 2 — SageMaker job config
# -----------------------
def get_training_config():
    return {
        "TrainingJobName": "telco-xgboost-" + datetime.now().strftime("%Y%m%d%H%M%S"),
        "AlgorithmSpecification": {
            "TrainingImage": f"811284229777.dkr.ecr.{REGION}.amazonaws.com/xgboost:1",
            "TrainingInputMode": "File"
        },
        "RoleArn": "arn:aws:iam::156041438776:role/service-role/AmazonMWAA-airflow-test-v-colton-DsCqCN",
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{BUCKET}/sagemaker/train/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv"
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": OUTPUT_PATH
        },
        "ResourceConfig": {
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 20
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 600
        },
        "HyperParameters": {
            "objective": "binary:logistic",
            "num_round": "200",
            "max_depth": "5",
            "eta": "0.1",
            "subsample": "0.9",
            "colsample_bytree": "0.9",
            "eval_metric": "auc"
        }
    }

# -----------------------
# DAG definition
# -----------------------
with DAG(
    dag_id="train_churn_model_sagemaker",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["telco", "sagemaker", "training"]
) as dag:

    step1_convert = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data
    )

    step2_train = SageMakerTrainingOperator(
        task_id="run_sagemaker_training",
        config=get_training_config(),
        aws_conn_id="aws_default",
        wait_for_completion=True
    )

    step1_convert >> step2_train
