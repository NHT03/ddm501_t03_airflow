# End-to-End MLOps Pipeline: Airflow + MLflow + Scikit-Learn + FastAPI

## Overview

Pipeline bao gồm đầy đủ quy trình MLOps:
1. **Airflow DAG (`wdbc_pipeline`)**: Ingest -> Validate -> Split -> Scale -> Train (Scikit-Learn RandomForest) -> Track MLflow -> Report.
2. **MLflow Tracking & Model Registry**: Theo dõi parameters, metrics (Accuracy, F1, ROC-AUC), artifacts (Confusion Matrix plot, Scaler.json) và quản lý phiên bản mô hình.
3. **FastAPI Model Serving API**: Phục vụ API dự đoán trực tiếp từ mô hình được đăng ký trên MLflow.

## Services & Ports

- **Airflow Web UI**: <http://127.0.0.1:18080> (User: `admin`)
- **MLflow Tracking Server**: <http://127.0.0.1:15000>
- **FastAPI Prediction API**: <http://127.0.0.1:18000> (Documentation Swagger tại <http://127.0.0.1:18000/docs>)

## Quick Start (Docker Compose)

```bash
# 1. Khởi chạy toàn bộ hệ thống (Airflow + MLflow + FastAPI)
docker compose up -d --build

# 2. Lấy mật khẩu admin Airflow
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt

# 3. Kích hoạt DAG chạy thử trong Airflow:
docker compose exec airflow airflow dags test wdbc_pipeline 2026-08-25

# 4. Kiểm tra API dự đoán:
python scripts/test_api.py
```

## Running Pipeline Locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.11.txt

export AIRFLOW_HOME=$PWD/.airflow
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_TRACKING_URI=http://localhost:15000

airflow standalone
```

## Pipeline Outputs

```
data/staging/2026-08-25/
  raw.parquet              snapshot of the extract
  clean.parquet            rows that passed validation
  rejected.parquet         rows that failed validation
  validation_report.json   counts and bad fraction
  train.parquet            scaled training dataset
  test.parquet             scaled test dataset
  scaler.json              Z-score mean and std
  confusion_matrix.png     confusion matrix visualization
  summary.json             summary report
data/staging/history.jsonl history log per date
```