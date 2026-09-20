# Tutorial 03's image: stock Airflow plus the two libraries the DAG imports.
#
FROM apache/airflow:2.8.4-python3.11

USER airflow

ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt" \
      "pandas==2.1.4" \
      "pyarrow==14.0.2" \
      "scikit-learn==1.3.2" \
      "fastapi==0.109.2" \
      "uvicorn==0.27.1" \
      "matplotlib==3.8.2" \
      "seaborn==0.13.2" \
      "boto3==1.33.13" \

      "psycopg2-binary==2.9.9"

RUN pip install --no-cache-dir "mlflow==2.11.1" "email-validator>=2.0.0"

# Pre-set default admin password to 'b' for standalone mode on any machine
RUN echo "b" > /opt/airflow/standalone_admin_password.txt




