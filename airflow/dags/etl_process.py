from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import shutil

BASE_PATH = "/opt/data"
EXTERNAL_PATH = os.path.join(BASE_PATH, "external", "bodyfat.csv")
RAW_PATH = os.path.join(BASE_PATH, "raw", "bodyfat.csv")

def check_external_file():
    if not os.path.exists(EXTERNAL_PATH):
        raise FileNotFoundError(f"No se encontró {EXTERNAL_PATH}")
    print(f"[OK] Archivo encontrado en external/: {EXTERNAL_PATH}")

with DAG(
    dag_id="etl_process",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bodyfat", "etl", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="check_file_external",
        python_callable=check_external_file,
    )

    t2= BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/ml/preprocess.py "
    )

    t1 >> t2
