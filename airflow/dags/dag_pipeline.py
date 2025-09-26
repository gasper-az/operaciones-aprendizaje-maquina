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

def move_to_raw():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    shutil.copy(EXTERNAL_PATH, RAW_PATH)
    print(f"[OK] Archivo copiado a raw/: {RAW_PATH}")

with DAG(
    dag_id="dag_bodyfat_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bodyfat", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="check_file_external",
        python_callable=check_external_file,
    )

    t2 = PythonOperator(
        task_id="copy_file_to_raw",
        python_callable=move_to_raw,
    )

    t3 = BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/ml/preprocess.py "
    )

    t4 = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/ml/train.py "
    )

    t5 = BashOperator(
        task_id="evaluate_model",
        bash_command="python /opt/airflow/ml/eval.py "
    )

    t6 = BashOperator(
        task_id="deploy_model",
        bash_command="python /opt/airflow/utilities/promote_model.py "
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
