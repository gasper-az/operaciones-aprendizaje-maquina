from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os, shutil
from dotenv import load_dotenv

# Configuración
load_dotenv("/opt/airflow/.env")
BASE_PATH = os.getenv("BASE_PATH", "/opt/data")
RAW_PATH = os.getenv("RAW_PATH", os.path.join(BASE_PATH, "raw", "bodyfat.csv"))
EXTERNAL_PATH = os.path.join(BASE_PATH, "external", os.path.basename(RAW_PATH))

def check_external_file():
    if not os.path.exists(EXTERNAL_PATH):
        raise FileNotFoundError(f"No se encontró {EXTERNAL_PATH}")
    print(f"[OK] Archivo encontrado: {EXTERNAL_PATH}")

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

    check_file = PythonOperator(
        task_id="check_file_external",
        python_callable=check_external_file,
    )

    copy_file = PythonOperator(
        task_id="copy_file_to_raw",
        python_callable=move_to_raw,
    )
    
    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command="python /opt/airflow/ml/preprocess.py"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/ml/train.py "
    )

    eval = BashOperator(
        task_id="evaluate_model",
        bash_command="python /opt/airflow/ml/eval.py "
    )

    deploy = BashOperator(
        task_id="deploy_model",
        bash_command="python /opt/airflow/ml/promote_model.py "
    )

    check_file >> copy_file >> preprocess >> train >> eval >> deploy