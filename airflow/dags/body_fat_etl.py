from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import shutil

# Esto nos permite acceder a los scripts ubicados en /opts/utilities/scripts
BASE_PATH = "/opt"
RAW_PATH = os.path.join(BASE_PATH, "data", "raw", "bodyfat.csv")
DAG_TEMPORAL_DATA_FOLDER = "/opt/data/processed"
TRAIN_PATH = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "train.csv")
TEST_PATH = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "test.csv")

# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../data/scripts')))

from utilities.scripts.procesamiento import (cargar_train_test_from_files,
            agregar_bmi, categorizar_bmi, aplicar_transformaciones_inline, one_hot_encoding_bmi,
            codificar_dummy_feature, escalar_features, guardar_train_test, split_dataframe_train_test)
from utilities.scripts.constants import TARGET, TEST_SIZE, RANDOM_STATE


def train_test_split():
    """
    Hace el split de train y test de un dataframe.

    Args:
        source (str): Path donde se encuentra el archivo del dataframe.
    Returns:
        (str, str): Path correspondientes a los archivos de train y test.
    """
    import pandas as pd
    # from utilities.scripts.procesamiento import guardar_train_test, split_dataframe_train_test
    # from utilities.scripts.constants import TARGET, TEST_SIZE, RANDOM_STATE
    dataset = pd.read_csv(RAW_PATH)
    X_train, X_test, y_train, y_test = split_dataframe_train_test(dataframe=dataset,
                                                                    random_state=RANDOM_STATE,
                                                                    target=TARGET,
                                                                    test_size=TEST_SIZE)
    
    train_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "train.csv")
    test_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "test.csv")

    guardar_train_test(X_train=X_train, X_test=X_test, y_train=y_train,
                        y_test=y_test, train_path=TRAIN_PATH, test_path=TEST_PATH, index=False)

    return {
        "X_train_file_path": TRAIN_PATH,
        "X_test_file_path": TEST_PATH
    }

def aplicar_feature_engineering():
    # from utilities.scripts.procesamiento import (cargar_train_test_from_files,
    #     agregar_bmi, categorizar_bmi, aplicar_transformaciones_inline, one_hot_encoding_bmi,
    #     codificar_dummy_feature, escalar_features, guardar_train_test)
    # from utilities.scripts.constants import TARGET

    X_train, X_test, y_train, y_test = cargar_train_test_from_files(train_path=TRAIN_PATH,
                                                                    test_path=TEST_PATH,
                                                                    target=TARGET)

    # Nueva feature: BMI + BMI_cat
    colBMI = "BMI"
    colBMI_cat = "BMI_cat"

    agregar_bmi(X_train, bmiCol=colBMI)
    categorizar_bmi(X_train, bmiCol=colBMI, categoryCol=colBMI_cat)

    agregar_bmi(X_test, bmiCol=colBMI)
    categorizar_bmi(X_test, bmiCol=colBMI, categoryCol=colBMI_cat)

    cols_a_transformar = list(set(X_train.columns) - set([colBMI_cat]))
    # Tratamiento de outliers con transformación Boxcox.
    # Aplica solo a las columna numéricas.
    aplicar_transformaciones_inline(dataframe=X_train, columnas=cols_a_transformar)
    aplicar_transformaciones_inline(dataframe=X_test, columnas=cols_a_transformar)

    X_train, X_test, new_cols_bmi = one_hot_encoding_bmi(train=X_train,
                                                        test=X_test,
                                                        categoryCol=colBMI_cat,
                                                        prefix=colBMI)

    X_train, X_test = codificar_dummy_feature(train=X_train, test=X_test,
                                            categoryCols=new_cols_bmi)
    
    cols_a_escalar = list(set(X_train.columns) - set(new_cols_bmi))
    escalar_features(train=X_train, test=X_test,
                    scaler="StandardScaler", columnas=cols_a_escalar)
    
    train_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "train_procesado.csv")
    test_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "test_procesado.csv")

    guardar_train_test(X_train=X_train, X_test=X_test, y_train=y_train,
                        y_test=y_test, train_path=train_path, test_path=test_path,
                        index=False)

    return {
        "X_train_file_path": train_path,
        "X_test_file_path": test_path
    }

with DAG(
    dag_id="body_fat_etl",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bodyfat", "pipeline"],
) as dag:

    t1 = PythonOperator(
        task_id="train_test_split",
        python_callable=train_test_split,
    )

    t2 = PythonOperator(
        task_id="aplicar_feature_engineering",
        python_callable=aplicar_feature_engineering,
    )

    t1 >> t2
