import datetime
import os
import sys
from airflow.decorators import dag, task

# Esto nos permite acceder a los scripts ubicados en /opts/utilities/scripts
BASE_PATH = "/opt"
DAG_TEMPORAL_DATA_FOLDER = "/opt/data"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), BASE_PATH)))

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="body_fat_etl",
    description="ETL para procesar el dataset de Body Fat.",
    default_args=default_args,
    catchup=False,
    tags=["ETL", "Body-Fat"],
)
def process_etl_taskflow():

    @task.virtualenv(
        task_id="get_data_from_url",
        system_site_packages=False
    )
    def get_data_from_url() -> str:
        """
        Descarga los datos desde una URL, y los guarda en el filesystem.

        Returns
            (str): Path del archivo csv creado.
        """
        from utilities.scripts.procesamiento import descargar_y_guardar_dataframe
        from utilities.scripts.constants import DATASET_DOWNLOAD_URL

        destination = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "raw.csv")
        descargar_y_guardar_dataframe(DATASET_DOWNLOAD_URL, destination)

        return destination
    
    @task.virtualenv(
        task_id="train_test_split",
        requirements=["pandas>=2.0"],
        system_site_packages=False
    )
    def train_test_split(source: str):
        """
        Hace el split de train y test de un dataframe.

        Args:
            source (str): Path donde se encuentra el archivo del dataframe.
        Returns:
            (str, str): Path correspondientes a los archivos de train y test.
        """
        import pandas as pd
        from utilities.scripts.procesamiento import guardar_train_test, split_dataframe_train_test
        from utilities.scripts.constants import TARGET, TEST_SIZE, RANDOM_STATE
        
        dataset = pd.read_csv(source)
        X_train, X_test, y_train, y_test = split_dataframe_train_test(dataframe=dataset,
                                                                      random_state=RANDOM_STATE,
                                                                      target=TARGET,
                                                                      test_size=TEST_SIZE)
        
        train_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "train.csv")
        test_path = os.path.join(DAG_TEMPORAL_DATA_FOLDER, "test.csv")

        guardar_train_test(X_train=X_train, X_test=X_test, y_train=y_train,
                           y_test=y_test, train_path=train_path, test_path=test_path, index=False)

        return {
            "X_train_file_path": train_path,
            "X_test_file_path": test_path
        }

    @task.virtualenv(
        task_id="feature_engineering",
        requirements=["pandas>=2.0"],
        system_site_packages=False
    )
    def aplicar_feature_engineering(train_path: str, test_path: str):
        from utilities.scripts.procesamiento import (cargar_train_test_from_files,
            agregar_bmi, categorizar_bmi, aplicar_transformaciones_inline, one_hot_encoding_bmi,
            codificar_dummy_feature, escalar_features, guardar_train_test)
        from utilities.scripts.constants import TARGET

        X_train, X_test, y_train, y_test = cargar_train_test_from_files(train_path=train_path,
                                                                        test_path=test_path,
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

    # Encadenamiento
    path = get_data_from_url()
    files = train_test_split(path)
    final_files = aplicar_feature_engineering(files["X_train_file_path"], files["X_test_file_path"])

dag = process_etl_taskflow()
