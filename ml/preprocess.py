import sys
import os
import pandas as pd




#from dotenv import load_dotenv

# Cargar variables desde .env (por compatibilidad local)
#load_dotenv()

# ============================
# PATHS DESDE .env
# ============================

BASE_PATH = os.getenv("BASE_PATH", "/opt/data")
RAW_PATH = os.getenv("RAW_PATH", os.path.join(BASE_PATH, "raw", "bodyfat.csv"))
PROCESSED_PATH = os.getenv("PROCESSED_PATH", os.path.join(BASE_PATH, "processed"))

# Imports de Airflow
sys.path.append("/opt/airflow/")

from utilities.scripts.procesamiento import (
    split_dataframe_train_test,
    agregar_bmi,
    categorizar_bmi,
    aplicar_transformaciones_inline,
    one_hot_encoding_bmi,
    codificar_dummy_feature,
    escalar_features
)
from utilities.scripts.constants import (
    TARGET, TEST_SIZE, RANDOM_STATE,
    S3_DATA_PRE_PROCESSED, S3_DATA_PROCESSED,
    TRAIN_SUBFOLDER, TEST_SUBFOLDER,
    X_CSV, Y_CSV
)
from utilities.scripts.s3 import save_data

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"[ERROR] No existe archivo RAW en {RAW_PATH}")

    dataframe = pd.read_csv(RAW_PATH)

    X_train, X_test, y_train, y_test = split_dataframe_train_test(
        dataframe=dataframe,
        target=TARGET,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE
    )
    print("[INFO][preprocess] Split train y test completo.")

    pre_train_path = os.path.join(S3_DATA_PRE_PROCESSED, TRAIN_SUBFOLDER)
    save_data(X_train, y_train, base_path=pre_train_path, x_name=X_CSV, y_name=Y_CSV)

    print("[INFO][preprocess] Archivos pre-preproesamiento guardados en S3.")

    colBMI = "BMI"
    colBMI_cat = "BMI_cat"

    agregar_bmi(X_train, bmiCol=colBMI)
    categorizar_bmi(X_train, bmiCol=colBMI, categoryCol=colBMI_cat)
    
    agregar_bmi(X_test, bmiCol=colBMI)
    categorizar_bmi(X_test, bmiCol=colBMI, categoryCol=colBMI_cat)

    print("[INFO][preprocess] BMI + categorización completas en train y test.")

    cols_a_transformar = list(set(X_train.columns) - set([colBMI_cat]))

    aplicar_transformaciones_inline(X_train, "boxcox", cols_a_transformar)
    aplicar_transformaciones_inline(X_test,  "boxcox", cols_a_transformar)


    print("[INFO][preprocess] Transformaciones BoxCox completas.")

    X_train, X_test, new_cols_bmi = one_hot_encoding_bmi(
        train=X_train, test=X_test,
        categoryCol=colBMI_cat, prefix=colBMI
    )

    X_train, X_test = codificar_dummy_feature(
        train=X_train, test=X_test, categoryCols=new_cols_bmi
    )

    cols_a_escalar = list(set(X_train.columns) - set(new_cols_bmi))
    escalar_features(X_train, X_test, scaler="StandardScaler", columnas=cols_a_escalar)
    print("[INFO][preprocess] Escalado completo.")

    # Guardar en MinIO
    train_path = os.path.join(S3_DATA_PROCESSED, TRAIN_SUBFOLDER)
    test_path = os.path.join(S3_DATA_PROCESSED, TEST_SUBFOLDER)

    save_data(X_train, y_train, base_path=train_path, x_name=X_CSV, y_name=Y_CSV)
    save_data(X_test, y_test, base_path=test_path, x_name=X_CSV, y_name=Y_CSV)
    print("[INFO][preprocess] Archivos guardados en S3.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL][preprocess] Error en preprocess.py: {e}")
        raise
