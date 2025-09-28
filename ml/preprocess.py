import sys
import os
import pandas as pd

BASE_PATH = "/opt/data"
RAW_PATH = os.path.join(BASE_PATH, "raw", "bodyfat.csv")
TRAIN_PROCESSED_PATH = os.path.join(BASE_PATH, "processed", "bodyfat_processed_train.csv")
TEST_PROCESSED_PATH = os.path.join(BASE_PATH, "processed", "bodyfat_processed_test.csv")

# Esto nos permite hacer import de /utilities/scripts
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

from utilities.scripts.procesamiento import (split_dataframe_train_test,
                                             agregar_bmi, categorizar_bmi,
                                             aplicar_transformaciones_inline,
                                             one_hot_encoding_bmi,
                                             codificar_dummy_feature,
                                             escalar_features)
from utilities.scripts.commons import (guardar_train_test)
from utilities.scripts.constants import (TARGET, TEST_SIZE, RANDOM_STATE)

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"[ERROR] No existe archivo RAW en {RAW_PATH}")

    dataframe = pd.read_csv(RAW_PATH)

    
    X_train, X_test, y_train, y_test = split_dataframe_train_test(dataframe=dataframe,
                                                                  target=TARGET,
                                                                  random_state=RANDOM_STATE,
                                                                  test_size=TEST_SIZE)
    print("[INFO][preprocess] Split train y test completo.")
    
    colBMI = "BMI"
    colBMI_cat = "BMI_cat"

    agregar_bmi(X_train, bmiCol=colBMI)
    categorizar_bmi(X_train, bmiCol=colBMI, categoryCol=colBMI_cat)

    agregar_bmi(X_test, bmiCol=colBMI)
    categorizar_bmi(X_test, bmiCol=colBMI, categoryCol=colBMI_cat)

    print("[INFO][preprocess] BMI + categorización completas en train y test.")

    cols_a_transformar = list(set(X_train.columns) - set([colBMI_cat]))
    aplicar_transformaciones_inline(dataframe=X_train, columnas=cols_a_transformar)
    aplicar_transformaciones_inline(dataframe=X_test, columnas=cols_a_transformar)

    print("[INFO][preprocess] Se transformaron las columnas mediante BoxCox.")

    X_train, X_test, new_cols_bmi = one_hot_encoding_bmi(train=X_train,
                                                         test=X_test,
                                                         categoryCol=colBMI_cat,
                                                         prefix=colBMI)

    print("[INFO][preprocess] One Hot encoding aplicado.")

    X_train, X_test = codificar_dummy_feature(train=X_train, test=X_test,
                                              categoryCols=new_cols_bmi)
    
    print("[INFO][preprocess] Codificación aplicada.")

    cols_a_escalar = list(set(X_train.columns) - set(new_cols_bmi))
    escalar_features(train=X_train, test=X_test,
                     scaler="StandardScaler", columnas=cols_a_escalar)
    
    print("[INFO][preprocess] Escalado aplicado.")

    guardar_train_test(X_train=X_train, X_test=X_test,
                       y_train=y_train, y_test=y_test,
                       train_path=TRAIN_PROCESSED_PATH,
                       test_path=TEST_PROCESSED_PATH)
    
    print("[INFO][preprocess] Archivos guardados.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL][preprocess] Error en preprocess.py: {e}")
        raise
