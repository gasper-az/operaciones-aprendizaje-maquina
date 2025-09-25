# ROD:
# REEMPLAZARLO POR EL PRE PROCESAMIENTO REAL DEL .CSV


# el PREPROCESADO SE TIENE QUE GUARDAR EN  DATA/PREPROCESSED


import os
import pandas as pd

BASE_PATH = "/opt/data"
RAW_PATH = os.path.join(BASE_PATH, "raw", "bodyfat.csv")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed", "bodyfat_clean.csv")

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"[ERROR] No existe archivo RAW en {RAW_PATH}")

    print(f"[INFO] Leyendo archivo RAW desde {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    # Simulación: eliminar nulos
    df_clean = df.dropna()
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)

    print(f"[OK] Preprocesamiento completado. Guardado en {PROCESSED_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Error en preprocess.py: {e}")
        raise
