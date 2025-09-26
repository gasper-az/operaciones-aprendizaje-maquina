# ROD:
# REEMPLAZARLO POR EL DEL MODELO


import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import mlflow

BASE_PATH = "/opt/data"
PROCESSED_PATH = os.path.join(BASE_PATH, "processed", "bodyfat_clean.csv")

def main():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"[ERROR] No existe archivo procesado en {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH)
    if "BodyFat" not in df.columns:
        raise ValueError("[ERROR] No se encontró la columna 'BodyFat' en el dataset")

    # 🚨 Simulación de evaluación
    y_true = df["BodyFat"].values
    y_pred = df["BodyFat"].values  # "modelo tonto": predice el mismo valor

    mse = mean_squared_error(y_true, y_pred)

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("bodyfat-experiment")

    with mlflow.start_run(run_name="bodyfat_eval"):
        mlflow.log_metric("mse", mse)
        print(f"[OK] Evaluación completada. MSE={mse}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] Error en eval.py: {e}")
        raise
