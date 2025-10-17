import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import os

# =====================================
# CONFIGURACIÓN
# =====================================
load_dotenv()

TRACKING_URL = "http://mlflow:5000"
EXPERIMENT_NAME = "body_fat_regression_model"
RUN_NAME = "decision_tree_regressor"
MODEL_NAME = "body_fat_decision_tree_regressor"
PROD_MODEL_NAME = "body_fat_productive"

mlflow.set_tracking_uri(TRACKING_URL)
client = MlflowClient()

print(f"[INFO] Promocionando modelo '{MODEL_NAME}' a producción...")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    order_by=["metrics.R2 DESC"]
)
if runs.empty:
    raise Exception(f"No se encontraron runs para el experimento '{EXPERIMENT_NAME}'")

best_run = runs.iloc[0]
run_id = best_run["run_id"]
model_uri = f"runs:/{run_id}/{MODEL_NAME}"

print(f"[INFO] Mejor modelo encontrado con run_id={run_id} (R2={best_run['metrics.R2']:.4f})")

try:
    client.get_registered_model(PROD_MODEL_NAME)
except Exception:
    client.create_registered_model(
        name=PROD_MODEL_NAME,
        description="Modelo productivo para predecir porcentaje de grasa corporal."
    )

model_version = client.create_model_version(
    name=PROD_MODEL_NAME,
    source=model_uri,
    run_id=run_id,
    description="Promoción automática del modelo entrenado más reciente."
)

client.set_registered_model_alias(PROD_MODEL_NAME, "champion", model_version.version)
print(f"[OK] Modelo '{PROD_MODEL_NAME}' actualizado como champion (versión {model_version.version}).")
