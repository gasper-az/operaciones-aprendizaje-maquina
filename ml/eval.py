import os
import pandas as pd
import mlflow
import mlflow.sklearn
import awswrangler as wr
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# ======================================================
# CONFIGURACIÓN DE ENTORNO Y CREDENCIALES
# ======================================================
load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_ACCESS_KEY", "minio123")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")

# ======================================================
# CONFIGURACIÓN DE MLflow
# ======================================================
TRACKING_URL = "http://mlflow:5000"
EXPERIMENT_NAME = "body_fat_regression_model"
RUN_NAME = "decision_tree_regressor"
MODEL_NAME = "body_fat_decision_tree_regressor"
PROD_MODEL_NAME = "body_fat_productive"

mlflow.set_tracking_uri(TRACKING_URL)
client = mlflow.MlflowClient()
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

runs = mlflow.search_runs(
    experiment_ids=experiment.experiment_id,
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'"
)
run_id = runs.iloc[0]["run_id"]
uri = f"runs:/{run_id}/{MODEL_NAME}"

print(f"[INFO] Cargando modelo desde: {uri}")
model = mlflow.sklearn.load_model(uri)

# ======================================================
# CARGA DE DATOS DESDE MINIO (S3)
# ======================================================
bucket = os.getenv("DATA_REPO_BUCKET_NAME", "data")
S3_BASE = f"s3://{bucket}/body_fat/processed"

print(f"[INFO] Leyendo dataset de evaluación desde {S3_BASE} ...")

try:
    X_test = wr.s3.read_csv(f"{S3_BASE}/test/X.csv")
    y_test = wr.s3.read_csv(f"{S3_BASE}/test/y.csv").iloc[:, 0]
except Exception as e:
    raise FileNotFoundError(f"[ERROR] No se pudieron leer los archivos desde MinIO ({S3_BASE}): {e}")

print("[INFO] Datos cargados correctamente desde MinIO.")

# ======================================================
# EVALUACIÓN Y PROMOCIÓN
# ======================================================
y_pred = model.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": root_mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}
print(f"[INFO] Métricas de evaluación: {metrics}")

# ======================================================
# REGISTRO / PROMOCIÓN DE MODELO EN MLFLOW
# ======================================================
try:
    client.get_registered_model(PROD_MODEL_NAME)
except Exception:
    client.create_registered_model(
        name=PROD_MODEL_NAME,
        description="Modelo productivo para predecir porcentaje de grasa corporal."
    )

model_version = client.create_model_version(
    name=PROD_MODEL_NAME,
    source=uri,
    run_id=run_id,
    description="Decision Tree Regressor actualizado"
)

client.set_registered_model_alias(PROD_MODEL_NAME, "champion", model_version.version)
print(f"[OK] Modelo '{PROD_MODEL_NAME}' actualizado como champion.")
