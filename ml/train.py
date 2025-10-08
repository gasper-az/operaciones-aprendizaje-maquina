import os
import mlflow
import mlflow.sklearn
import awswrangler as wr
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# ======================================================
# CONFIGURACIÓN DE ENTORNO Y CREDENCIALES
# ======================================================
load_dotenv()  # Carga variables desde .env (si existe)

# Variables de entorno
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ACCESS_KEY", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_SECRET_ACCESS_KEY", "minio123")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")

# ======================================================
# CONFIGURACIÓN DE MLflow
# ======================================================
TRACKING_URL = "http://mlflow:5000"
EXPERIMENT_NAME = "body_fat_regression_model"
MODEL_NAME = "body_fat_decision_tree_regressor"
RUN_NAME = "decision_tree_regressor"

mlflow.set_tracking_uri(TRACKING_URL)
client = mlflow.MlflowClient()

# Crear experimento si no existe
if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        tags={"project": "body_fat", "team": "mlops1-fiuba"}
    )

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# ======================================================
# CARGA DE DATOS DESDE MINIO (S3)
# ======================================================
bucket = os.getenv("DATA_REPO_BUCKET_NAME", "data")
S3_BASE = f"s3://{bucket}/body_fat/processed"

print(f"[INFO] Leyendo datasets desde {S3_BASE} ...")

try:
    X_train = wr.s3.read_csv(f"{S3_BASE}/train/X.csv")
    y_train = wr.s3.read_csv(f"{S3_BASE}/train/y.csv").iloc[:, 0]

    X_test = wr.s3.read_csv(f"{S3_BASE}/test/X.csv")
    y_test = wr.s3.read_csv(f"{S3_BASE}/test/y.csv").iloc[:, 0]
except Exception as e:
    raise FileNotFoundError(f"[ERROR] No se pudieron leer los archivos desde MinIO ({S3_BASE}): {e}")

print("[INFO] Datos cargados correctamente desde MinIO.")

# ======================================================
# ENTRENAMIENTO Y LOGGING EN MLFLOW
# ======================================================
model = DecisionTreeRegressor(random_state=42)

with mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name=RUN_NAME,
    tags={"model": "decision_tree_regressor"}
):
    # Entrenamiento
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}

    # Logueo en MLflow
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, MODEL_NAME, input_example=X_test[:1])

    print(f"[OK] Entrenamiento completado. Métricas: {metrics}")

print("[OK] Entrenamiento y logging en MLflow completado correctamente.")
