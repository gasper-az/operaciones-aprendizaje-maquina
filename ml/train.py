# ROD:
# ES UN TRAIN QUE SOLO ME SIRVE PARA PODER ARMAR EL AIRFLOY Y MLFLOW A SU ALREDEDOR
# REEMPLAZARLO POR EL DEL MODELO


import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar dataset de ejemplo (aunque no sea el final)
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model", registered_model_name="bodyfat_model")
