import mlflow
import os
import sys
import pandas as pd

from typing import Any, Tuple

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

import awswrangler as wr
from scipy import stats

sys.path.append("../../")

os.environ["AWS_ACCESS_KEY_ID"]="minio"   
os.environ["AWS_SECRET_ACCESS_KEY"]="minio123" 
os.environ["MLFLOW_S3_ENDPOINT_URL"]="http://localhost:9000"
os.environ["AWS_ENDPOINT_URL_S3"]="http://localhost:9000"

from utilities.scripts.constants import (
    MODEL_NAME, MODEL_ALIAS,
    S3_DATA_PRE_PROCESSED, TRAIN_SUBFOLDER,
    X_CSV, Y_CSV
)

from utilities.scripts.procesamiento import (
    agregar_bmi,
    categorizar_bmi,
    aplicar_transformaciones_inline,
    one_hot_encoding_bmi,
    codificar_dummy_feature,
    escalar_features
)

from utilities.scripts.s3 import load_data

MLFLOW_TRACKING_URL = "http://localhost:5001" #os.getenv("MLFLOW_TRACKING_URL", "http://mlflow:5001")

def cargar_modelo(name: str, alias: str) -> Tuple[Any, int]:
    """
    Carga un modelo de Machine Learning desde MLFlow.

    Args:
        name (str): nombre del modelo.
        alias (str): alias del modelo.

    Returns:
        Tuple[Any, int]: modelo de ML junto a su versión.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    client = mlflow.MlflowClient()

    data = client.get_model_version_by_alias(name=name, alias=alias)
    model = mlflow.sklearn.load_model(data.source)
    version = int(data.version)

    return model, version

def cargar_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga datos de entrenamiento.
    Returns
        Tuple[pd.DataFrame, pd.Series]: X_train, y_train.
    """
    train_path = os.path.join(S3_DATA_PRE_PROCESSED, TRAIN_SUBFOLDER)

    # X_train, y_train = load_data(base_path=train_path, x_name=X_CSV, y_name=Y_CSV)
    X_train = wr.s3.read_csv("s3://data/body_fat/preprocessed/train/X.csv")
    y_train = wr.s3.read_csv("s3://data/body_fat/preprocessed/train/y.csv")
    return X_train, y_train

class ModelInput(BaseModel):
    """
    Input schema para el modelo de predicción de `Body Fat`.
    """
    density: float = Field(
        description="Densidad, determinada a partir del pesaje debajo del agua.",
        ge=0,
    )

    age: int = Field(
        description="Edad, medida en años.",
        ge=0,
        le=100
    )

    weight:  float = Field(
        description="Peso, en libras.",
        ge=0,
    )

    height: float = Field(
        description="Altura, en pulgadas.",
        ge=0,
    )

    neck:  float = Field(
        description="Circunferencia del cuello, medida en centímetros.",
        ge=0,
    )

    chest: float = Field(
        description="Circunferencia del pecho, en centímetros.",
        ge=0,
    )

    abdomen:  float = Field(
        description="Circunferencia del abdomen, en centímetros.",
        ge=0,
    )

    hip: float = Field(
        description="Circunferencia de la cintura, en centímetros.",
        ge=0,
    )

    thigh:  float = Field(
        description="Circunferencia del muslo, en centímetros.",
        ge=0,
    )

    knee: float = Field(
        description="Circunferencia de la rodilla, en centímetros.",
        ge=0,
    )

    ankle:  float = Field(
        description="	Circunferencia del tobillo, en centímetros.",
        ge=0,
    )

    bicep: float = Field(
        description="Circunferencia del bicep (extendido), en centímetros.",
        ge=0,
    )

    forearm: float = Field(
        description="Circunferencia del antebrazo, en centímetros.",
        ge=0,
    )

    wrist: float = Field(
        description="Circunferencia de la muñeca, en centímetros.",
        ge=0,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "density": 1.0708,
                    "age": 23,
                    "weight": 154.25,
                    "height": 67.75,
                    "neck": 36.2,
                    "chest": 93.1,
                    "abdomen": 85.2,
                    "hip": 94.5,
                    "thigh": 59.8,
                    "knee": 37.3,
                    "ankle": 24.0,
                    "bicep": 32.4,
                    "forearm": 26.5,
                    "wrist": 16.6
                }
            ]
        }
    }

class ModelOutput(BaseModel):
    """
    Output schema para el modelo de predicción de `Body Fat`.
    """
    prediccion: float = Field(
        description="Predicción de `Body Fat` del modelo."
    )

    prediccion_str: str = Field(
        description="Predicción de `Body Fat` del modelo como string."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediccion": 12.3,
                    "prediccion_str": "El valor de `Body Fat` predicho es de 12.3."
                }
            ]
        }
    }

def model_to_dataframe(model: ModelInput) -> pd.DataFrame:
    """
    Convierte un objeto del tipo ModelInput a un dataframe.

    Args:
        model (ModelInput): objeto a convertir en dataframe.
    Returns
        pd.DataFrame: representación en dataframe del objeto.
    """
    columns = ["Density", "Age", "Weight",
               "Height", "Neck", "Chest",
               "Abdomen", "Hip", "Thigh",
               "Knee", "Ankle", "Biceps",
               "Forearm", "Wrist"]

    dataframe = pd.DataFrame(columns=columns)

    dataframe["Density"] = [model.density]
    dataframe["Age"] = [model.age]
    dataframe["Weight"] = [model.weight]
    dataframe["Height"] = [model.height]
    dataframe["Neck"] = [model.neck]
    dataframe["Chest"] = [model.chest]
    dataframe["Abdomen"] = [model.abdomen]
    dataframe["Hip"] = [model.hip]
    dataframe["Thigh"] = [model.thigh]
    dataframe["Knee"] = [model.knee]
    dataframe["Ankle"] = [model.ankle]
    dataframe["Biceps"] = [model.bicep]
    dataframe["Forearm"] = [model.forearm]
    dataframe["Wrist"] = [model.wrist]

    return dataframe

def preprocesar(X_train:pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesa los datos enviados mediante una API.

    Args:
        X_train (pd.DataFrame): Datos de entrenamiento Originales. Se usan para obtener un escalado
            similar al utilizado cuando se entrenó el modelo.
        X_test (pd.DataFrame): Datos enviados mediante la API, en formato de DataFrame.
    Returns
        Tuple[pd.DataFrame, pd.DataFrame]: X_train y X_test. X_test listo para realizar predicciones
            mediante el modelo de ML.
    """
    colBMI = "BMI"
    colBMI_cat = "BMI_cat"

    agregar_bmi(X_train, bmiCol=colBMI)
    categorizar_bmi(X_train, bmiCol=colBMI, categoryCol=colBMI_cat)
    
    agregar_bmi(X_test, bmiCol=colBMI)
    categorizar_bmi(X_test, bmiCol=colBMI, categoryCol=colBMI_cat)

    cols_a_transformar = list(set(X_train.columns) - set([colBMI_cat]))
    aplicar_transformaciones_inline(X_train, "boxcox", cols_a_transformar)

    # Al haber solo un elemento, boxcox y otras transformaciones NO funcionan
    # aplicar_transformaciones_inline(X_test,  "boxcox", cols_a_transformar)

    X_train, X_test, new_cols_bmi = one_hot_encoding_bmi(
        train=X_train, test=X_test,
        categoryCol=colBMI_cat, prefix=colBMI
    )

    X_train, X_test = codificar_dummy_feature(
        train=X_train, test=X_test, categoryCols=new_cols_bmi
    )

    cols_a_escalar = list(set(X_train.columns) - set(new_cols_bmi))
    escalar_features(X_train, X_test, scaler="StandardScaler", columnas=cols_a_escalar)

    return X_train, X_test

# Carga de Modelo para predicción.
modelo, version = cargar_modelo(name=MODEL_NAME, alias=MODEL_ALIAS)
# Carga de datos para procesamiento.
X_train, y_train = cargar_train_data()

app = FastAPI()

@app.get("/")
async def root():
    """
    Root endpoint de Body Fat Predictor API.

    Devuelve un JSON response con un welcome message..
    """
    return JSONResponse(content=jsonable_encoder({"message": "Bienvenido a `Body Fat Predictor` API!!!"}))

@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    df = model_to_dataframe(model=features)
    _, processed = preprocesar(X_train=X_train, X_test=df)

    prediccion = modelo.predict(processed)

    prediccion_value = prediccion[0].item()
    prediccion_str = f"El valor de `Body Fat` predicho es de {prediccion_value}."
    return ModelOutput(prediccion=prediccion_value, prediccion_str=prediccion_str)