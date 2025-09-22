
# Proyecto Final - MLOps1 (CEIA - FIUBA)

Implementación de un modelo de Machine Learning (**Body Fat Analysis**) en un entorno productivo simulado con **Docker Compose**.  
El objetivo es desplegar y orquestar un pipeline de MLOps con los siguientes servicios:

- **Apache Airflow**: orquestación de flujos de trabajo (DAGs de entrenamiento y predicción).
- **MLflow**: gestión del ciclo de vida de modelos (tracking, versionado y registro).
- **FastAPI**: servicio REST para exponer el modelo entrenado.
- **MinIO (S3 compatible)**: almacenamiento de datasets y artefactos de modelos.
- **PostgreSQL**: base de datos backend para MLflow y Airflow.
- **Valkey (Redis fork)**: backend para ejecución distribuida de Airflow.


Entrega 1
================
* Notebook + train.py loggeando en MLflow.
* DAG de entrenamiento básico corriendo.
* API /health y /predict sirviendo un modelo fijo.
* README inicial.



Entrega final
===============
* DAG con hiperparámetros + promoción a Registry.
* Batch predict funcionando en Airflow.
* API usando modelo en Production.
* Documentación + tests básicos.


## 📂 Estructura de carpetas


├── notebooks/ # Exploración y prototipado de modelos
│ └── body_fat_analysis.ipynb
│
├── ml/ # Código fuente del modelo
│ ├── train.py # Entrenamiento y guardado del modelo
│ ├── infer.py # Predicción en batch
│ ├── data_utils.py # Utilidades de carga y preprocesamiento
│ └── config.yaml # Configuración de hiperparámetros y paths
│
├── airflow/
│ ├── dags/ # DAGs de Airflow
│ │ ├── train_model_dag.py
│ │ └── batch_predict_dag.py
│ ├── logs/ # Logs (excluidos del repo)
│ └── plugins/ # Operadores y hooks personalizados
│
├── api/ # Servicio de inferencia con FastAPI
│ ├── app.py
│ ├── schemas.py
│ ├── Dockerfile
│ └── requirements.txt
│
├── utilities/
│ └── scripts/ # Scripts auxiliares
│ ├── seed_minio.sh # Inicializar buckets y subir dataset
│ └── promote_model.py # Promover modelo en MLflow Registry
│
├── DOCKERFILES/ # Dockerfiles personalizados (opcional)
│
├── requirements.txt # Dependencias globales del proyecto
├── mlflow.sql # Script de inicialización de la DB de MLflow
├── docker-compose.yaml # Orquestación de servicios
└── README.md # Documentación del proyecto


