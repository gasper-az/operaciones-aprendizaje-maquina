# Operaciones de Aprendizaje de Máquina

## Integrantes

- Acevedo Zain, Gaspar (acevedo.zain.gaspar@gmail.com)
- Chunga, Juan Miguel (juanmiguel.ch2014@gmail.com)
- Gonzalez, Juan (juan.gonzalez.working@gmail.com)
- Lauro, Rodrigo (ing.rodrigo.lauro@gmail.com)
- Rodrigues da Cruz, Nicolás (Nicolasrdc151@gmail.com)

## Descripción

Implementación de un modelo de Machine Learning (**Body Fat Analysis**) en un entorno productivo simulado con **Docker Compose**.  
El objetivo es desplegar y orquestar un pipeline de MLOps con los siguientes servicios:

- **Apache Airflow**: orquestación de flujos de trabajo (DAGs de entrenamiento y predicción).
- **MLflow**: gestión del ciclo de vida de modelos (tracking, versionado y registro).
- **FastAPI**: servicio REST para exponer el modelo entrenado.
- **MinIO (S3 compatible)**: almacenamiento de datasets y artefactos de modelos.
- **PostgreSQL**: base de datos backend para MLflow y Airflow.
- **Valkey (Redis fork)**: backend para ejecución distribuida de Airflow.

### Entrega 1

- Notebook loggeando en MLflow.
- DAG de entrenamiento básico corriendo.
- README inicial.

### Entrega final

- DAG con hiperparámetros + promoción a Registry.
- Batch predict funcionando en Airflow.
- API usando modelo en Production.
- Documentación + tests básicos.

### 📂 Estructura de carpetas

```YAML
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
│   └── scripts/
│       └── mlflow.sql
│ ├── seed_minio.sh # Inicializar buckets y subir dataset
│ └── promote_model.py # Promover modelo en MLflow Registry
│
├── dockerfiles/
│   └── airflow/
│       └── Dockerfile
│
├── requirements.txt # Dependencias globales del proyecto
├── mlflow.sql # Script de inicialización de la DB de MLflow
├── docker-compose.yaml # Orquestación de servicios
└── README.md # Documentación del proyecto
```

### Comandos útiles

***Requerimientos***

- *Docker* o *docker desktop* instalados

***Comandos***

- Para correr el proyecto:
  - `cd ./OPERACIONES-APRENDIZAJE-MAQUINA`.
  - `docker compose up`.
  - Opcionalmente `docker compose -f docker-compose.yaml up` para espeficiar el archivo compose.
- Para hacer build luego de actualizar imágenes y/o `docker-compose.yaml`:
  - `cd ./OPERACIONES-APRENDIZAJE-MAQUINA`.
  - `docker compose build`.
- Para hacer build y ejecutar nuevas imágenes:
  - `cd ./OPERACIONES-APRENDIZAJE-MAQUINA`.
  - `docker compose up --build`
- Para hacer build y forzar la recreación de containers:
  - `docker compose up --build --force-recreate`
- Para borrar los containers:
  - `cd ./OPERACIONES-APRENDIZAJE-MAQUINA`.
  - `docker compose down`.
  - Opcionalmente `docker compose down -v` para borrar los ***volúmenes*** creados.
- Para hacer un *clean build*:
  - `cd ./OPERACIONES-APRENDIZAJE-MAQUINA`.
  - `docker compose build --no-cache`
  - `docker compose up --force-recreate`
- ***Hints***
  - Ejecutar `docker compose up --force-recreate` para forzar la recreación de imágenes y containers.
    - Ideal para cuando se están probando configuraciones nuevas, pero los updates *no se reflejan*.
    - Puede demorar un poco más de tiempo de lo habitual.
  - Si no se notan updates luego del comando anterior, probar con hacer un *clean build*.
    - Esto demora incluso mucho más tiempo, ya que tiene que descargar las imágenes nuevamente.
  - Eventualmente se pueden borrar las imágenes que no se vieron afectadas por el update.
    - Desde docker desktop, primero eliminar los containers, luego las imágenes y volúmenes.
    - Alternativa: ver respuestas en este [post de Stack Overflow](https://stackoverflow.com/questions/44785585/how-can-i-delete-all-local-docker-images).
- ***Recomendación*** - Recrear ambiente local
  - `docker compose down -v`
  - Borrar las siguientes carpetas
    - `/airflow/dags/__pycache__`
    - `/minio_data`
  - `docker compose up --build`
