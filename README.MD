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

### Cómo inicializar la aplicación?

***Pre requisitos***

1. `git`.
1. `docker` o `docker desktop`.

***Pasos necesarios***

1. Clonar el repo: `git clone https://github.com/gasper-az/operaciones-aprendizaje-maquina.git`.
1. Una vez descargado, ubicarse en el directorio del repositorio: `cd operaciones-aprendizaje-maquina`.
1. Asegurarse que `docker` (o `docker desktop`) esté inicializado y funcionando correctamente.
1. Ejecutar el comando `docker compose up`.

### Diagrama

![diagrama](./diagrams/arquitectura.drawio.png)

***Referencias***

1. **Redis** se utiliza para configurar **Airflow** al inicializar los servicios.
1. **Apache Airflow** consume el dataset de *BodyFat*, realizando distintas operaciones sobre el mismo.
1. **Apache Airflow** guarda el dataset procesado de *BodyFat* en **PostgreSQL**.
1. Mediante **MLFlow** se consume el dataset desde **PostgreSQL** con el fin de generar un modelo de *Machine Learning*.
1. Una vez generado el modelo, **MLFlow** lo guarda en **s3**.
1. Independientemente de los otros pasos, un *Usuario Final* realiza un llamado a la **REST API** para realizar una inferencia.
1. La **REST API** consume el modelo desde **s3**, realiza la inferencia, y le devuelve un resultado al *Usuario Final*.

### Entrega 1

- Servicios containerizados (*MLFlow*, *Airflow*, s3, *PostgreSQL*)
- Notebook loggeando en MLflow.
- *DAG* de procesamiento básico corriendo.
- *README* inicial.

### Entrega final

- DAG con hiperparámetros + promoción a Registry.
- Batch predict funcionando en Airflow.
- API usando modelo en Production.
- Documentación + tests básicos.

### 📂 Estructura de carpetas

```YAML
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
├── diagrams/
│
├── dockerfiles/
│   ├── airflow/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── mlflow/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── postgres/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── mlflow.sql
│
├── ml/ # Código fuente del modelo
│ ├── train.py # Entrenamiento y guardado del modelo
│ ├── infer.py # Predicción en batch
│ ├── data_utils.py # Utilidades de carga y preprocesamiento
│ └── config.yaml # Configuración de hiperparámetros y paths
│
├── notebooks/ # Exploración y prototipado de modelos
│ ├── body_fat_analysis.ipynb
│ └── mlflow_playground.ipynb
│
├── utilities/
│   └── scripts/ # scripts y otras utilidades
│       └── mlflow.sql
│ ├── seed_minio.sh # Inicializar buckets y subir dataset
│ └── promote_model.py # Promover modelo en MLflow Registry
│
├── .env
├── docker-compose.yaml # Orquestación de servicios
├── README.md # Documentación del proyecto
└── requirements.txt # Documentación del proyecto
```

### Comandos útiles

***Requerimientos***

- *Docker* o *docker desktop* instalados

***Comandos***

- Para correr el proyecto:
  - `docker compose up`.
  - Opcionalmente `docker compose -f docker-compose.yaml up` para espeficiar el archivo compose.
- Para hacer build de las imágenes:
  - `docker compose build`.
- Para hacer build y correr el proyecto:
  - `docker compose up --build`
- Para hacer build y forzar la recreación de containers:
  - `docker compose up --build --force-recreate`
- Para borrar los containers:
  - `docker compose down`.
  - Opcionalmente `docker compose down -v` para borrar los ***volúmenes*** creados.
- Para hacer un *clean build*:
  - ***NOTA***: esto va a volver a crear las imágenes de docker, por lo cual puede *demorar más tiempo* de lo habitual.
  - `docker compose build --no-cache`
  - `docker compose up --force-recreate`
- ***Recomendación*** - Recrear ambiente local
  - `docker compose down -v`
  - Borrar las siguientes carpetas
    - `/airflow/dags/__pycache__`
    - `/minio_data`
  - Opcional: borrar las imágenes de docker
    - Ver el siguiente [post de Stack Overflow](https://stackoverflow.com/questions/44785585/how-can-i-delete-all-local-docker-images).
  - `docker compose up --build`
