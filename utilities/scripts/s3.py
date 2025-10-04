import awswrangler as wr
import pandas as pd
import os

def load_data(base_path: str, x_name: str, y_name: str):
    """
    Obtiene un dataframe cuyas features (X) y targets (y)
        se grabaron por separado.

    Args:
        base_path (str): Path base en donde se encuentran los archivos.
        x_name (str): Nombre específico del archivo de X.
        y_name (str): Nombre específico del archivo de y.
    
    Returns:
        X, y
    """
    X = wr.s3.read_csv(os.path.join(base_path, x_name))
    y = wr.s3.read_csv(os.path.join(base_path, y_name))

    return X, y

def save_data(X: pd.DataFrame, y: pd.Series, base_path: str,
              x_name: str, y_name: str, index: bool = False):
    """
    Guarda un dataframe dividido en Features (X) y target (y) en s3.

    Args:
        X (pd.DataFrame): Corresponde al dataframe de X.
        y (pd.Series): Corresponde a y.
        base_path (str): Path base en donde se encuentran los archivos.
        x_name (str): Nombre específico del archivo de X.
        y_name (str): Nombre específico del archivo de y.
        index (bool): Indica si se debe utilizar índices al grabar.
            Default to False.
    """
    wr.s3.to_csv(df=X,
                 path=os.path.join(base_path, x_name),
                 index=index)

    wr.s3.to_csv(df=y,
                 path=os.path.join(base_path, y_name),
                 index=index)
