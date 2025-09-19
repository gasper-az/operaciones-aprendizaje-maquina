import numpy as np
import pandas as pd


def metricas_iqr(data: pd.Series) -> tuple[float, float, float, float, float, float]:
    """
    Calcula y devuelve métricas de IQR sobre una serie de pandas, incluyendo
    Q1, Q2, Q3, IQR, valor mínimo del IQR, y valor máximo del IQR.

    Args:
        data (pd.Series): Serie sobre la cual se calcularán las métricas de IQR.
    
    Returns:
        tuple: métricas de IQR, incluyendo
        Q1, Q2, Q3, IQR, valor mínimo del IQR, y valor máximo del IQR.
    """
    Q1 = np.quantile(data, 0.25)
    Q2 = np.quantile(data, 0.50)
    Q3 = np.quantile(data, 0.75)

    IQR = Q3 - Q1
    min_iqr = Q1 - 1.5*IQR
    max_iqr = Q3 + 1.5*IQR

    return Q1, Q2, Q3, IQR, min_iqr, max_iqr

def obtener_metricas_columna(data: pd.Series) -> tuple[float, float, float, float, float,
                                                       float, int, int,float, float, float,
                                                       float, float, float, float]:
    """
    Calcula y devuelve varias métricas de una columna de un datafram dado,
    incluyendo media, moda, mediana, varianza, valor mínimo, valor máximo,
    cantidad de valores únicos, cantidad de ceros, Q1, Q2, Q3, IQR,
    valor mínimo del IQR, valor máximo del IQR, skewness y kurtosis.

    Args:
        data (pd.Series): Serie sobre la cual se calcularán las métricas.

    Returns:
        tuple: métricas calculadas a partir de los argumentos.
    """
    skewness = data.skew()
    kurtosis = data.kurt()
    media = data.mean()
    moda = data.mode()[0]
    mediana = data.median()
    var = data.std()
    min = data.min()
    max = data.max()
    nunique = data.nunique()
    mask_cant_ceros = data == 0
    cant_ceros = len(data[mask_cant_ceros])

    Q1, _, Q3, IQR, min_iqr, max_iqr = metricas_iqr(data)

    return media, moda, mediana, var, min, max, nunique, cant_ceros, Q1, Q3, IQR, min_iqr, max_iqr, skewness, kurtosis

def calcular_cantidad_outliers(data: pd.Series) -> tuple[int, int, int, float, float,
                                                         float, float, float, float, float]:
    """
    Calcula y devuelve la cantidad de outliers que tiene un serie de pandas.

    Args:
        data (pd.Series): Serie sobre la cual se calcularán las métricas.

    Returns:
        tuple: cantidad de outliers de la serie, junto con datos del IQR.
    """
    Q1, Q2, Q3, IQR, min_iqr, max_iqr = metricas_iqr(data)

    mascara_min_iqr = data < min_iqr
    mascara_max_iqr = data > max_iqr

    cant_outliers_izq = len(data[mascara_min_iqr])
    cant_outliers_der = len(data[mascara_max_iqr])
    total_outliers = cant_outliers_izq + cant_outliers_der

    return total_outliers, cant_outliers_izq, cant_outliers_der, Q1, Q2, Q3, IQR, min_iqr, max_iqr