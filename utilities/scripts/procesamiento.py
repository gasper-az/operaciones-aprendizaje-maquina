import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats

def split_dataframe_train_test(dataframe: pd.DataFrame, target: str, random_state: int, test_size:float = 0.3) -> tuple:
    """
    Realiza el split de un dataframe en train y test.

    Args:
        dataframe: Dataframe sobre el cual se realizará el split.
        target: Variable target.
        random_state: Random State, que nos permitirá obtener siempre el mismo resultado.
        test_size: Tamaño del set de test. Default a 0.3.
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=target),
        dataframe[target],
        random_state=random_state,
        test_size=test_size)

    return X_train, X_test, y_train, y_test

def imputar_ceros_con_metrica(data: pd.Series, metrica: str = "mean") -> pd.Series:
    """
    Imputa los ceros de una serie.

    Args:
        data: Serie sobre la cual se imputarán los ceros.
        metrica: Metrica de imputación. Puede ser mean, median o mode. Default: mean.
    Returns:
        Serie imputada.
    Raises:
        ValueError: si el valor de 'metrica' es erróneo.
    """
    valor_metrica = 0

    match metrica:
        case "mean":
            valor_metrica = data.mean()
        case "median":
            valor_metrica = data.median()
        case "mode":
            valor_metrica = data.mode()[0]
        case _:
            raise ValueError("Valor del parámetro 'metrica' incorrecto. Debe ser 'mean', 'median' o 'mode'.")
    
    return data.replace(0, valor_metrica)

def aplicar_transformaciones_inline(dataframe: pd.DataFrame, transformacion:str = "boxcox"):
    """
    Aplica transformaciones "inline" sobre un dataframe.

    Args:
        dataframe: Dataframe sobre el cual se aplicarán transformaciones a todas sus columnas.
        transformacion: Técnica a aplicar. Puede ser boxcox, sqrt o log1p. Default: boxcox.
    Raises:
        ValueError: si el valor de 'transformacion' es erróneo.
    """
    if transformacion not in ["boxcox", "sqrt", "log1p"]:
            raise ValueError("Valor del parámetro 'transformacion' incorrecto. Debe ser 'boxcox', 'sqrt' o 'log1p'.")
    
    for col in dataframe.columns:
        match transformacion:
            case "boxcox":
                dataframe[col], _ = stats.boxcox(dataframe[col])
            case "sqrt":
                dataframe[col] = np.sqrt(dataframe[col])
            case _: # log1p
                dataframe[col] = np.log1p(dataframe[col])

def agregar_bmi(dataframe: pd.DataFrame, heightCol: str = "Height", weightCol: str = "Weight", bmiCol:str = "BMI"):
    """
    Calcula el BMI a un dataframe dado, en función de las columnas heightCol y weightCol.
    Agrega la columna BMI inline

    Args:
        dataframe: Dataframe sobre el cual se trabajará.
        heightCol: nombre de la columna correspondiente al Height. Default a 'Height'.
        weightCol: nombre de la columna correspondiente al Weight. Default a 'Weight'.
        bmiCol: nombre de la nueva columna. Default a "BMI".
    Raises:
        ValueError: si heightCol o weightCol no existen en el dataframe.
    """
    if heightCol not in dataframe.columns:
        raise ValueError(f"La columna {heightCol} no existe en el dataframe especificado.")
    
    if weightCol not in dataframe.columns:
        raise ValueError(f"La columna {weightCol} no existe en el dataframe especificado.")
    
    h = pd.to_numeric(dataframe[heightCol], errors="coerce")
    w = pd.to_numeric(dataframe[weightCol], errors="coerce")
    h = h.where(h > 0, np.nan)
    dataframe[bmiCol] = 703 * w / (h ** 2)

def categorizar_bmi(dataframe: pd.DataFrame, bmiCol:str = "BMI", categoryCol: str = "BMI_cat"):
    """
    Categoriza la columna BMI de un dataframe, según definición de OMS, inline.

    Args:
        dataframe: Dataframe sobre el cual se trabajará.
        bmiCol: nombre de la columna con datos de BMI. Default a "BMI".
        categoryCol: nombre de la nueva columna. Default a "BMI_cat".
    Raises:
        ValueError: si bmiCol no existe en el dataframe.
    """
    if bmiCol not in dataframe.columns:
        raise ValueError(f"La columna {bmiCol} no existe en el dataframe especificado.")
        
    bins = [-np.inf, 18.5, 25, 30, np.inf]
    labels = ["bajo_peso", "normal", "sobrepeso", "obesidad"]
    dataframe[categoryCol] = pd.cut(dataframe[bmiCol], bins=bins, labels=labels, right=False)

def one_hot_encoding_bmi(train: pd.DataFrame, test: pd.DataFrame, categoryCol: str = "BMI_cat",
                         prefix: str = "BMI") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica one hot encoding a las categorías de BMI en dataframes de train y test.
    NO es un método inline.

    Args:
        train: DataFrame de train.
        test: DataFrame de test.
        categoryCol: nombre de la columna con categorías de BMI. Default a "BMI_cat".
        prefix: prefijo de las nuevas columnas. Default a "BMI".
    Returns:
        tuple: nuevos dataframes de train y test.
    Raises:
        ValueError: si categoryCol no existe los dataframes especificados.
    """
    if categoryCol not in train.columns or categoryCol not in test.columns:
        raise ValueError(f"La columna {categoryCol} no existe en los dataframe especificados.")
    
    bmi_train = pd.get_dummies(train[categoryCol], prefix=prefix, dtype=int)

    bmi_cols = bmi_train.columns
    bmi_test  = pd.get_dummies(test[categoryCol],  prefix=prefix, dtype=int).reindex(columns=bmi_cols, fill_value=0)

    train = pd.concat([train.drop(columns=[categoryCol]), bmi_train], axis=1)
    test  = pd.concat([test.drop(columns=[categoryCol]),  bmi_test],  axis=1)

    return train, test