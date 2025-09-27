import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple

def descargar_y_guardar_dataframe(url: str, path: str, index: bool = False):
    """
    Descarga un dataframe de una URL, y lo guarda en un path de filesystem.

    Args:
        url (str): URL de donde se descarga el dataframe.
        path (str): PATH en donde se guarda el csv.
        index (bool): indica si en el archivo final se deben guardar los
            índices. Default a False.
    """
    dataframe = pd.read_csv(url)

    dataframe.to_csv(path_or_buf=path, index=index)

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

def aplicar_transformaciones_inline(dataframe: pd.DataFrame, transformacion:str = "boxcox", columnas: list[str] = []):
    """
    Aplica transformaciones "inline" sobre un dataframe.

    Args:
        dataframe: Dataframe sobre el cual se aplicarán transformaciones a todas sus columnas.
        transformacion: Técnica a aplicar. Puede ser boxcox, sqrt o log1p. Default: boxcox.
        columnas: columnas a procesar. Default: empty list (se procesarán todas las columnas).
    Raises:
        ValueError: si el valor de 'transformacion' es erróneo.
    """
    if transformacion not in ["boxcox", "sqrt", "log1p"]:
            raise ValueError("Valor del parámetro 'transformacion' incorrecto. Debe ser 'boxcox', 'sqrt' o 'log1p'.")
    
    if columnas == []:
        columnas = dataframe.columns

    for col in columnas:
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
    Agrega la columna BMI inline.

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
                         prefix: str = "BMI") -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Aplica one hot encoding a las categorías de BMI en dataframes de train y test.
    NO es un método inline.

    Args:
        train: DataFrame de train.
        test: DataFrame de test.
        categoryCol: nombre de la columna con categorías de BMI. Default a "BMI_cat".
        prefix: prefijo de las nuevas columnas. Default a "BMI".
    Returns:
        tuple: nuevos dataframes de train y test, y la lista con las nuevas columnas.
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

    return train, test, bmi_cols

def codificar_dummy_feature(train: pd.DataFrame, test: pd.DataFrame,
                            categoryCols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Codifica dummy/ies feature/s.

    Args:
        train: DataFrame de train.
        test: DataFrame de test.
        categoryCols: lista de nombres de la columnas con categorías de BMI. Estas serán reescritas con los dummies.
    Returns:
        tuple: nuevos dataframes de train y test.
    Raises:
        ValueError: si categoryCol no existe los dataframes especificados.
    """
    for categoryCol in categoryCols:
        if categoryCol not in train.columns or categoryCol not in test.columns:
            raise ValueError(f"La columna {categoryCol} no existe en los dataframe especificados.")
        
    bmi_train = pd.get_dummies(train[categoryCols], dtype=int)

    bmi_cols = bmi_train.columns
    bmi_test  = pd.get_dummies(test[categoryCols], dtype=int).reindex(columns=bmi_cols, fill_value=0)

    train = pd.concat([train.drop(columns=categoryCols), bmi_train], axis=1)
    test = pd.concat([test.drop(columns=categoryCols),  bmi_test],  axis=1)

    return train, test

def escalar_features(train: pd.DataFrame, test: pd.DataFrame, scaler: str = "StandardScaler",  columnas: list[str] = []):
    """
    Escala las features del dataset de train y test.
    
    Args:
        train: DataFrame de train.
        test: DataFrame de test.
        scaler: Nombre del scaler a implementar. Puede ser 'StandardScaler' o 'MinMaxScaler'
        columnas: columnas a procesar. Default: empty list (se procesarán todas las columnas).
    Raises:
        ValueError: el valor de scaler es erróneo.
    """
    if scaler not in ["StandardScaler", "MinMaxScaler"]:
        raise ValueError(f"El valor {scaler} del parámetro 'scaler' es erróneo. Los posibles valores son StandardScaler, MinMaxScaler.")
    
    if columnas == []:
        columnas = train.columns

    match scaler:
        case "StandardScaler":
            scaler = StandardScaler()
        case _: # MinMaxScaler
            scaler = MinMaxScaler()
    
    train.loc[:, columnas] = np.float64(scaler.fit_transform(train[columnas]))
    test.loc[:, columnas] = np.float64(scaler.transform(test[columnas]))

def procesar_dataframe_completo(dataframe: pd.DataFrame, target: str, random_state: int,
                     test_size:float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Procesa todo el dataframe.

    Args:
        dataframe: Dataframe sobre el cual se realizará el procesamiento.
        target: Variable target.
        random_state: Random State, que nos permitirá obtener siempre el mismo resultado.
        test_size: Tamaño del set de test. Default a 0.3.
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = split_dataframe_train_test(dataframe=dataframe,
                                                                  target=target,
                                                                  random_state=random_state,
                                                                  test_size=test_size)
    
    # Nueva feature: BMI + BMI_cat
    colBMI = "BMI"
    colBMI_cat = "BMI_cat"

    agregar_bmi(X_train, bmiCol=colBMI)
    categorizar_bmi(X_train, bmiCol=colBMI, categoryCol=colBMI_cat)

    agregar_bmi(X_test, bmiCol=colBMI)
    categorizar_bmi(X_test, bmiCol=colBMI, categoryCol=colBMI_cat)

    cols_a_transformar = list(set(X_train.columns) - set([colBMI_cat]))
    # Tratamiento de outliers con transformación Boxcox.
    # Aplica solo a las columna numéricas.
    aplicar_transformaciones_inline(dataframe=X_train, columnas=cols_a_transformar)
    aplicar_transformaciones_inline(dataframe=X_test, columnas=cols_a_transformar)

    X_train, X_test, new_cols_bmi = one_hot_encoding_bmi(train=X_train,
                                                         test=X_test,
                                                         categoryCol=colBMI_cat,
                                                         prefix=colBMI)

    X_train, X_test = codificar_dummy_feature(train=X_train, test=X_test,
                                              categoryCols=new_cols_bmi)
    
    cols_a_escalar = list(set(X_train.columns) - set(new_cols_bmi))
    escalar_features(train=X_train, test=X_test,
                     scaler="StandardScaler", columnas=cols_a_escalar)

    return X_train, X_test, y_train, y_test

def separar_columna_dataset(dataset: pd.DataFrame, columna: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa una columna de un dataset.

    Args:
        dataset (pd.DataFrame): Dataser sobre el cual trabajar.
        columna (str): Nombre de la columna a separar.
    Returns:
        Tuple: pd.DataFrame, pd.Series.
    """
    col_data = dataset[columna]
    dataset = dataset.drop(columns=[columna])

    return dataset, col_data

def cargar_train_test_from_files(train_path: str, test_path: str, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Lee archivos correspondientes a dataframes de train y test.

    Args:
        train_path (str): Path al dataset de train.
        test_path (str): Path al dataset de test.
        target (str): Columna correspondiente al target.
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = separar_columna_dataset(dataset=train, columna=target)
    X_test, y_test = separar_columna_dataset(dataset=test, columna=target)

    return X_train, X_test, y_train, y_test

def guardar_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, train_path: str, test_path: str, index: bool = False):
    """
    Guarda los datos de X_train, X_test, y_train e y_test en dos archivos distintos.

    Args:
        X_train (pd.DataFrame): Features de train.
        X_test (pd.DataFrame): Features de test.
        y_train (pd.Series): Objetivo de train.
        y_test (pd.Series): Objetivo de test.
        train_path (str): Archivo donde se guarda el train.
        test_path (str): Archivo donde se guarda el test.
        index (bool): Indica si los archivos finales incluyen índixes. Default a False.
    """
    train = X_train.join(y_train)
    test = X_test.join(y_test)

    train.to_csv(train_path, index=index)
    test.to_csv(test_path, index=index)    