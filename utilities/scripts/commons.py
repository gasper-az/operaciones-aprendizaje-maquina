import pandas as pd

def separar_columna_dataset(dataset: pd.DataFrame, columna: str):
    """
    Separa columnas de un dataset.

    Args:
        dataset (pd.DataFrame): dataset del cual se obtendrán datos de una columna.
        columna (str): nombre de la columna a separar.

    Returns:
        (dataset, col_data): dataset original + datos de la columna determinada.
    """
    col_data = dataset[columna]
    dataset = dataset.drop(columns=[columna])

    return dataset, col_data

def cargar_train_test_from_files(train_path: str, test_path: str, target: str):
    """
    Carga los datasets de train y test.

    Args:
        train_path (str): Path donde se encuentra el dataset de train.
        test_path (str): Path donde se encuentra el dataset de test.
        target (str): Target column, de ambos datasets.

    Returns:
        X_train, X_test, y_train, y_test: datasets de train y test,
            dividios en `X` e `y`, siendo `y` la columna target y `X`
            las features del dataset.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = separar_columna_dataset(dataset=train, columna=target)
    X_test, y_test = separar_columna_dataset(dataset=test, columna=target)

    return X_train, X_test, y_train, y_test

def guardar_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                       y_test: pd.Series, train_path: str, test_path: str, index: bool = False):
    """
    Guarda los dataset de train y test, dados sus `X` e `y`.

    Args:
        X_train (pd.DataFrame): features del dataset de train.
        X_test (pd.DataFrame): features del dataset de test.
        y_train (pd.Series): target del dataset de train.
        y_test (pd.Series): target del dataset de test.
        train_path (str): Path donde se guardará el dataset de train.
        test_path (str): Path donde se guardará el dataset de test.
        index (bool): índica si al guardar se deben utilizar índices en los
            datasets o no. Default a False.
    """
    train = X_train.join(y_train)
    test = X_test.join(y_test)

    train.to_csv(train_path, index=index)
    test.to_csv(test_path, index=index)