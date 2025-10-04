import pandas as pd

def separar_columna_dataset(dataset: pd.DataFrame, columna: str):
    """
    """
    col_data = dataset[columna]
    dataset = dataset.drop(columns=[columna])

    return dataset, col_data

def cargar_train_test_from_files(train_path: str, test_path: str, target: str):
    """
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train, y_train = separar_columna_dataset(dataset=train, columna=target)
    X_test, y_test = separar_columna_dataset(dataset=test, columna=target)

    return X_train, X_test, y_train, y_test

def guardar_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, train_path: str, test_path: str, index: bool = False):
    """
    """
    train = X_train.join(y_train)
    test = X_test.join(y_test)

    train.to_csv(train_path, index=index)
    test.to_csv(test_path, index=index)