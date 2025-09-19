import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_histogramas(data: pd.DataFrame, columns: list[str], figsize=(5,5), kde=True,
                     ncols: int=3, colores: list[str] = ["gray", "blue", "red"]):
    """
    Grafica un histograma para cada una de las columnas especificadas.

    Args:
        data: Dataframe cuyas columnas serán graficadas.
        columns: Columnas del dataset a graficas.
        figsize: tamaño total del gráfico. Default: (5,5).
        kde: indica si se debe graficar la distribución de las
            observaciones. Default: True.
        ncols: cantidad de columnas a graficar por fila. Default: 3.
        colores: lista de colores a utilizar en los gráficos.
    """
    nrows = math.floor(len(columns)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, col in enumerate(columns):
        sns.histplot(data[col], kde=kde, ax=axes[math.floor(i/ncols), i%ncols], color=colores[i%ncols])
    
    plt.tight_layout()
    plt.show()

    return

def plot_boxplot(data: pd.DataFrame, columns: list[str], figsize=(5,5), ncols: int=3,
                 colores: list[str] = ["gray", "lightblue", "green"]):
    """
    Grafica un boxplot para cada una de las columnas especificadas.

    Args:
        data: Dataframe cuyas columnas serán graficadas.
        columns: Columnas del dataset a graficas.
        figsize: tamaño total del gráfico. Default: (5,5).
        ncols: cantidad de columnas a graficar por fila. Default: 3.
        colores: lista de colores a utilizar en los gráficos.
    """
    nrows = math.floor(len(columns)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, col in enumerate(columns):
        sns.boxplot(x=data[col], ax=axes[math.floor(i/ncols), i%ncols], color=colores[i%ncols])

        axes[math.floor(i/ncols), i%ncols].set_xlabel(col)
        axes[math.floor(i/ncols), i%ncols].set_title(f"Boxplot para {col}")
    
    plt.tight_layout()
    plt.show()

    return

def plot_scatterplot(data: pd.DataFrame, columns: list[str], ycol: str, figsize=(5,5),
                     ncols: int=3, colores: list[str] = ["gray", "lightblue", "green"]):
    """
    Grafica un boxplot para cada una de las columnas especificadas en función de otra columna.

    Args:
        data: Dataframe cuyas columnas serán graficadas.
        columns: Columnas del dataset a graficas.
        ycol: Columna contra la cual se graficarán los scatterplot. Debe estar también en "columns".
        figsize: tamaño total del gráfico. Default: (5,5).
        ncols: cantidad de columnas a graficar por fila. Default: 3.
        colores: lista de colores a utilizar en los gráficos.
    """
    nrows = math.floor(len(columns)/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    columns = columns.drop(ycol)

    for i, col in enumerate(columns):
        sns.scatterplot(data=data, x=col, y=ycol, ax=axes[math.floor(i/ncols), i%ncols], color=colores[i%ncols])

        axes[math.floor(i/ncols), i%ncols].set_xlabel(col)
        axes[math.floor(i/ncols), i%ncols].set_ylabel(ycol)
        axes[math.floor(i/ncols), i%ncols].set_title(f"{ycol} en función de {col}")
    
    plt.tight_layout()
    plt.show()

    return

def plot_correlaciones(data: pd.DataFrame):
    """
    Grafica correlaciones de un dataframe dado.
    Fuente: https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    Args:
        data: Dataframe sobre el cual se calcularán las correlaciones y su gráfico.
    """
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    
def plot_residuals(modelo, X_test: pd.DataFrame, y_test: pd.Series, figsize=(6, 6),
                   xlabel: str="Valores predichos", ylabel: str="Residuos",
                   title: str="Residual Plot"):
    """
    Permite graficar los residuos de un modelo dado.

    Args:
        modelo: Modelo de ML sobre el cual se calcularán los outputs de X_test e y_test.
        X_test: Dataframe sobre el cual se realizarán estimaciones, utilizando el modelo.
        y_test: Output real, sobre el cual se compararán los resultados de predicciones.
        figsize: tamaño total del gráfico. Default: (6,6).
        xlabel: Label que se mostrará en el eje de X.
        ylabel: Label que se mostrará en el eje de Y.
        title: Título del gráfico.
    """
    y_pred = modelo.predict(X_test)
    residuals = y_test - y_pred
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Esta función permite graficar los true values contra sus predicciones, para un modelo dado
# Source: https://stackoverflow.com/a/65322979
def plot_true_vs_predicted(modelo, X_test: pd.DataFrame, y_test: pd.Series,
                           figsize=(6,6), title: str="True values vs Predicciones"):
    """
    Dado un modelo y un dataset, permite graficar la predicción obtenida vs los valores reales.

    Args:
        modelo: Modelo de ML sobre el cual se calcularán los outputs de X_test e y_test.
        X_test: Dataframe sobre el cual se realizarán estimaciones, utilizando el modelo.
        y_test: Output real, sobre el cual se compararán los resultados de predicciones.
        figsize: tamaño total del gráfico. Default: (6,6).
        title: Título del gráfico.
    """
    y_pred = modelo.predict(X_test)

    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_pred, c='crimson')

    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predicciones', fontsize=15)
    plt.title(title)
    plt.axis('equal')
    plt.show()