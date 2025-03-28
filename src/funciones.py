import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd

def relacion_num_target(Dataframe, lista):
    n = len(lista)
    # Se crean n filas y 2 columnas de subgráficas
    fig, axes = plt.subplots(n, 2, figsize=(7, 3*n))
    
    # En caso de que solo se ingrese una columna, forzamos que axes sea 2D
    if n == 1:
        axes = np.array([axes])
    
    for i, col in enumerate(lista):
        # Lineplot en la primera columna de la fila i
        sns.lineplot(data=Dataframe, x='target', y=col, ax=axes[i, 0])
        axes[i, 0].set_title(f'Lineplot: {col}')
        
        # Heatmap de la correlación entre la columna y "target" en la segunda columna de la fila i
        corr_matrix = Dataframe[[col, "target"]].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=axes[i, 1], cbar=False)
        axes[i, 1].set_title(f'Correlación: {col} vs target')
    
    plt.tight_layout()
    plt.show()


def funciones_num(Dataframe, lista=[]):

    n_cols = len(Dataframe.columns)
    n_grid = int(n_cols**0.5) + 1

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(30, 30))
    axes = axes.flatten()

    for id, column in enumerate(lista):
        sns.histplot(ax=axes[id], data = Dataframe, x = Dataframe[column])
        axes[id].set_title(column)
    
    for j in range(id + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def analisis_outliers(Dataframe, lista=[]):

    n_cols = len(Dataframe.columns)
    n_grid = int(n_cols**0.5) + 1

    fig, axes = plt.subplots(n_grid, n_grid, figsize=(30, 30))
    axes = axes.flatten()

    for id, column in enumerate(lista):
        sns.boxplot(ax=axes[id], data = Dataframe, x = Dataframe[column])
        axes[id].set_title(column)
    
    for j in range(id + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


# Funcion para reemplazar los outliers de mis columnas
def reemplazar_outliers(column, df):
    stats = df[column].describe()
    iqr = stats["75%"] - stats["25%"]
    upper_limit = stats["75%"] + 1.5 * iqr
    lower_limit = stats["25%"] - 1.5 * iqr
    if lower_limit < 0:
        lower_limit = df[column].min()
    df[column] = df[column].apply(lambda x: x if x <= upper_limit else upper_limit)
    df[column] = df[column].apply(lambda x: x if x >= lower_limit else lower_limit)
    return df.copy(), [lower_limit, upper_limit]




def impute_zeros(df, cols_with_zero=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']):
    """
    Imputa los valores 0 en las columnas especificadas por la mediana de los valores no cero.
    
    Parámetros:
    - df: DataFrame de pandas con los datos.
    - cols_with_zero: Lista de columnas donde se imputará el valor 0 (default: las columnas relevantes en el dataset de diabetes).
    
    Retorna:
    - df: DataFrame con los valores 0 imputados.
    """
    
    # Mostrar el número de ceros antes de la imputación
    print("Número de ceros antes de la imputación:")
    for col in cols_with_zero:
        zeros = np.sum(df[col] == 0)
        print(f"{col}: {zeros}")
    
    # Imputar cada columna reemplazando los 0 por la mediana de los valores distintos de 0
    for col in cols_with_zero:
        median_val = df.loc[df[col] != 0, col].median()
        df.loc[df[col] == 0, col] = median_val
    
    # Mostrar el número de ceros después de la imputación
    print("\nNúmero de ceros después de la imputación:")
    for col in cols_with_zero:
        zeros = np.sum(df[col] == 0)
        print(f"{col}: {zeros}")
    
    return df







# Creo la funcion para que me haga la transformacion
def onehot_encode_and_pickle(df, cat_vars, encoder_filename='encoder.pkl', drop_original=True, drop_first=True):
    """
    Aplica one-hot encoding a las columnas categóricas especificadas, guarda el encoder usando pickle 
    y retorna el DataFrame transformado junto con el encoder entrenado.
    
    Args:
        df (pd.DataFrame): DataFrame de entrada.
        cat_vars (list): Lista de nombres de columnas categóricas a transformar.
        encoder_filename (str): Ruta y nombre del archivo para guardar el encoder (por defecto 'encoder.pkl').
        drop_original (bool): Si True, elimina las columnas originales después de la transformación (por defecto True).
        drop_first (bool): Si True, elimina la primera categoría de cada variable para evitar la trampa de las dummies (por defecto True).
        
    Returns:
        df_transformed (pd.DataFrame): DataFrame con las nuevas columnas codificadas.
        encoder (OneHotEncoder): Objeto encoder entrenado.
    """
    # Inicializamos el OneHotEncoder usando sparse_output en vez de sparse
    encoder = OneHotEncoder(drop='first' if drop_first else None, sparse_output=False)
    
    # Ajustamos el encoder y transformamos las columnas categóricas
    encoded_array = encoder.fit_transform(df[cat_vars])
    
    # Obtenemos los nombres de las nuevas columnas codificadas
    encoded_feature_names = encoder.get_feature_names_out(cat_vars)
    
    # Creamos un DataFrame con los datos codificados
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)
    
    # Concatenamos el DataFrame original con el DataFrame de variables codificadas
    df_transformed = pd.concat([df, df_encoded], axis=1)
    
    # Eliminamos las columnas originales si se indica
    if drop_original:
        df_transformed.drop(columns=cat_vars, inplace=True)
    
    # Guardamos el objeto encoder utilizando pickle
    with open(encoder_filename, 'wb') as f:
        pickle.dump(encoder, f)
    
    return df_transformed, encoder