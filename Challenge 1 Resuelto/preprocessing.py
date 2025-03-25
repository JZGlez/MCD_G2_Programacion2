# Importar librerías. 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Función para eliminar columnas a partir de una lista
def columns_deletion(df,column_list):
    # Guardas una copia del DataFrame Original
    df_clean = df.copy()

    # Borramos las columnas indicadas
    for column in column_list:
        df_clean = df_clean.drop(column, axis=1)

    return df_clean

# Función para separar las variables dependientes e independientes en dos variables X y Y
def variables_split (df, dependent_variables):
    # Guardas copia de df a procesar
    X = df.copy()
    y = pd.DataFrame()

    for variable in dependent_variables:
        y[variable] = X[variable].copy()
        X = X.drop(variable, axis=1)
    
    return X,y

# Función para codificar variables
def columns_encoding(df, columns_to_encode, encoding=1):
    df_scaled = df.copy()
    
    if encoding == 1:
        for column in columns_to_encode:
            df_scaled[column] = LabelEncoder().fit_transform(df_scaled[column])
    
    return df_scaled

# Función para escalar variables 
def data_scaler(df, columns_to_scale = 'all', transformation='standar'):
    
    df_scaled = df.copy()

    # Selecciona entre el tipo de escalar que se utilizará
    if transformation == 'standar':
        scaler = StandardScaler()
    elif transformation == 'minmax':
        scaler = MinMaxScaler()

    # Guarda las variables a escalar
    if columns_to_scale == 'all':
        col_to_scale = df.columns
    else:
        col_to_scale = columns_to_scale

    # Escalamos las variables
    for column in col_to_scale:
        df_scaled[column]=scaler.fit_transform(df[[column]])

    return df_scaled

# Función para revisar si el DataFrame está vacio.
def preprocess_data(df, columns_to_delete, columns_to_encode, dependent_variable):
    # Verificar si el DataFrame está vacío
    if df.empty:
        return "El DataFrame está vacío."
    else:
        # Mostrar las primeras 5 filas del DataFrame
        print(df.head())
        # Verificar el tipo de datos de cada columna
        print(df.info())
        df = columns_deletion(df,columns_to_delete)
        df = columns_encoding(df,columns_to_encode)
        X,y = variables_split(df, dependent_variable)
        X = data_scaler(X)

    return X,y
