import pandas as pd
from preprocessing import preprocess_data
from mlops_pipeline import mlflow_func

# Cargar el archivo
file_path = "./data/breast-cancer-wisconsin.data.csv"
df = pd.read_csv(file_path)
columns_to_delete = ["id", "Unnamed: 32"]
columns_to_encode = ['diagnosis']
dependent_variable = ['diagnosis']

# Hacemos el procesamiento de los datos
X,y = preprocess_data(df,columns_to_delete, columns_to_encode, dependent_variable)

#Corremos MLOps
mlflow_func(X,y)
