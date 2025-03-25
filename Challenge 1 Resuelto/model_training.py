
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def modeling(X,y,model_selector):
    # Dividir en conjuntos de entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if model_selector == 'random_forest': 
        # Entrenar un modelo RandomForest
        model = RandomForestClassifier(n_estimators=500, random_state=42)
        model_type = "random_forest_model"

    elif model_selector == 'log_regression':
        # Entrenar un modelo de Regresion Logistica
        model = LogisticRegression(max_iter=200)
        model_type = "logistic_regression"
    
    elif model_selector == 'SVM':
         # Entrenar un modelo de Máquina de Soporte Vectorial
        model = SVC(kernel='rbf', random_state=42, probability=True)
        model_type = "SVM"       
    
    elif model_selector == 'KNN':
        # Entrenar un modelo de k-Vecinos más cercanos
        model = KNeighborsClassifier(n_neighbors=5)
        model_type = "KNN"    
    
    model.fit(X_train,y_train)

    return  X_train, X_test, y_train, y_test, model, model_type
