
from model_training import modeling
from evaluation import model_evaluation, ROC_curve, Conf_Mtx
import mlflow
import mlflow.sklearn


def mlflow_func(X,y):
    # Iniciar un experimento en MLflow
    mlflow.set_experiment("Breast Cancer Wisconsin")

    with mlflow.start_run():
        # Entrenar un modelo RandomForest
        X_train, X_test, y_train, y_test, model, model_type = modeling(X,y,'random_forest')

        # Evaluar el modelo
        accuracy, report, y_pred = model_evaluation(X_test, y_test, model)

        # Registrar métricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(report, "classification_report.txt")

        # Calcular y grafica la curva ROC
        ROC_curve(X_test,y_test,model)

        # Registrar la curva ROC como una imagen
        mlflow.log_artifact("roc_curve.png")

        # Calcular y mostrar la matriz de confusión
        Conf_Mtx(y_test,y_pred)

        # Registrar la matriz de confusión como una imagen
        mlflow.log_artifact("confusion_matrix.png")

        # Registrar el modelo
        mlflow.sklearn.log_model(model, model_type)

        # Imprimir resultados
        print(f'Precisión del modelo: {accuracy:.2f}')
        print('Reporte de Clasificación:')
        print(report)
