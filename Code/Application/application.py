import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import log_loss, f1_score, confusion_matrix, accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef
import seaborn as sns

mlflow.set_tracking_uri("sqlite:///Code/Logistic Regression/mlruns/mlruns.db")

experiment_name = 'kobe_shots'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

cols = ['lat','lon','minutes_remaining', 'period','playoffs','shot_distance']

with mlflow.start_run(experiment_id=experiment_id, run_name='PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe_lr@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('./Data/Raw/dataset_kobe_prod.parquet')

    data_prod = data_prod.dropna().reset_index(drop=True).drop_duplicates()

    size_production_data = data_prod.shape[0]
    print(f"Tamanho da Base de Produção: {size_production_data} amostras")


    Y = loaded_model.predict_proba(data_prod[cols])[:,1]
    data_prod['predict_score'] = Y
    true_labels = data_prod['shot_made_flag']

    # Calcular e plotar a matriz de confusão
    conf_matrix = confusion_matrix(true_labels, np.round(Y))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig('./Code/Application/Plots/confusion_matrix.jpg')
    plt.close()

    # Calcula as métricas de avaliação do modelo na base de produção
    accuracy = accuracy_score(true_labels, np.round(Y))
    auc = roc_auc_score(true_labels, Y)
    recall = recall_score(true_labels, np.round(Y))
    precision = precision_score(true_labels, np.round(Y))
    f1 = f1_score(true_labels, np.round(Y))
    kappa = cohen_kappa_score(true_labels, np.round(Y))
    mcc = matthews_corrcoef(true_labels, np.round(Y))

    # Imprime as métricas de avaliação do modelo na base de produção
    print("Métricas de Avaliação do Modelo na Base de Produção:")
    print(f"Acurácia: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1 Score: {f1}")
    print(f"Kappa: {kappa}")
    print(f"MCC: {mcc}")

    # Imprimir desempenho do modelo
    log_loss_value = log_loss(true_labels, Y)
    f1_score_value = f1_score(true_labels, np.round(Y))
    print(f'Log Loss: {log_loss_value}')
    print(f'F1 Score: {f1_score_value}')

    if len(true_labels) == 0:
        print("Todos os valores verdadeiros são NaN, não é possível calcular as métricas.")
    else:
        data_prod.to_parquet('./Data/Processed/prediction_prod.parquet')
        mlflow.log_artifact('./Data/Processed/prediction_prod.parquet')

    mlflow.log_metrics({
        'log_loss_app': log_loss_value,
        'f1_app': f1_score_value
    })