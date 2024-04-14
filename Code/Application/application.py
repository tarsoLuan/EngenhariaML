import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, validation_curve
import pycaret.classification as pc
import os
from sklearn.metrics import log_loss, f1_score

mlflow.set_tracking_uri("sqlite:///Code/Logistic Regression/mlruns/mlruns.db")

experiment_name = 'kobe_shots'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

cols = ['lat','lon','minutes_remaining', 'period','playoffs','shot_distance']

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe_lr@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    data_prod = pd.read_parquet('./Data/Raw/dataset_kobe_prod.parquet')

    data_prod = data_prod.dropna()
    data_prod = data_prod.reset_index(drop=True)
    data_prod = data_prod.drop_duplicates()


    Y = loaded_model.predict_proba(data_prod[cols])[:,1]
    data_prod['predict_score'] = Y
    true_labels = data_prod['shot_made_flag']


    if len(true_labels) == 0:
        print("Todos os valores verdadeiros são NaN, não é possível calcular as métricas.")
    else:
        data_prod.to_parquet('./Data/Processed/prediction_prod.parquet')
        mlflow.log_artifact('./Data/Processed/prediction_prod.parquet')

    mlflow.log_metrics({
        'log_loss_app': log_loss(true_labels, Y),
        'f1_app': f1_score(true_labels, np.round(Y))
    })