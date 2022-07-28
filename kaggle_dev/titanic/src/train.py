import os

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
from rich import print

import params
from utils import read_pkl, save_pkl
from trainer import LGBMTrainer
from data_processing.preprocessing import preprocess_dataset


def train(df_train):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'n_estimators': params.NUM_ITERATES,
        'random_state': 42}
    if os.path.exists(params.OUTPUT_DIR/'best_params.pkl'):
        best_params = read_pkl(params.OUTPUT_DIR/'best_params.pkl')
    else:
        best_params = {}
        
    model = lgb.LGBMClassifier(**lgbm_params, **best_params)
    lgbmtrainer = LGBMTrainer(
        estimator=model,
        matrics=[accuracy_score, log_loss], 
        df=df_train, 
        target=params.TARGET,
        cross_val=False)
    lgbmtrainer.train()
    trained_model = lgbmtrainer.get_model()
    save_pkl(trained_model, params.OUTPUT_DIR/'trained_model.pkl')


def train_crossval(df_train):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'random_state': 42}
    if os.path.exists(params.OUTPUT_DIR/'best_params.pkl'):
        best_params = read_pkl(params.OUTPUT_DIR/'best_params.pkl')
    else:
        best_params = {}

    model = lgb.LGBMClassifier(**lgbm_params, **best_params)
    lgbmtrainer = LGBMTrainer(
        estimator=model,
        matrics=[accuracy_score, log_loss], 
        df=df_train, 
        target=params.TARGET,
        cross_val=True)
    lgbmtrainer.train()
    df_matric: pd.DataFrame = lgbmtrainer.get_matrics()
    print(df_matric)
    df_matric.to_csv(params.OUTPUT_DIR/'crossval_matrics.csv')
    

if __name__ == '__main__':
    # df_train: pd.DataFrame = pd.read_csv(params.TRAINING_DATA)
    df_train: pd.DataFrame = pd.read_pickle(params.TRAINING_DATA)
    df_train = preprocess_dataset(df=df_train)

    # train(df_train=df_train)
    train_crossval(df_train=df_train)
