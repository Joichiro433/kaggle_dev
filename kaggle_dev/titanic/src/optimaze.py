"""
ハイパーパラメータのチューニングを行う
"""

from typing import Dict
from nptyping import NDArray

import numpy as np
from nptyping import NDArray, Shape, Float, Int
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from rich import print

import params
from utils import save_pkl
from data_processing.preprocessing import preprocess_dataset


class Objective:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def __call__(self, trial) -> float:
        max_depth_limit = 10
        lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_leaves': trial.suggest_int('num_leaves', 2**6, 2**(max_depth_limit-1)),
            'max_depth': trial.suggest_int('max_depth', 6, max_depth_limit),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 40),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        }
        X_tr, X_val, y_tr, y_val = train_test_split(self.X, self.y, test_size=0.25, shuffle=True, stratify=self.y, random_state=42)
        model = lgb.LGBMRegressor(**lgbm_params, random_state=42)
        model.fit(X=X_tr, y=y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], early_stopping_rounds=100)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, np.rint(y_pred))
        return accuracy


if __name__ == '__main__':
    df: pd.DataFrame = pd.read_pickle(params.TRAINING_DATA)
    df = preprocess_dataset(df=df)
    X: NDArray[Shape['Sample, Features'], Float] = df.drop(params.TARGET, axis=1).values
    y: NDArray[Shape['Sample'], Float] = df[params.TARGET].values

    objective = Objective(X=X, y=y)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_params: Dict[str, float] = study.best_trial.params
    print('Number of finished trials:', len(study.trials))
    print('Best trial: ')
    print(best_params)

    save_pkl(best_params, path=params.OUTPUT_DIR/'best_params.pkl')
