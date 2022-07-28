from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict
from copy import deepcopy

import numpy as np
from nptyping import NDArray, Shape, Float, Int
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import shap

import params
from utils import save_pkl

sns.set_style('whitegrid')


class LGBMTrainer:
    def __init__(
            self,
            estimator: lgb.LGBMModel,
            matrics: List[Callable],
            df: pd.DataFrame, 
            target: str,
            features: Optional[List[str]] = None,
            cross_val: bool = False) -> None:
        self.estimator = estimator
        self.features: List[str] = df.drop(target, axis=1).columns if features is None else features
        self.X: NDArray[Shape['Sample, Features'], Float] = df[self.features].values
        self.y: NDArray[Shape['Sample'], Float] = df[target].values
        self.y_pred: NDArray[Shape['Sample'], Float] = np.zeros_like(self.y)
        self.matrics: List[Callable] = matrics
        self.matrics_results: Dict[str, List[float]] = defaultdict(list)
        self.cross_val: bool = cross_val
        self.early_stopping_rounds: int = 100
        self._is_trained: bool = False
        self._shap_values_list = []
        self._val_indices_list = []

    def get_y_pred(self) -> pd.Series:
        if not self._is_trained:
            raise Exception('Untrained!')
        return pd.Series(self.y_pred)

    def get_matrics(self) -> pd.DataFrame:
        if not self._is_trained:
            raise Exception('Untrained!')
        return pd.DataFrame(self.matrics_results)

    def get_model(self) -> lgb.LGBMModel:
        if not self._is_trained:
            raise Exception('Untrained!')
        return self.estimator

    def train(self, n_splits: int = params.NUM_SPLITS) -> None:
        if self.cross_val:    
            cv_indices = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(self.X, self.y)
            for nfold, (train_idx, val_idx) in enumerate(cv_indices):
                self.estimator.random_state = nfold
                self._train_crossval_step(train_idx, val_idx)
                lgb.plot_metric(self.estimator, title=f'logloss during training ({nfold+1}fold)')
                plt.savefig(params.OUTPUT_DIR/f'loss_{nfold+1}.pdf')
                plt.close()
                save_pkl(self.estimator, params.OUTPUT_DIR/f'trained_model_{nfold}fold.pkl')
            self._is_trained = True
            self._save_shap_result()
        else:
            self.estimator.fit(X=self.X, y=self.y)
            y_train_pred = self.estimator.predict(self.X)
            for matric in self.matrics:
                self.matrics_results[f'train_{matric.__name__}'].append(matric(self.y, y_train_pred))
            self.y_pred = y_train_pred.copy()
            self._is_trained = True

    def _train_crossval_step(self, train_idx: List[int], val_idx: List[int]) -> None:
        X_train, y_train = self.X[train_idx], self.y[train_idx]
        X_val, y_val = self.X[val_idx], self.y[val_idx]
        self.estimator.fit(
            X=X_train, 
            y=y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=self.early_stopping_rounds)
        y_train_pred = self.estimator.predict(X_train)
        y_val_pred = self.estimator.predict(X_val)
        for matric in self.matrics:
            self.matrics_results[f'train_{matric.__name__}'].append(matric(y_train, y_train_pred))
            self.matrics_results[f'val_{matric.__name__}'].append(matric(y_val, y_val_pred))
        self.matrics_results['best_iteration'].append(self.estimator.best_iteration_)
        self.y_pred[val_idx] = y_val_pred
        # Shap値の計算
        explainer = shap.TreeExplainer(self.estimator)
        shap_values = explainer.shap_values(X_val)[0]
        self._shap_values_list.append(shap_values)
        self._val_indices_list.append(val_idx)

    def _save_shap_result(self) -> None:
        shap_values = np.concatenate(self._shap_values_list)
        val_indices = np.concatenate(self._val_indices_list)
        X_val = pd.DataFrame(self.X[val_indices], columns=self.features)
        shap.summary_plot(shap_values, X_val, alpha=0.9, show=False)
        plt.savefig(params.OUTPUT_DIR/f'shap_summary.pdf', format='pdf', dpi=1200, bbox_inches='tight')
        plt.close()
        shap.summary_plot(shap_values, X_val, plot_type='bar', show=False)
        plt.savefig(params.OUTPUT_DIR/f'shap_summary_bar.pdf', format='pdf', dpi=1200, bbox_inches='tight')
        plt.close()
