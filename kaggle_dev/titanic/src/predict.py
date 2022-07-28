import os

import numpy as np
from nptyping import NDArray, Shape, Float, Int
import pandas as pd
from rich import print

import params
from utils import read_pkl
from data_processing.preprocessing import preprocess_dataset


def predict(X: NDArray[Shape['Sample, Freatures'], Float]) -> NDArray[Shape['Sample, Class'], Float]:
    model_path = params.OUTPUT_DIR/'trained_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError('trained model is not exist')
    model = read_pkl(model_path)
    pred_proba: NDArray[Shape['Sample, Class'], Float] = model.predict_proba(X)
    return pred_proba


def predict_crossval(X: NDArray[Shape['Sample, Freatures'], Float]) -> NDArray[Shape['Sample, Class'], Float]:
    model_paths = params.OUTPUT_DIR.glob('trained_model_*fold.pkl')
    models = [read_pkl(path) for path in model_paths]
    pred_probas = np.array([model.predict_proba(X) for model in models])
    return pred_probas.mean(axis=0)


if __name__ == '__main__':
    # df_test = pd.read_csv(params.TEST_DATA)
    df_test = pd.read_pickle(params.TEST_DATA)
    df_test = preprocess_dataset(df_test)
    # pred_proba = predict(X=df_test.values)
    pred_proba = predict_crossval(X=df_test.values)

    df_gender_submission: pd.DataFrame = pd.read_csv(params.DATA_DIR/'gender_submission.csv')
    df_gender_submission['Survived'] = np.argmax(pred_proba, axis=1)
    df_gender_submission.to_csv(params.OUTPUT_DIR/'my_pred.csv', index=False)
    