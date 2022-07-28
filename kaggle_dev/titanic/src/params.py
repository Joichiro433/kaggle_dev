import os
from pathlib import Path


DATA_DIR = Path()/'data'
# TRAINING_DATA = DATA_DIR/'train.csv'
TRAINING_DATA = DATA_DIR/'dataframes'/'train.pkl'
# TEST_DATA = DATA_DIR/'test.csv'
TEST_DATA = DATA_DIR/'dataframes'/'test.pkl'
OUTPUT_DIR = Path()/'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = 'Survived'  # 目的変数

NUM_SPLITS = 5  # 交差検証の分割数
# NUM_ITERATES = 300
NUM_ITERATES = 350