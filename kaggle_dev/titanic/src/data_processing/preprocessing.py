import pandas as pd
import swifter

from data_processing.processing_functions import title_to_num, encord_to_label


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categories = ['Sex', 'Embarked', 'Title']
    df[categories] = df[categories].swifter.apply(encord_to_label, axis=0)
    return df
