import pandas as pd
import swifter

from data_processing.processing_functions import title_to_num, encord_to_label


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # df['Title'] = df['Name'].str.extract('([A-Za-z]+\.)', expand=False)
    # df['Title'].replace(['Mlle.', 'Ms.'], 'Miss.', inplace=True)
    # df['Title_num'] = df.swifter.apply(title_to_num, axis=1)
    # df['TicketFreq'] = df.groupby(['Ticket'])['PassengerId'].transform('count')
    # categories = ['Sex', 'Embarked']
    categories = ['Sex', 'Embarked', 'Title', 'Cabin_label']
    df[categories] = df[categories].swifter.apply(encord_to_label, axis=0)
    # df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'Title'], axis=1, inplace=True)
    return df
