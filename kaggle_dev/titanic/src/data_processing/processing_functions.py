import pandas as pd


def title_to_num(row: pd.Series) -> int:
    title: str = row['Title']
    if title == 'Master.':
        return 0
    elif title == 'Miss.':
        return 1
    elif title == 'Mr.':
        return 2
    elif title == 'Mrs.':
        return 3
    return 4

def encord_to_label(col: pd.Series) -> pd.Series:
    labels, _ = pd.factorize(col)
    return labels