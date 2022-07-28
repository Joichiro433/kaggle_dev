from typing import Any, Union
from pathlib import Path
import pickle


def save_pkl(obj: Any, path: Union[str, Path]) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def read_pkl(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
