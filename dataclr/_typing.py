from typing import TypedDict

import pandas as pd


class DataSplits(TypedDict):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
