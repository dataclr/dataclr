from typing import TypedDict

import numpy as np
import pandas as pd


class BaseModel:
    """
    Abstract base class for machine learning models.

    This class defines the interface that models must adhere to for compatibility with
    feature selection methods. Subclasses must implement the ``fit`` and ``predict``
    methods.

    Attributes for Wrapper Method Compatibility:
        - ``feature_importances_``: Attribute for feature importance scores
            (e.g., tree-based models).
        - ``coef_``: Attribute for feature coefficients (e.g., linear models).

    Subclasses must ensure that at least one of these attributes is implemented to
    support wrapper-based feature selection methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Abstract method to train the model.

        Args:
            X_train (pd.DataFrame): Feature matrix for training data.
            y_train (pd.Series): Target variable for training data.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError()

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Abstract method to generate predictions.

        Args:
            X_test (pd.DataFrame): Feature matrix for testing data.

        Returns:
            np.ndarray: Array of predictions.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError()


class DataSplits(TypedDict):
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
