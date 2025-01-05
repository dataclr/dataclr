from __future__ import annotations

import pandas as pd
from sklearn.metrics import mutual_info_score

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import Metric, is_regression


class MaximalInformationCoefficient(FilterMethod):
    """
    Maximal Information Coefficient (MIC) filter method for feature selection.

    This method measures the strength of the relationship between each feature and
    the target variable. MIC is capable of detecting both linear and non-linear
    relationships, making it a versatile metric for feature selection.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.

    Extended Arguments:
        bins (int, optional): Number of bins used for discretizing the data during
                              the MIC calculation. Defaults to 4.
    """

    def __init__(
        self, model, metric: Metric, bins: int = 4, n_results: int = 3, seed: int = 42
    ) -> None:
        super().__init__(model, metric, n_results, seed)

        self.bins = bins

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> MaximalInformationCoefficient:
        """
        Computes the Maximal Information Coefficient (MIC) for each feature and
        ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            MaximalInformationCoefficient: The fitted instance with ranked features
            stored in ``self.ranked_features_``.
        """
        mic_values = self._calc_mic(X_train, y_train)

        self.ranked_features_ = (
            pd.Series(mic_values, index=X_train.columns)
            .sort_index()
            .sort_values(ascending=True, kind="stable")
        )

        return self

    def _calc_mutual_information(self, x_train: pd.Series, y_train: pd.Series) -> float:
        x_binned = pd.qcut(x_train, self.bins, labels=False, duplicates="drop")
        y_binned = y_train
        if is_regression(self.metric):
            y_binned = pd.qcut(y_train, self.bins, labels=False, duplicates="drop")

        return float(mutual_info_score(x_binned, y_binned))

    def _calc_mic(self, X_train: pd.DataFrame, y_train: pd.Series) -> list[float]:
        mic_values: list[float] = []
        for col in X_train.columns:
            mic_value = self._calc_mutual_information(X_train[col], y_train)
            mic_values.append(mic_value)
        return mic_values
