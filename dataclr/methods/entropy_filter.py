from __future__ import annotations

import numpy as np
import pandas as pd

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import Metric, is_regression


class Entropy(FilterMethod):
    """
    Entropy-based filter method for feature selection.

    This method evaluates the importance of features by calculating the information gain
    with respect to the target variable. For regression tasks, the target is discretized
    into bins before calculating entropy.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.

    Extended Arguments:
        bins (int, optional): The number of bins to discretize the target variable into
                              for regression tasks. Defaults to 4.
    """

    def __init__(
        self, model, metric: Metric, bins: int = 4, n_results: int = 3, seed: int = 42
    ) -> None:
        super().__init__(model, metric, n_results, seed)

        self.bins = bins

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Entropy:
        """
        Computes the information gain for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            Entropy: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the target variable cannot be discretized for regression
                        tasks.
        """
        if is_regression(self.metric):
            y_train = pd.qcut(y_train, self.bins, labels=False, duplicates="drop")

        total_entropy = self._calculate_entropy(y_train)

        feature_importances: dict[str, float] = {}
        for column in X_train.columns:
            feature_importances[column] = self._information_gain(
                X_train[column], y_train, total_entropy
            )

        self.ranked_features_ = (
            pd.Series(feature_importances).sort_index().sort_values(kind="stable")
        )

        return self

    def _information_gain(
        self, feature: pd.Series, y_train: pd.Series, total_entropy: float
    ) -> float:
        weighted_entropy = 0.0

        for value in feature.unique():
            subset_target = y_train[feature == value]
            subset_probability = len(subset_target) / len(y_train)
            weighted_entropy += subset_probability * self._calculate_entropy(
                subset_target
            )

        info_gain = total_entropy - weighted_entropy

        return info_gain

    def _calculate_entropy(self, data: pd.Series) -> float:
        probabilities = data.value_counts(normalize=True)

        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy
