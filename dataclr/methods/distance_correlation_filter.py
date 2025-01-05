from __future__ import annotations

import dcor
import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class DistanceCorrelation(FilterMethod):
    """
    Distance Correlation filter method for feature selection.

    This method evaluates the dependency between each feature and the target variable
    using distance correlation, a measure of both linear and non-linear relationships.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DistanceCorrelation:
        """
        Computes distance correlation for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            DistanceCorrelation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        correlation = {
            col: dcor.distance_correlation(X_train[col], y_train)
            for col in X_train.columns
        }

        self.ranked_features_ = (
            pd.Series(correlation).sort_index().sort_values(kind="stable")
        )

        return self
