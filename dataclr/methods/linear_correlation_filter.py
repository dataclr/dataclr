from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class LinearCorrelation(FilterMethod):
    """
    Linear Correlation filter method for feature selection.

    This method evaluates the linear relationship between each feature and the
    target variable using Pearson's correlation coefficient. It is suitable for both
    regression and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearCorrelation:
        """
        Computes Pearson's correlation coefficient for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            LinearCorrelation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        correlation = X_train.corrwith(y_train)

        self.ranked_features_ = (
            correlation.abs().sort_index().sort_values(kind="stable")
        )

        return self
