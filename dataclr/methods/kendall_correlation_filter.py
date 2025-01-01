from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class KendallCorrelation(FilterMethod):
    """
    Kendall's Tau Correlation filter method for feature selection.

    This method evaluates the monotonic relationship between each feature and the
    target variable using Kendall's Tau correlation. It is suitable for both regression
    and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> KendallCorrelation:
        """
        Computes Kendall's Tau correlation for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            KendallCorrelation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        correlation = X_train.corrwith(y_train, method="kendall")

        self.ranked_features_ = (
            correlation.abs().sort_index().sort_values(kind="stable")
        )

        return self
