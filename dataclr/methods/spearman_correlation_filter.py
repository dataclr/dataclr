from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class SpearmanCorrelation(FilterMethod):
    """
    Spearman Correlation filter method for feature selection.

    This method evaluates the monotonic relationship between each feature and the
    target variable using Spearman's rank correlation coefficient. It is suitable for
    both regression and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> SpearmanCorrelation:
        """
        Computes Spearman's rank correlation coefficient for each feature and ranks
        them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            SpearmanCorrelation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        correlation = X_train.corrwith(y_train, method="spearman")

        self.ranked_features_ = (
            correlation.abs().sort_index().sort_values(kind="stable")
        )

        return self
