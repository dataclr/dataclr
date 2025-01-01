from __future__ import annotations

import numpy as np
import pandas as pd

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression


class CohensD(FilterMethod):
    """
    Cohen's D filter method for feature selection.

    This method calculates the effect size (Cohen's D) for each feature, comparing
    the mean differences between two target classes. It is applicable only for
    binary classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> CohensD:
        """
        Computes Cohen's D effect size for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            CohensD: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the target task is regression or if the target variable
                        has more than two unique classes, as Cohen's D is only
                        applicable to binary classification tasks.
        """
        if is_regression(self.metric) or y_train.nunique() != 2:
            raise ValueError(
                "Cohen's D cannot be used for regression task or with multiclass "
                "targets!"
            )

        feature_scores: dict[str, float] = {}
        for feature in X_train.columns:
            group1 = X_train[y_train == y_train.unique()[0]][feature]
            group2 = X_train[y_train == y_train.unique()[1]][feature]

            d = self._cohens_d(group1, group2)
            feature_scores[feature] = abs(d)

        self.ranked_features_ = (
            pd.Series(feature_scores).sort_index().sort_values(kind="stable")
        )

        return self

    def _cohens_d(self, group1: pd.Series[float], group2: pd.Series[float]) -> float:
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return np.nan

        return (mean1 - mean2) / pooled_std
