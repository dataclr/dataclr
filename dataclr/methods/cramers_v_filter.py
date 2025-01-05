from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression


class CramersV(FilterMethod):
    """
    Cramér's V filter method for feature selection.

    This method measures the association between categorical features and the
    target variable using Cramér's V statistic. It is applicable only for
    classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "CramersV":
        """
        Computes Cramér's V statistic for each categorical feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            CramersV: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the target task is regression, as Cramér's V is only
                        applicable to classification tasks.
        """
        if is_regression(self.metric):
            raise ValueError("Cramer's V cannot be used for regression task!")

        feature_scores: dict[str, float] = {}
        for feature in X_train.columns:
            contingency_table = pd.crosstab(X_train[feature], y_train)

            v = self._cramers_v(contingency_table)
            feature_scores[feature] = v

        self.ranked_features_ = (
            pd.Series(feature_scores).sort_index().sort_values(kind="stable")
        )

        return self

    def _cramers_v(self, contingency_table: pd.DataFrame) -> float:
        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        rows, cols = contingency_table.shape
        k = min(rows, cols)

        return np.sqrt(chi2 / (n * (k - 1)))
