from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import chi2

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression


class Chi2(FilterMethod):
    """
    Chi-squared (Chi2) filter method for feature selection.

    This method evaluates the dependency between each feature and the target variable
    using the chi-squared statistic. It is applicable only for classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Chi2:
        """
        Computes the Chi-squared statistic for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            Chi2: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the target task is regression, as Chi2 is only
                        applicable to classification tasks.
        """
        if is_regression(self.metric):
            raise ValueError("Chi2 cannot be used for regression task!")

        chi2_scores, _ = chi2(X_train, y_train)

        self.ranked_features_ = (
            pd.Series(chi2_scores, index=X_train.columns)
            .sort_index()
            .sort_values(kind="stable")
        )

        return self
