from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import f_classif, f_regression

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression


class ANOVA(FilterMethod):
    """
    ANOVA filter method for feature selection.

    This method ranks features based on the ANOVA F-statistic, which evaluates the
    variance between groups relative to the variance within groups. It supports both
    regression and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> ANOVA:
        """
        Computes the ANOVA F-statistics for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            ANOVA: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        if is_regression(self.metric):
            f_stats, _ = f_regression(X_train, y_train)
        else:
            f_stats, _ = f_classif(X_train, y_train)

        self.ranked_features_ = (
            pd.Series(f_stats, index=X_train.columns)
            .sort_index()
            .sort_values(kind="stable")
        )

        return self
