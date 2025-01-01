from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class VarianceThreshold(FilterMethod):
    """
    Variance Threshold filter method for feature selection.

    This method ranks features based on their variance. Features with higher variance
    are considered more informative, while features with low variance may contribute
    less to the model's performance.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()
    ) -> VarianceThreshold:
        """
        Computes the variance for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            VarianceThreshold: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        self.ranked_features_ = X_train.var().sort_values()

        return self
