from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class Skewness(FilterMethod):
    """
    Skewness filter method for feature selection.

    This method evaluates the asymmetry of the distribution of each feature by
    calculating skewness. Features with higher skewness may indicate potential outliers
    or non-normal distributions.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()) -> Skewness:
        """
        Computes the skewness for each feature and ranks them in descending order.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            Skewness: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        skewness_values = X_train.skew()

        self.ranked_features_ = (
            pd.Series(skewness_values, index=X_train.columns)
            .sort_index()
            .sort_values(ascending=False, kind="stable")
        )

        return self
