from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class Kurtosis(FilterMethod):
    """
    Kurtosis filter method for feature selection.

    This method evaluates the shape of the distribution of each feature by calculating
    kurtosis, which measures the "tailedness" of the distribution. Features with
    higher kurtosis may capture more extreme values or outliers.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()) -> Kurtosis:
        """
        Computes the kurtosis for each feature and ranks them in descending order.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            Kurtosis: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        kurtosis_values = X_train.kurtosis()

        self.ranked_features_ = (
            pd.Series(kurtosis_values, index=X_train.columns)
            .sort_index()
            .sort_values(ascending=False, kind="stable")
        )

        return self
