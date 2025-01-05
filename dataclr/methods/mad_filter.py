from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class MeanAbsoluteDeviation(FilterMethod):
    """
    Mean Absolute Deviation (MAD) filter method for feature selection.

    This method calculates the average absolute deviation of each feature from its mean.
    Features with lower deviation are considered less informative for distinguishing
    patterns.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()
    ) -> MeanAbsoluteDeviation:
        """
        Computes the Mean Absolute Deviation (MAD) for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            MeanAbsoluteDeviation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        mean = X_train.mean()
        mad_values = (X_train - mean).abs().mean()

        self.ranked_features_ = pd.Series(
            mad_values, index=X_train.columns
        ).sort_values()

        return self
