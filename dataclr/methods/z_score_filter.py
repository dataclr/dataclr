from __future__ import annotations

import pandas as pd

from dataclr.methods.filter_method import FilterMethod


class ZScore(FilterMethod):
    """
    Z-Score filter method for feature selection.

    This method evaluates the importance of features by calculating the mean of the
    absolute Z-scores for each feature. Features with higher mean absolute Z-scores
    are considered more informative.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()) -> ZScore:
        """
        Computes the mean absolute Z-score for each feature and ranks them in
        descending order.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            ZScore: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        z_scores = X_train.apply(lambda x: (x - x.mean()) / x.std(ddof=0))

        mean_abs_z_scores = z_scores.abs().mean()

        self.ranked_features_ = pd.Series(
            mean_abs_z_scores, index=X_train.columns
        ).sort_values(ascending=False)

        return self
