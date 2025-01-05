from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression


class MutualInformation(FilterMethod):
    """
    Mutual Information filter method for feature selection.

    This method evaluates the dependency between each feature and the target variable
    using mutual information, which measures both linear and non-linear relationships.
    It supports both regression and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> MutualInformation:
        """
        Computes mutual information between each feature and the target variable,
        and ranks the features.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            MutualInformation: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        if is_regression(self.metric):
            mi_list = mutual_info_regression(X_train, y_train, random_state=self.seed)
        else:
            mi_list = mutual_info_classif(X_train, y_train, random_state=self.seed)

        self.ranked_features_ = pd.Series(mi_list, index=X_train.columns).sort_values()

        return self
