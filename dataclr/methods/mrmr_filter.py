from __future__ import annotations

import pandas as pd
from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import is_regression
from sklearn.feature_selection import f_classif, f_regression


class mRMR(FilterMethod):
    """
    Minimum Redundancy Maximum Relevance (mRMR) filter method for feature selection.

    This method selects features by maximizing relevance to the target variable while
    minimizing redundancy among selected features. Relevance is measured using either
    ANOVA F-statistic for classification or regression tasks, and redundancy is computed
    using Pearson correlation.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> mRMR:
        """
        Selects features by optimizing for both relevance and minimal redundancy.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            mRMR: The fitted instance with ranked features stored in
                  ``self.ranked_features_``.

        Raises:
            ValueError: If the input is incompatible with regression or classification
                        tasks.
        """
        if is_regression(self.metric):
            fvals, _ = f_regression(X_train, y_train)
        else:
            fvals, _ = f_classif(X_train, y_train)

        relevance = pd.Series(fvals, index=X_train.columns)

        corr_matrix = X_train.corr(method="pearson")

        selected_features: dict[str, float] = {}
        not_selected = list(X_train.columns)

        while not_selected:
            scores: dict[str, float] = {}
            for feature in not_selected:
                if len(selected_features) == 0:
                    redundancy = 0.0
                else:
                    feature_corrs = corr_matrix.loc[feature]
                    if isinstance(feature_corrs, pd.DataFrame):
                        feature_corrs = feature_corrs.squeeze()
                    redundancy = feature_corrs[list(selected_features.keys())].mean()

                scores[feature] = relevance[feature] - redundancy

            best_feature = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
            selected_features[best_feature] = scores[best_feature]
            not_selected.remove(best_feature)

        self.ranked_features_ = pd.Series(selected_features).sort_values()

        return self
