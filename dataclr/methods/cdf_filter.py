from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import Metric, is_regression


class CumulativeDistributionFunction(FilterMethod):
    """
    Cumulative Distribution Function (CDF) filter method for feature selection.

    This method evaluates the separability of feature distributions across
    target bins or classes using the Kolmogorov-Smirnov test, applicable for
    both regression and classification tasks.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    Extended Arguments:
        bins (int, optional): The number of bins to divide the target variable
                              into for distribution comparison. Defaults to 4.
    """

    def __init__(
        self, model, metric: Metric, bins: int = 4, n_results: int = 3, seed: int = 42
    ) -> None:
        super().__init__(model, metric, n_results, seed)

        self.bins = bins

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> CumulativeDistributionFunction:
        """
        Fits the CDF feature selection process by computing feature scores
        based on distribution separability.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            CumulativeDistributionFunction: The fitted instance with ranked
            features stored in ``self.ranked_features_``.
        """
        feature_bins = pd.qcut(y_train, self.bins, labels=False, duplicates="drop")

        feature_scores: dict[str, float] = {}
        for feature in X_train.columns:
            ks_stats = []

            if is_regression(self.metric):
                for i in range(self.bins - 1):
                    group1 = X_train[feature_bins == i][feature]
                    group2 = X_train[feature_bins == i + 1][feature]

                    ks_stat, _ = ks_2samp(group1, group2)
                    ks_stats.append(ks_stat)
            else:  # classification
                classes = np.unique(y_train)
                for i, class_from in enumerate(classes):
                    for class_to in classes[i + 1 :]:
                        class1 = X_train[y_train == class_from][feature]
                        class2 = X_train[y_train == class_to][feature]

                        ks_stat, _ = ks_2samp(class1, class2)
                        ks_stats.append(ks_stat)

            feature_scores[feature] = np.mean(ks_stats).astype(float)

        self.ranked_features_ = (
            pd.Series(feature_scores).sort_index().sort_values(kind="stable")
        )

        return self
