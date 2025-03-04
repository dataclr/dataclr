from __future__ import annotations

import pandas as pd

from dataclr.methods.anova_filter import ANOVA
from dataclr.methods.filter_method import FilterMethod
from dataclr.methods.linear_correlation_filter import LinearCorrelation
from dataclr.methods.mutual_information_filter import MutualInformation
from dataclr.methods.spearman_correlation_filter import SpearmanCorrelation


class DataclrFilter(FilterMethod):
    """
    Combined filter method for feature selection using both Mutual Information
    and Linear Correlation techniques. This method ranks features based on
    their relationship with the target variable using both mutual information
    (for non-linear relationships) and Pearson's correlation (for linear relationships).
    The rankings from both methods are combined with equal weighting.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DataclrFilter:
        """
        Computes and ranks features using the selected methods (mutual information
        and linear correlation) with equal weight and combines their results.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            DataclrFilter: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """

        mi_method = MutualInformation(model=self.model, metric=self.metric)
        mi_method.fit(X_train, y_train)
        mi_series = mi_method.ranked_features_

        corr_method = LinearCorrelation(model=self.model, metric=self.metric)
        corr_method.fit(X_train, y_train)
        corr_series = corr_method.ranked_features_

        anova_method = ANOVA(model=self.model, metric=self.metric)
        anova_method.fit(X_train, y_train)
        anova_series = anova_method.ranked_features_

        spearman_method = SpearmanCorrelation(model=self.model, metric=self.metric)
        spearman_method.fit(X_train, y_train)
        spearman_series = spearman_method.ranked_features_

        # mi_series_normalized = (
        #     (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min())
        #     if mi_series.any()
        #     else mi_series
        # )
        # corr_series_normalized = (
        #     (corr_series - corr_series.min()) / (corr_series.max() - corr_series.min())
        #     if corr_series.any()
        #     else corr_series
        # )
        # anova_series_normalized = (
        #     (anova_series - anova_series.min())
        #     / (anova_series.max() - anova_series.min())
        #     if anova_series.any()
        #     else anova_series
        # )
        # spearman_series_normalized = (
        #     (spearman_series - spearman_series.min())
        #     / (spearman_series.max() - spearman_series.min())
        #     if spearman_series.any()
        #     else spearman_series
        # )

        # combined_scores = (
        #     0.5 * mi_series_normalized
        #     + 0.1 * corr_series_normalized
        #     + 0.3 * anova_series_normalized
        #     + 0.1 * spearman_series_normalized
        # )

        mi_ranks = mi_series.rank(method="min")
        corr_ranks = corr_series.rank(method="min")
        anova_ranks = anova_series.rank(method="min")
        spearman_ranks = spearman_series.rank(method="min")

        combined_ranks = (mi_ranks + corr_ranks + anova_ranks + spearman_ranks) / 4

        self.ranked_features_ = combined_ranks.sort_values()

        return self
