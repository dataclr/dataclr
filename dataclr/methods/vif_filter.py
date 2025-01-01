from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLS
from statsmodels.tools.tools import add_constant

from dataclr.methods.filter_method import FilterMethod
from dataclr.metrics import Metric


class VarianceInflationFactor(FilterMethod):
    """
    Variance Inflation Factor (VIF) filter method for feature selection.

    This method identifies features with multicollinearity by calculating the Variance
    Inflation Factor (VIF) for each feature. Features with a VIF above the specified
    threshold are considered collinear and ranked based on their VIF values.

    Inherits from:
        :class:`FilterMethod`: The base class that provides the structure for filter
                              methods.

    Extended Arguments:
        threshold (int, optional): The VIF threshold above which features are considered
                                   multicollinear. Defaults to 5.
    """

    def __init__(
        self,
        model,
        metric: Metric,
        threshold: int = 5,
        n_results: int = 3,
        seed: int = 42,
    ) -> None:
        super().__init__(model, metric, n_results, seed)
        self.threshold = threshold

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()
    ) -> VarianceInflationFactor:
        """
        Computes the Variance Inflation Factor (VIF) for each feature and filters those
        exceeding the specified threshold.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series, optional): Target variable of the training data. Not
                                           used in this method but included for
                                           compatibility. Defaults to an empty Series.

        Returns:
            VarianceInflationFactor: The fitted instance with ranked features stored in
            ``self.ranked_features_``.
        """
        X_train_const = add_constant(X_train)

        vif_data = pd.DataFrame(
            {
                "feature": X_train.columns,
                "VIF": [
                    self._variance_inflation_factor(X_train_const, i + 1)
                    for i in range(X_train.shape[1])
                ],
            }
        )

        filtered_features = vif_data[vif_data["VIF"] >= self.threshold]["feature"]

        self.ranked_features_ = (
            vif_data[vif_data["feature"].isin(filtered_features)]
            .set_index("feature")["VIF"]
            .sort_values(ascending=False)
        )

        return self

    def _variance_inflation_factor(
        self, exog: pd.DataFrame | np.ndarray, exog_idx: int
    ) -> float:
        k_vars = exog.shape[1]
        exog = np.asarray(exog)
        x_i = exog[:, exog_idx]
        mask = np.arange(k_vars) != exog_idx
        x_noti = exog[:, mask]
        r_squared_i = OLS(x_i, x_noti).fit().rsquared

        if not math.isclose(r_squared_i, 1.0):
            return 1.0 / (1.0 - r_squared_i)

        return float("inf")
