from __future__ import annotations

import numpy as np
import pandas as pd
from shap import Explainer, Explanation, LinearExplainer, TreeExplainer

from dataclr._typing import DataSplits
from dataclr.methods.filter_method import FilterMethod
from dataclr.methods.wrapper_method import WrapperMethod
from dataclr.metrics import Metric
from dataclr.results import Result, ResultPerformance


class ShapMethod(WrapperMethod):
    """
    SHAP-based wrapper method for feature selection.

    This method utilizes SHAP (SHapley Additive exPlanations) values to evaluate
    the importance of features based on the model's predictions. It supports models
    with ``feature_importances_`` (e.g., tree-based models) or ``coef_``
    (e.g., linear models).

    Inherits from:
        :class:`WrapperMethod`: The base class that provides the structure for wrapper
                                methods.
    """

    def __init__(self, model, metric: Metric, n_results: int = 3, seed: int = 42):
        super().__init__(model, metric, n_results, seed)
        self.ranked_features_ = pd.Series(dtype=float)

    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> ShapMethod:
        """
        Computes SHAP values for each feature and ranks them.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Testing feature matrix.

        Returns:
            ShapMethod: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the model lacks both ``feature_importances_`` and ``coef_``
                        attributes, which are required for SHAP computation.
        """
        if hasattr(self.model, "feature_importances_"):
            self.ranked_features_ = self._get_shap_series(
                X_train, X_test, TreeExplainer
            )
        elif hasattr(self.model, "coef_"):
            self.ranked_features_ = self._get_shap_series(
                X_train, X_test, LinearExplainer
            )
        else:
            raise ValueError(
                "Model does not have neither feature_importances_ nor coef_ attributes!"
            )

        return self

    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> list[Result]:
        """
        Applies the SHAP-based feature selection process to evaluate and optimize
        subsets.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Testing feature matrix.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.

        Returns:
            list[Result]: A list of results containing feature subsets and their
                          corresponding performance metrics.

        Raises:
            ValueError: If ``fit()`` has not been called prior to ``transform``.
        """
        if self.ranked_features_.empty:
            raise ValueError("You need to call fit() before transform()!")

        data_splits: DataSplits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        return FilterMethod(self.model, self.metric, self.n_results)._optimize(
            data_splits=data_splits,
            sorted_list=self.ranked_features_,
            cached_performance={},
        )

    def _get_shap_series(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, explainer: type[Explainer]
    ) -> pd.Series:
        shap_explainer = explainer(self.model, X_train)
        shap_values: Explanation
        if hasattr(self.model, "feature_importances_"):
            shap_values = shap_explainer(X_test, check_additivity=False)
        else:
            shap_values = shap_explainer(X_test)

        mean_shap_values = np.abs(shap_values.values).mean(axis=0)

        return pd.Series(mean_shap_values, index=X_train.columns).sort_values()

    def _get_results(
        self, data_splits: DataSplits, cached_performance: dict[int, ResultPerformance]
    ) -> list[Result]:
        try:
            self.model.fit(data_splits["X_train"], data_splits["y_train"])
            self.fit(data_splits["X_train"], data_splits["X_test"])
        except ValueError:
            return []

        return FilterMethod(self.model, self.metric, self.n_results)._optimize(
            data_splits=data_splits,
            sorted_list=self.ranked_features_,
            cached_performance=cached_performance,
        )
