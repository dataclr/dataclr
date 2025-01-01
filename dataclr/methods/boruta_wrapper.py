from __future__ import annotations

import pandas as pd
from boruta import BorutaPy
from dataclr.methods.filter_method import FilterMethod
from dataclr.methods.method import DataSplits
from dataclr.methods.wrapper_method import WrapperMethod
from dataclr.metrics import Metric
from dataclr.results import Result, ResultPerformance


class BorutaMethod(WrapperMethod):
    """
    Boruta-based wrapper method for feature selection.

    This method utilizes a model's ``feature_importances_`` attribute to
    iteratively identify important features. Boruta performs a rigorous test to
    distinguish real features from noise.

    Inherits from:
        :class:`WrapperMethod`: The base class that provides the structure for filter
                              methods.
    """

    def __init__(self, model, metric: Metric, n_results: int = 3, seed: int = 42):
        super().__init__(model, metric, n_results, seed)
        self.ranked_features_ = pd.Series(dtype=float)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> BorutaMethod:
        """
        Fits the Boruta feature selection process.

        Args:
            X_train (pd.DataFrame): Feature matrix of the training data.
            y_train (pd.Series): Target variable of the training data.

        Returns:
            BorutaMethod: The fitted instance with ranked features stored in
            ``self.ranked_features_``.

        Raises:
            ValueError: If the model does not have a ``feature_importances_`` attribute.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute!")

        boruta_series = self._get_boruta_series(X_train, y_train)

        self.ranked_features_ = boruta_series.sort_values(ascending=False)

        return self

    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> list[Result]:
        """
        Transforms the dataset by selecting the top-ranked features.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Testing feature matrix.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.

        Returns:
            list[Result]: List of results for the selected features.

        Raises:
            ValueError: If ``fit()`` has not been called prior to transform.
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

    def _get_results(
        self, data_splits: DataSplits, cached_performance: dict[int, ResultPerformance]
    ) -> list[Result]:
        try:
            self.fit(data_splits["X_train"], data_splits["y_train"])
        except ValueError:
            return []

        return FilterMethod(self.model, self.metric, self.n_results)._optimize(
            data_splits=data_splits,
            sorted_list=self.ranked_features_,
            cached_performance=cached_performance,
        )

    def _get_boruta_series(self, X_train: pd.DataFrame, y_train: pd.Series):
        boruta_selector = BorutaPy(
            self.model, n_estimators="auto", verbose=0, random_state=self.seed
        )

        boruta_selector.fit(X_train, y_train)

        return pd.Series(boruta_selector.ranking_, index=X_train.columns)
