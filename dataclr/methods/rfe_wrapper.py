from __future__ import annotations

import copy

import pandas as pd

from dataclr._evaluate import train_eval
from dataclr._typing import DataSplits
from dataclr.methods.wrapper_method import WrapperMethod
from dataclr.metrics import Metric, is_maximizing_metric
from dataclr.results import Result, ResultPerformance


class RecursiveFeatureElimination(WrapperMethod):
    """
    Recursive Feature Elimination (RFE) feature selection method.

    This method iteratively removes the least important feature and evaluates the model's
    performance to determine the optimal subset of features.

    Inherits from:
        :class:`WrapperMethod`: The base class for wrapper-based feature selection methods.

    Attributes:
        result_list (list[Result]): Stores feature selection results during the process.
    """

    def __init__(self, model, metric: Metric, n_results: int = 3, seed: int = 42):
        super().__init__(model, metric, n_results, seed)
        self.result_list: list[Result] = []

    def fit(
        self, X_train: pd.DataFrame = pd.DataFrame(), y_train: pd.Series = pd.Series()
    ) -> RecursiveFeatureElimination:
        """
        Fits the model.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target variable.

        Returns:
            RecursiveFeatureElimination: Returns self.
        """
        return self

    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        max_features: int = -1,
    ) -> list[Result]:
        """
        Performs Recursive Feature Elimination and selects the optimal subset of features.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Testing feature matrix.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.
            max_features (int): Number of max features count in results.
                Defaults to -1 (all features number).

        Returns:
            list[Result]: A list of feature subsets and their corresponding performance metrics.
        """
        data_splits: DataSplits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        return self._get_results(data_splits, {}, max_features=max_features)

    def _get_results(
        self,
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
        keep_features: list[str] = [],
        max_features: int = -1,
    ) -> list[Result]:
        """
        Performs Recursive Feature Elimination and selects the optimal subset of features.
        """
        selected_features = list(data_splits["X_train"].columns)
        remaining_features = list(data_splits["X_train"].columns)

        if max_features == -1:
            max_features = len(remaining_features)

        for feature in keep_features:
            remaining_features.remove(feature)

        while remaining_features:
            worst_feature = None
            best_performance = None

            print(len(remaining_features))

            for feature in remaining_features:
                current_features = copy.deepcopy(selected_features)
                current_features.remove(feature)
                performance = self._evaluate_features(
                    current_features, data_splits, cached_performance
                )

                if best_performance is None or self._compare(
                    performance, best_performance
                ):
                    best_performance = performance
                    worst_feature = feature

            if worst_feature is not None:
                selected_features.remove(worst_feature)
                remaining_features.remove(worst_feature)
                if len(selected_features) <= max_features:
                    self.result_list.append(
                        Result(
                            params={"k": len(selected_features)},
                            performance=best_performance,
                            feature_list=selected_features[:],
                        )
                    )
            else:
                break

        base_result = train_eval(self.model, self.metric, data_splits)
        return self._get_n_best_params(self.result_list, base_result)

    def _evaluate_features(
        self,
        features: list[str],
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
    ) -> ResultPerformance:
        """
        Evaluates a given set of features using the model.
        """
        features_key = hash(tuple(features))
        if features_key in cached_performance:
            return cached_performance[features_key]

        self.total_combinations += 1

        new_data_splits = copy.deepcopy(data_splits)
        new_data_splits["X_train"] = data_splits["X_train"][features]
        new_data_splits["X_test"] = data_splits["X_test"][features]

        performance_result = train_eval(self.model, self.metric, new_data_splits)
        cached_performance[features_key] = performance_result

        return performance_result

    def _compare(
        self, performance1: ResultPerformance, performance2: ResultPerformance
    ) -> bool:
        """
        Compares two performance results based on the evaluation metric.
        """
        reverse = is_maximizing_metric(self.metric)
        tolerance = 1e-9

        return (
            performance1[self.metric] > performance2[self.metric] - tolerance
            if reverse
            else performance1[self.metric] < performance2[self.metric] + tolerance
        )

    def _get_n_best_params(
        self, results: list[Result], base_result: ResultPerformance
    ) -> list[Result]:
        """
        Selects the top N best-performing feature subsets.
        """
        if not results:
            return []

        reverse = is_maximizing_metric(self.metric)
        filtered_results: list[Result] = []
        seen_performance: set[float] = set()

        for result in results:
            performance_value = result.performance[self.metric]
            if performance_value is None or performance_value in seen_performance:
                continue

            res_value = base_result[self.metric]
            if res_value is None:
                raise ValueError("Base result is None!")

            tolerance = 1e-9
            if (reverse and performance_value > res_value - tolerance) or (
                not reverse and performance_value < res_value + tolerance
            ):
                filtered_results.append(result)
                seen_performance.add(performance_value)

        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.performance[self.metric] or float("inf"),
            reverse=reverse,
        )

        return sorted_results[: self.n_results]
