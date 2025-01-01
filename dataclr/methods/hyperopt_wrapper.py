from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from dataclr._evaluate import train_eval
from dataclr.methods.method import DataSplits
from dataclr.methods.wrapper_method import WrapperMethod
from dataclr.metrics import Metric, is_maximizing_metric
from dataclr.results import Result, ResultPerformance
from hyperopt import Trials, fmin, hp, tpe


class HyperoptMethod(WrapperMethod):
    """
    Hyperparameter optimization (Hyperopt) wrapper method for feature selection.

    This method evaluates feature subsets by optimizing a given metric through a
    hyperparameter optimization process. It leverages a specified number of trials
    to explore feature combinations and returns the best subsets.

    Inherits from:
        :class:`WrapperMethod`: The base class that provides the structure for
                                wrapper methods.

    Extended Arguments:
        n_trials (int, optional): The number of trials for hyperparameter optimization.
                                  Defaults to ``config.HYPEROPT_METHOD_N_TRIALS``.
    """

    def __init__(
        self,
        model,
        metric: Metric,
        n_trials: int = None,
        n_results: int = 3,
        seed: int = 42,
    ):
        super().__init__(model, metric, n_results, seed)
        self.n_trials = n_trials
        self.result_list: list[Result] = []

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()
    ) -> HyperoptMethod:
        """
        Placeholder method for fitting the feature selection process.

        Args:
            X_train (pd.DataFrame, optional): Feature matrix of the training data.
                                              Defaults to an empty DataFrame.
            y_train (pd.Series, optional): Target variable of the training data.
                                           Defaults to an empty Series.

        Returns:
            HyperoptMethod: The instance itself.
        """
        return self

    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> list[Result]:
        """
        Applies the feature selection process by evaluating subsets using
        hyperparameter optimization.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            X_test (pd.DataFrame): Testing feature matrix.
            y_train (pd.Series): Training target variable.
            y_test (pd.Series): Testing target variable.

        Returns:
            list[Result]: A list of results containing feature subsets and their
                          corresponding performance metrics.
        """
        data_splits: DataSplits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        return self._get_results(data_splits, {})

    def _get_results(
        self,
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
    ) -> list[Result]:
        if self.n_trials is None:
            self.n_trials = len(data_splits["X_train"].columns) * 10
        trials = Trials()

        space = {
            f"feature_{i}": hp.choice(f"feature_{i}", [0, 1])
            for i in range(data_splits["X_train"].shape[1])
        }

        fmin(
            fn=lambda params: self._objective(params, data_splits, cached_performance),
            space=space,
            algo=tpe.suggest,
            max_evals=self.n_trials,
            trials=trials,
            return_argmin=False,
            rstate=np.random.default_rng(self.seed),
            show_progressbar=False,
        )

        base_result = train_eval(self.model, self.metric, data_splits)

        return self._get_n_best_params(self.result_list, base_result)

    def _objective(
        self,
        params,
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
    ) -> float:
        cols = [i for i, j in params.items() if j == 1]
        feature_indices = [int(name.split("_")[1]) for name in cols]
        selected_features = data_splits["X_train"].columns[feature_indices]

        if len(selected_features) == 0:
            return float("inf")

        new_data_splits = copy.deepcopy(data_splits)
        new_data_splits["X_train"] = data_splits["X_train"][selected_features]
        new_data_splits["X_test"] = data_splits["X_test"][selected_features]

        features_key = hash(tuple(new_data_splits["X_train"].columns))
        if features_key in cached_performance:
            self.result_list.append(
                Result(
                    params={
                        "k": len(data_splits["X_train"].columns)
                        - len(list(selected_features))
                    },
                    performance=cached_performance[features_key],
                    feature_list=list(selected_features),
                )
            )

            value = cached_performance[features_key][self.metric]
            if value is None:
                raise ValueError("Could not retrieve metric")

            return -value if is_maximizing_metric(self.metric) else value

        self.total_combinations += 1

        performance_result = train_eval(self.model, self.metric, new_data_splits)

        cached_performance[features_key] = performance_result

        self.result_list.append(
            Result(
                params={
                    "k": len(data_splits["X_train"].columns)
                    - len(list(selected_features))
                },
                performance=performance_result,
                feature_list=list(selected_features),
            )
        )

        value = cached_performance[features_key][self.metric]
        if value is None:
            raise ValueError("Could not retrieve metric")

        return -value if is_maximizing_metric(self.metric) else value

    def _get_n_best_params(
        self, results: list[Result], base_result: ResultPerformance
    ) -> list[Result]:
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
