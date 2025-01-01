from __future__ import annotations

import copy
from functools import partial

import optuna
import pandas as pd
from dataclr._evaluate import train_eval
from dataclr.methods.method import DataSplits
from dataclr.methods.wrapper_method import WrapperMethod
from dataclr.metrics import Metric, is_maximizing_metric, is_regression
from dataclr.metrics.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from dataclr.results import Result, ResultPerformance
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaMethod(WrapperMethod):
    """
    Optuna-based wrapper method for feature selection.

    This method utilizes Optuna for hyperparameter optimization to evaluate and select
    feature subsets. It performs a specified number of trials to explore feature
    combinations and identify the best subsets based on the provided metric.

    Inherits from:
        :class:`WrapperMethod`: The base class that provides the structure for wrapper
                                methods.

    Extended Arguments:
        n_trials (int, optional): The number of trials for Optuna's optimization
                                  process. Defaults to
                                  ``config.OPTUNA_METHOD_N_TRIALS``.
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
        self.n_trials = None

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series = pd.Series()
    ) -> OptunaMethod:
        """
        Placeholder method for fitting the feature selection process.

        Args:
            X_train (pd.DataFrame, optional): Feature matrix of the training data.
                                              Defaults to an empty DataFrame.
            y_train (pd.Series, optional): Target variable of the training data.
                                           Defaults to an empty Series.

        Returns:
            OptunaMethod: The instance itself.
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
        Applies the feature selection process by evaluating subsets using Optuna
        optimization.

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
        self, data_splits: DataSplits, cached_performance: dict[int, ResultPerformance]
    ) -> list[Result]:
        return self._optimize(
            data_splits=data_splits,
            cached_performance=cached_performance,
        )

    def _optimize(
        self,
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
    ) -> list[Result]:
        if self.n_trials is None:
            self.n_trials = len(data_splits["X_train"].columns) * 10
        study = optuna.create_study(
            directions=[
                "maximize" if is_maximizing_metric(self.metric) else "minimize"
            ],
            sampler=TPESampler(
                seed=self.seed,
                n_startup_trials=max(5, len(data_splits["X_test"].columns) // 4),
            ),
        )

        self.total_combinations = 0
        objective_with_params = partial(
            self._objective,
            data_splits=data_splits,
            cached_performance=cached_performance,
        )
        study.optimize(objective_with_params, n_trials=self.n_trials)

        base_result = train_eval(self.model, self.metric, data_splits)

        return self._get_n_best_params(
            study=study,
            base_result=base_result,
            base_feature_list_length=len(data_splits["X_train"].columns),
        )

    def _objective(
        self,
        trial: optuna.Trial,
        data_splits: DataSplits,
        cached_performance: dict[int, ResultPerformance],
    ) -> float:
        feature_mask = [
            trial.suggest_categorical(f"feature_{i}", [0, 1])
            for i in range(data_splits["X_train"].shape[1])
        ]
        selected_features_indexes = [
            i for i, use in enumerate(feature_mask) if use == 1
        ]

        selected_features = data_splits["X_train"].columns[selected_features_indexes]

        trial.set_user_attr("selected_features", selected_features)

        if len(selected_features) == 0:
            return float("inf")

        new_data_splits = copy.deepcopy(data_splits)

        new_data_splits["X_train"] = data_splits["X_train"][selected_features]
        new_data_splits["X_test"] = data_splits["X_test"][selected_features]

        features_key = hash(tuple(new_data_splits["X_train"].columns))
        if features_key in cached_performance:
            self._set_trial_attributes(trial, cached_performance[features_key])

            value = cached_performance[features_key][self.metric]
            if value is None:
                raise ValueError("Could not retrieve metric")

            return value

        self.total_combinations += 1

        performance_result = train_eval(self.model, self.metric, new_data_splits)

        cached_performance[features_key] = performance_result
        self._set_trial_attributes(trial, performance_result)

        value = performance_result[self.metric]
        if value is None:
            raise ValueError("Could not retrieve metric")

        return value

    def _set_trial_attributes(
        self, trial: optuna.Trial, performance_result: ResultPerformance
    ):
        for attr in (
            REGRESSION_METRICS if is_regression(self.metric) else CLASSIFICATION_METRICS
        ):
            trial.set_user_attr(attr, getattr(performance_result, attr))

    def _get_n_best_params(
        self,
        study: optuna.Study,
        base_result: ResultPerformance,
        base_feature_list_length: int,
    ) -> list[Result]:
        df: pd.DataFrame = study.trials_dataframe()
        df = df.dropna()

        tolerance = 1e-9
        result = base_result[self.metric]
        if result is None:
            raise ValueError("Base result is None!")

        if is_maximizing_metric(self.metric):
            better_values = df["value"] > result - tolerance
        else:
            better_values = df["value"] < result + tolerance

        selected_df = df[better_values].drop_duplicates(subset=["value"])
        sorted_df = selected_df.sort_values(
            "value", ascending=(not is_maximizing_metric(self.metric))
        )

        best_params = []
        for i in range(min(self.n_results, len(sorted_df))):
            best_trial = study.trials[sorted_df.index[i]]

            performance_metrics = {
                key: best_trial.user_attrs.get(key)
                for key in ["r2", "rmse", "accuracy", "precision", "recall", "f1"]
            }

            selected_features = best_trial.user_attrs.get("selected_features")

            best_params.append(
                Result(
                    params={
                        "k": base_feature_list_length - len(list(selected_features))
                    },
                    performance=ResultPerformance(**performance_metrics),
                    feature_list=list(selected_features),
                )
            )

        return best_params
