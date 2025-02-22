from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from dataclr.methods import HyperoptMethod
from dataclr.metrics.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from dataclr.results import Result, ResultPerformance


def generate_dataset(
    dataset_generator: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = dataset_generator(
        n_samples=10000, n_features=10, n_informative=3, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (
        pd.DataFrame(X_train),
        pd.DataFrame(X_test),
        pd.Series(y_train),
        pd.Series(y_test),
    )


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_regression, RandomForestRegressor(max_depth=10, n_estimators=10), "rmse"),
        (
            make_classification,
            RandomForestClassifier(max_depth=10, n_estimators=10),
            "accuracy",
        ),
    ],
)
def test_results_list(dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    hyperopt = HyperoptMethod(model=model, metric=metric, n_results=3, seed=42)
    results = hyperopt.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, Result) for r in results)


@pytest.mark.parametrize(
    "dataset, model, metric, metrics",
    [
        (
            make_regression,
            RandomForestRegressor(max_depth=10, n_estimators=10),
            "rmse",
            REGRESSION_METRICS,
        ),
        (
            make_classification,
            RandomForestClassifier(max_depth=10, n_estimators=10),
            "accuracy",
            CLASSIFICATION_METRICS,
        ),
    ],
)
def test_result_performance(dataset, model, metric, metrics):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    hyperopt = HyperoptMethod(model=model, metric=metric, n_results=3, seed=42)
    results = hyperopt.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    hyperopt = HyperoptMethod(
        model=RandomForestRegressor(max_depth=10, n_estimators=10),
        metric="rmse",
        n_results=3,
        seed=42,
    )
    results = hyperopt.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
