from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dataclr.methods import RecursiveFeatureElimination
from dataclr.metrics.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from dataclr.results import Result, ResultPerformance


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
def test_results_list(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    rfe = RecursiveFeatureElimination(model=model, metric=metric, n_results=3, seed=42)
    results = rfe.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, Result) for r in results)


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
def test_results_features_count(generate_dataset, dataset, model, metric):
    max_features = 3
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    rfe = RecursiveFeatureElimination(model=model, metric=metric, n_results=3, seed=42)
    results = rfe.fit_transform(
        X_train, X_test, y_train, y_test, max_features=max_features
    )

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, Result) for r in results)
    for result in results:
        assert len(result.feature_list) <= max_features


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
def test_result_performance(generate_dataset, dataset, model, metric, metrics):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    rfe = RecursiveFeatureElimination(model=model, metric=metric, n_results=3, seed=42)
    results = rfe.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    rfe = RecursiveFeatureElimination(
        model=RandomForestRegressor(max_depth=10, n_estimators=10),
        metric="rmse",
        n_results=3,
        seed=42,
    )
    results = rfe.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
