from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from dataclr.methods import ShapMethod
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
        (make_regression, LinearRegression(), "rmse"),
        (
            make_classification,
            LogisticRegression(solver="liblinear"),
            "accuracy",
        ),
    ],
)
def test_ranked_features(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    model.fit(X_train, y_train)
    shap = ShapMethod(model=model, metric=metric, n_results=3, seed=42)
    shap.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(shap.ranked_features_, pd.Series)
    assert not shap.ranked_features_.empty
    assert shap.ranked_features_.dtype.kind in {"i", "f"}
    assert shap.ranked_features_.is_monotonic_increasing


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
    shap = ShapMethod(model=model, metric=metric, n_results=3, seed=42)
    results = shap.fit_transform(X_train, X_test, y_train, y_test)

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
def test_result_performance(generate_dataset, dataset, model, metric, metrics):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    shap = ShapMethod(model=model, metric=metric, n_results=3, seed=42)
    results = shap.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    shap = ShapMethod(
        model=RandomForestRegressor(max_depth=10, n_estimators=10),
        metric="rmse",
        n_results=3,
        seed=42,
    )
    results = shap.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
