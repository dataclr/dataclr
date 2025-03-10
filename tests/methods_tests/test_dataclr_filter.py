from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression

from dataclr.methods.dataclr_filter import DataclrFilter
from dataclr.metrics.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS
from dataclr.results import Result, ResultPerformance


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_regression, LinearRegression(), "rmse"),
        (make_classification, LogisticRegression(solver="liblinear"), "accuracy"),
    ],
)
def test_ranked_features(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    dataclr_filter = DataclrFilter(model=model, metric=metric, n_results=3, seed=42)
    dataclr_filter.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(dataclr_filter.ranked_features_, pd.Series)
    assert not dataclr_filter.ranked_features_.empty
    assert dataclr_filter.ranked_features_.dtype.kind in {"i", "f"}
    assert dataclr_filter.ranked_features_.is_monotonic_increasing


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_regression, LinearRegression(), "rmse"),
        (make_classification, LogisticRegression(solver="liblinear"), "accuracy"),
    ],
)
def test_results_list(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    dataclr_filter = DataclrFilter(model=model, metric=metric, n_results=3, seed=42)
    results = dataclr_filter.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, Result) for r in results)


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_regression, LinearRegression(), "rmse"),
        (make_classification, LogisticRegression(solver="liblinear"), "accuracy"),
    ],
)
def test_results_features_count(generate_dataset, dataset, model, metric):
    max_features = 3
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    dataclr_filter = DataclrFilter(model=model, metric=metric, n_results=3, seed=42)
    results = dataclr_filter.fit_transform(
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
        (make_regression, LinearRegression(), "rmse", REGRESSION_METRICS),
        (
            make_classification,
            LogisticRegression(solver="liblinear"),
            "accuracy",
            CLASSIFICATION_METRICS,
        ),
    ],
)
def test_result_performance(generate_dataset, dataset, model, metric, metrics):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    dataclr_filter = DataclrFilter(model=model, metric=metric, n_results=3, seed=42)
    results = dataclr_filter.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    dataclr_filter = DataclrFilter(
        model=LinearRegression(), metric="rmse", n_results=3, seed=42
    )
    results = dataclr_filter.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
