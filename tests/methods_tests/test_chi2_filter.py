from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from dataclr.methods import Chi2
from dataclr.metrics.metrics import CLASSIFICATION_METRICS
from dataclr.results import Result, ResultPerformance


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
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
    chi2 = Chi2(model=model, metric=metric, n_results=3, seed=42)
    chi2.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(chi2.ranked_features_, pd.Series)
    assert not chi2.ranked_features_.empty
    assert chi2.ranked_features_.dtype.kind in {"i", "f"}
    assert chi2.ranked_features_.is_monotonic_increasing


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (
            make_classification,
            LogisticRegression(solver="liblinear"),
            "accuracy",
        ),
    ],
)
def test_results_list(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    chi2 = Chi2(model=model, metric=metric, n_results=3, seed=42)
    results = chi2.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, Result) for r in results)


@pytest.mark.parametrize(
    "dataset, model, metric, metrics",
    [
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
    chi2 = Chi2(model=model, metric=metric, n_results=3, seed=42)
    results = chi2.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    chi2 = Chi2(
        model=LogisticRegression(solver="liblinear"),
        metric="rmse",
        n_results=3,
        seed=42,
    )
    results = chi2.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
