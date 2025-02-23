from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from dataclr.methods import CohensD
from dataclr.metrics.metrics import CLASSIFICATION_METRICS
from dataclr.results import Result, ResultPerformance


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_classification, LogisticRegression(solver="liblinear"), "accuracy"),
    ],
)
def test_ranked_features(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    cohens_d = CohensD(model=model, metric=metric, n_results=3, seed=42)
    cohens_d.fit_transform(X_train, X_test, y_train, y_test)

    assert isinstance(cohens_d.ranked_features_, pd.Series)
    assert not cohens_d.ranked_features_.empty
    assert cohens_d.ranked_features_.dtype.kind in {"i", "f"}
    assert cohens_d.ranked_features_.is_monotonic_increasing


@pytest.mark.parametrize(
    "dataset, model, metric",
    [
        (make_classification, LogisticRegression(solver="liblinear"), "accuracy"),
    ],
)
def test_results_list(generate_dataset, dataset, model, metric):
    X_train, X_test, y_train, y_test = generate_dataset(dataset)
    cohens_d = CohensD(model=model, metric=metric, n_results=3, seed=42)
    results = cohens_d.fit_transform(X_train, X_test, y_train, y_test)

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
    cohens_d = CohensD(model=model, metric=metric, n_results=3, seed=42)
    results = cohens_d.fit_transform(X_train, X_test, y_train, y_test)

    for r in results:
        assert isinstance(r.performance, ResultPerformance)
        for metric in metrics:
            assert r.performance[metric] is not None


def test_empty_dataset():
    cohens_d = CohensD(
        model=LogisticRegression(solver="liblinear"),
        metric="accuracy",
        n_results=3,
        seed=42,
    )
    results = cohens_d.fit_transform(
        pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    )

    assert len(results) == 0
