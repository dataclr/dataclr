from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture
def generate_dataset():
    """Fixture that returns a function to generate train-test split datasets."""

    def _generate(
        dataset_generator: Callable,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = dataset_generator(
            n_samples=100, n_features=10, n_informative=3, random_state=42
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

    return _generate
