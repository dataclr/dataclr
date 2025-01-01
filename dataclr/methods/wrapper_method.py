from __future__ import annotations

from dataclr.methods.method import Method
from dataclr.metrics import Metric


class WrapperMethod(Method):
    """
    A base class for wrapper feature selection methods.

    Wrapper methods use a machine learning model to evaluate feature subsets by training
    and validating the model on different combinations of features. This class serves as
    the foundation for implementing specific wrapper-based feature selection algorithms.

    Inherits from:
        :class:`Method`: The base class that provides the structure for feature
        selection methods.
    """

    def __init__(
        self, model, metric: Metric, n_results: int = 3, seed: int = 42
    ) -> None:
        super().__init__(model, metric, n_results, seed)
