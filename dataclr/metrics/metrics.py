from typing import Literal

Metric = Literal["rmse", "r2", "accuracy", "precision", "recall", "f1"]

REGRESSION_METRICS = {"rmse", "r2"}
CLASSIFICATION_METRICS = {"accuracy", "precision", "recall", "f1"}
MAXIMIZE_METRICS = {"r2", "accuracy", "precision", "recall", "f1"}


def is_regression(metric: Metric) -> bool:
    """
    Checks whether the specified metric is used for regression tasks.

    Regression metrics, such as RMSE and R², evaluate the performance of models
    that predict continuous outcomes. This function determines if the given metric
    belongs to the category of regression metrics.

    Args:
        metric (Metric): The performance metric to evaluate.

    Returns:
        bool:
            - True if the metric is a regression metric (e.g., RMSE, R²).
            - False if the metric is a classification metric (e.g., accuracy,
                    precision).

    Raises:
        ValueError: If the metric does not belong to either regression or
                    classification.
    """
    if metric in REGRESSION_METRICS:
        return True
    if metric in CLASSIFICATION_METRICS:
        return False
    raise ValueError(f"Invalid metric type: {metric}")


def is_maximizing_metric(metric: Metric) -> bool:
    """
    Checks if the specified metric is one where higher values indicate better
    performance.

    Some metrics, such as accuracy or R², are considered "maximizing metrics" because
    higher values represent better outcomes. This function identifies whether the given
    metric belongs to this category.

    Args:
        metric (Metric): The metric to check.

    Returns:
        bool: True if the metric is a maximizing metric, False otherwise.
    """
    return metric in MAXIMIZE_METRICS
