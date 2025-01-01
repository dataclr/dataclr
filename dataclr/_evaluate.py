from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from dataclr._typing import BaseModel, DataSplits
from dataclr.metrics import Metric, is_regression
from dataclr.results import ResultPerformance
from dataclr.results.performance import ClassificationPerformance, RegressionPerformance


def train_eval(
    model: BaseModel, metric: Metric, data_splits: DataSplits
) -> ResultPerformance:
    if len(data_splits["X_train"].columns) == 0:
        return ResultPerformance(
            rmse=float("inf"),
            r2=-float("inf"),
            accuracy=-float("inf"),
            precision=-float("inf"),
            recall=-float("inf"),
            f1=-float("inf"),
        )

    model.fit(data_splits["X_train"], data_splits["y_train"])
    y_pred = model.predict(data_splits["X_test"])
    y_test = data_splits["y_test"]

    if is_regression(metric):
        metrics = {
            "rmse": float(root_mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        return RegressionPerformance(**metrics)
    else:
        is_binary = len(set(y_test)) == 2
        average = "binary" if is_binary else "weighted"
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": float(
                precision_score(y_test, y_pred, average=average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, average=average, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        }
        return ClassificationPerformance(**metrics)
