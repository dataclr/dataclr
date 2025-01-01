class ResultPerformance:
    """
    Represents the performance metrics of a model or result.

    This class serves as a base class for specific performance metrics, such as
    those for regression or classification tasks.

    Subclasses:
        - :class:`RegressionPerformance`
        - :class:`ClassificationPerformance`

    Attributes:
        r2 (float): Coefficient of determination (R²) score.
        rmse (float): Root Mean Squared Error.
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        f1 (float): F1 score.
    """

    def __init__(
        self,
        r2: float = None,
        rmse: float = None,
        accuracy: float = None,
        precision: float = None,
        recall: float = None,
        f1: float = None,
    ) -> None:
        self.rmse = rmse
        self.r2 = r2
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __getitem__(self, key: str) -> float:
        attributes = {
            "r2": self.r2,
            "rmse": self.rmse,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        if key not in attributes:
            raise KeyError(f"'{key}' is not a valid performance metric.")
        return attributes[key]

    def __str__(self) -> str:
        metrics = []
        if self.rmse is not None:
            metrics.append(f"RMSE: {self.rmse:.4f}")
        if self.r2 is not None:
            metrics.append(f"R2: {self.r2:.4f}")
        if self.accuracy is not None:
            metrics.append(f"Accuracy: {self.accuracy:.4f}")
        if self.precision is not None:
            metrics.append(f"Precision: {self.precision:.4f}")
        if self.recall is not None:
            metrics.append(f"Recall: {self.recall:.4f}")
        if self.f1 is not None:
            metrics.append(f"F1: {self.f1:.4f}")
        return " | ".join(metrics)


class RegressionPerformance(ResultPerformance):
    """
    Represents performance metrics for regression tasks.

    Attributes:
        r2 (float): Coefficient of determination (R²) score.
        rmse (float): Root Mean Squared Error.

    Inherits from:
        :class:`ResultPerformance`
    """

    def __init__(self, r2: float, rmse: float) -> None:
        super().__init__(r2=r2, rmse=rmse)


class ClassificationPerformance(ResultPerformance):
    """
    Represents performance metrics for classification tasks.

    Attributes:
        accuracy (float): Accuracy score.
        precision (float): Precision score.
        recall (float): Recall score.
        f1 (float): F1 score.

    Inherits from:
        :class:`ResultPerformance`
    """

    def __init__(
        self, accuracy: float, precision: float, recall: float, f1: float
    ) -> None:
        super().__init__(accuracy=accuracy, precision=precision, recall=recall, f1=f1)
