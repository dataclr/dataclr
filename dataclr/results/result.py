from dataclr.results import ResultPerformance


class MethodResult:
    """
    This class provides a representation of the final result and the sequence
    of methods applied during the process. I

    Attributes:
        result (:class:`Result`): The final result of the feature selection or model
                                  evaluation.
        methods_list (list[:class:`~dataclr.methods.Method`]): A list of methods
                                        applied in the order they were executed.

    Args:
        node: A ``GraphNode`` object containing the result and its associated methods.
    """

    def __init__(self, node):
        self.result: Result = node.result
        self.methods_list = self._trace_methods(node)

    def _trace_methods(self, node) -> list:
        methods: list = []
        while node:
            if node.method:
                methods.append(node.method)
            node = node.parent
        return methods[::-1]

    def __str__(self) -> str:
        methods = ""
        for method in self.methods_list:
            methods += method.__class__.__name__ + " "
        return f"{methods} {self.result}"


class Result:
    """
    Represents the result of a feature selection or model evaluation process.

    Attributes:
        params (dict[str, object]): The parameters used by the method to achieve
               this result.
        performance (:class:`ResultPerformance`): The performance metrics of the result.
        feature_list (list[str]): A list of selected features.
    """

    def __init__(
        self,
        params: dict[str, object],
        performance: ResultPerformance,
        feature_list: list[str],
    ) -> None:
        self.params = params
        self.performance = performance
        self.feature_list = feature_list

    def __str__(self) -> str:
        return (
            f"Performance: {self.performance} | Feature count: {len(self.feature_list)}"
        )
