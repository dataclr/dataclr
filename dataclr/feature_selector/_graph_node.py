from __future__ import annotations

from dataclr._typing import DataSplits
from dataclr.feature_selector._hash import get_combination_hash
from dataclr.methods.method import Method
from dataclr.results import Result, ResultPerformance


class GraphNode:
    def __init__(
        self,
        feature_list: list[str],
        future_methods: set[Method],
        method: Method = None,
        result: Result = None,
        parent: GraphNode = None,
    ) -> None:
        self.feature_list = feature_list
        self.future_methods = future_methods
        self.method = method
        self.result = result
        self.parent = parent

    def get_results(
        self,
        data_splits: DataSplits,
        cached_results: dict[int, list[Result]],
        cached_performance: dict[int, ResultPerformance],
        method: Method,
    ) -> list[Result]:
        if method is None:
            raise ValueError("Error in get_results!")

        combination_hash = get_combination_hash({method}, self.feature_list)
        if combination_hash in cached_results:
            return cached_results[combination_hash]

        filtered_data_splits = DataSplits(
            X_train=data_splits["X_train"][self.feature_list],
            y_train=data_splits["y_train"],
            X_test=data_splits["X_test"][self.feature_list],
            y_test=data_splits["y_test"],
        )

        results = method._get_results(filtered_data_splits, cached_performance)
        cached_results[combination_hash] = results

        return results
