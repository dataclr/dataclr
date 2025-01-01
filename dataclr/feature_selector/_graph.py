from __future__ import annotations

import concurrent.futures
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

from dataclr._console_ui import console_ui
from dataclr._typing import DataSplits
from dataclr.feature_selector._graph_node import GraphNode
from dataclr.feature_selector._hash import get_combination_hash
from dataclr.methods._method_list import wrapper_classes
from dataclr.methods.method import Method
from dataclr.metrics import Metric, is_maximizing_metric
from dataclr.results import Result, ResultPerformance


class Graph:
    def __init__(
        self,
        data_splits: DataSplits,
        metric: Metric,
        method_set: set[Method],
        wrapper_method_set: set[Method],
        n_wrapper_results: int,
        n_jobs: int = 1,
        level_wrapper_results: int = 0,
        verbose: bool = True,
        max_depth: int = 3,
        start_wrappers: bool = True,
        level_cutoff_threshold: int = 300,
    ) -> None:
        self.data_splits = data_splits
        self.metric = metric
        self.method_set = method_set
        self.n_jobs = n_jobs
        self.wrapper_method_set = wrapper_method_set
        self.n_wrapper_results = n_wrapper_results
        self.final_wrapper_results_on_new_level = level_wrapper_results
        self.verbose = verbose
        self.max_depth = max_depth
        self.start_wrappers = start_wrappers
        self.level_cutoff_threshold = level_cutoff_threshold

        self.root_node = GraphNode(
            feature_list=list(data_splits["X_train"].columns),
            future_methods=self.method_set,
        )

        self.n_results: int = 3

        self.best_results: list[GraphNode] = None
        self.best_results_history: list[float] = None
        self.future_possibilities: dict[int, bool] = None
        self.cached_results: dict[int, list[Result]] = None
        self.cached_performance: dict[int, ResultPerformance] = None
        self.shared_stats: dict[str, int] = None

        self.cur_methods: list[str] = None

        self.results_lock = None

    def __is_combination_processed(
        self, method_set: set[Method], feature_list: list[str]
    ) -> bool:
        combination_hash = get_combination_hash(method_set, feature_list)

        if combination_hash in self.future_possibilities:
            return True

        self.future_possibilities[combination_hash] = True

        return False

    def __update_best_results(self, node: GraphNode) -> None:
        if self.results_lock is None:
            raise ValueError("Results lock undefined")

        with self.results_lock:
            self.best_results.append(node)
            self.best_results.sort(
                key=self._result_key, reverse=is_maximizing_metric(self.metric)
            )
            if len(self.best_results) > self.n_results:
                self.best_results.pop(-1)

            best_result = self.best_results[0].result

            if best_result is None or node.result is None:
                raise ValueError("Result is None")

            value = best_result.performance[self.metric]
            if value is None:
                raise ValueError(f"Performance criterion '{self.metric}' is None")

            self.best_results_history.append(value)

            if (
                best_result.performance[self.metric]
                == node.result.performance[self.metric]
            ):
                self.shared_stats["result_with_no_improvement"] = 0
            else:
                self.shared_stats["result_with_no_improvement"] += 1

    def _new_task(
        self,
        method: Method,
        future_methods: set[Method],
        node: GraphNode,
        depth: int,
    ):
        if (
            self.shared_stats["result_with_no_improvement"]
            > self.level_cutoff_threshold
        ):
            if self.shared_stats["depth_to_remove"] == -1:
                self.shared_stats["depth_to_remove"] = depth
            elif self.shared_stats["depth_to_remove"] == depth:
                return [], None, None
            else:
                self.shared_stats["result_with_no_improvement"] = 0
                self.shared_stats["depth_to_remove"] = -1

        self.cur_methods.append(method.__class__.__name__)

        new_method_set = future_methods - {method}
        new_node = GraphNode(node.feature_list, new_method_set, method)

        new_results = new_node.get_results(
            self.data_splits,
            self.cached_results,
            self.cached_performance,
            method,
        )

        future_params = []

        for result in new_results:
            new_result_node = GraphNode(
                feature_list=result.feature_list,
                future_methods=new_method_set,
                method=method,
                result=result,
                parent=node,
            )

            self.__update_best_results(new_result_node)

            if self.__is_combination_processed(new_method_set, result.feature_list):
                continue

            for future_method in new_method_set:
                if self.max_depth < 0 or depth < self.max_depth:
                    future_params.append(
                        [future_method, new_method_set, new_result_node, depth + 1]
                    )

        return future_params, new_results[0] if new_results else None, method

    def _result_key(self, node: GraphNode) -> float:
        if node.result is None:
            raise ValueError("Error in result_key!")

        value = node.result.performance[self.metric]
        if value is None:
            raise ValueError(f"Performance criterion '{self.metric}' is None")

        return value

    def _handle_starter_methods(self, executor: ProcessPoolExecutor):
        start_wrappers = []
        if self.start_wrappers:
            start_wrappers = self._handle_new_tasks_from_method_set(
                executor, self.wrapper_method_set, self.root_node, 1, True
            )

        future_tasks = self._handle_new_tasks_from_method_set(
            executor, self.method_set, self.root_node, 1
        )

        future_tasks = start_wrappers + future_tasks
        return future_tasks

    def _handle_new_depth_entrance(
        self, executor: ProcessPoolExecutor, results, cur_max_depth
    ):
        future_tasks = []
        new_max_depth = cur_max_depth
        for _, _, _, depth in results:
            if depth > cur_max_depth:
                new_max_depth = depth

                results_for_wrapper_methods = self.best_results[
                    : min(
                        self.final_wrapper_results_on_new_level,
                        len(self.best_results),
                    )
                ]
                for res in results_for_wrapper_methods:
                    new_wrapppers = self._handle_new_tasks_from_method_set(
                        executor, self.wrapper_method_set, res, depth, True
                    )
                    future_tasks = new_wrapppers + future_tasks
                break
        return future_tasks, new_max_depth

    def _handle_new_tasks_from_method_set(
        self,
        executor: ProcessPoolExecutor,
        method_set,
        root_node,
        cur_depth,
        only_one_level=False,
    ):
        new_tasks = [
            executor.submit(
                self._new_task,
                method,
                method_set if not only_one_level else {method},
                root_node,
                cur_depth,
            )
            for method in method_set
        ]
        return new_tasks

    def _handle_new_tasks(self, executor: ProcessPoolExecutor, results):
        new_tasks = [
            executor.submit(
                self._new_task,
                additional_method,
                additional_method_set,
                additional_result_node,
                depth=depth,
            )
            for (
                additional_method,
                additional_method_set,
                additional_result_node,
                depth,
            ) in results
        ]
        return new_tasks

    def _handle_after_filters_wrappers(self, executor):
        results_for_wrapper_methods = []
        for result in self.best_results:
            if len(results_for_wrapper_methods) >= self.n_wrapper_results:
                break
            if result.method.__class__ not in wrapper_classes:
                results_for_wrapper_methods.append(result)

        # with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
        future_tasks = []

        for result in results_for_wrapper_methods:
            future_tasks += self._handle_new_tasks_from_method_set(
                executor, self.wrapper_method_set, result, 1000, True
            )
        if self.verbose:
            console_ui._increment_total_combinations(len(future_tasks))

        while future_tasks:
            futures_to_process = list(future_tasks)
            for future in concurrent.futures.as_completed(futures_to_process):
                _, result, method = future.result()
                future_tasks.remove(future)
                if self.verbose:
                    console_ui._send_result(result, list(self.cur_methods))
                self.cur_methods.remove(method.__class__.__name__)

    def _search_feature_sets(self) -> None:
        cur_max_depth = 1
        with multiprocessing.Manager() as manager:
            self.best_results = manager.list()
            self.best_results_history = manager.list()
            self.future_possibilities = manager.dict()
            self.cached_results = manager.dict()
            self.results_lock = manager.Lock()
            self.cached_performance = manager.dict()
            self.shared_stats = manager.dict()
            self.cur_methods = manager.list()
            self.shared_stats["result_with_no_improvement"] = 0
            self.shared_stats["depth_to_remove"] = -1

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs
            ) as executor:
                future_tasks = self._handle_starter_methods(executor)
                if self.verbose:
                    console_ui._init_pbar(len(future_tasks), self.metric)

                while future_tasks:
                    done, _ = concurrent.futures.wait(
                        future_tasks, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        try:
                            results, result, method = future.result()
                            future_tasks.remove(future)
                            if self.verbose:
                                console_ui._send_result(result, list(self.cur_methods))
                            if method:
                                self.cur_methods.remove(method.__class__.__name__)

                            if results:
                                (
                                    new_future_tasks,
                                    cur_max_depth,
                                ) = self._handle_new_depth_entrance(
                                    executor, results, cur_max_depth
                                )
                                future_tasks = new_future_tasks + future_tasks
                                new_tasks = self._handle_new_tasks(executor, results)
                                future_tasks.extend(new_tasks)
                                if self.verbose:
                                    console_ui._increment_total_combinations(
                                        len(new_tasks)
                                    )
                        except Exception as e:
                            print(e)
                self._handle_after_filters_wrappers(executor)
            if self.verbose:
                console_ui._send_result(self.best_results[0].result, [], end=True)

            self.best_results_history = list(self.best_results_history)
            self.best_results = list(self.best_results)

    def _get_best_results(self, n_results=3) -> list[Result]:
        self.n_results = n_results

        start_time = time.perf_counter()
        self._search_feature_sets()
        duration = time.perf_counter() - start_time

        if self.best_results_history:
            best_performance = self.best_results_history[-1]
            print(f"\nBest found performance: {best_performance} in {duration}")
        else:
            print(f"No results found in {duration}")

        return self.best_results
