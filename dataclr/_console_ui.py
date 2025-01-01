from __future__ import annotations

import sys
from datetime import datetime

from tqdm import tqdm

from dataclr.metrics import Metric, is_maximizing_metric
from dataclr.results import Result


class ConsoleUI:
    def __init__(self) -> None:
        self.max_line_length: int = 110
        self.cleaner: str = " " * (self.max_line_length + 10)

        self.reset_ui()

    def _init_pbar(self, init_combinations: int, metric: Metric) -> None:
        self.start_time = datetime.now()
        self.metric = metric
        self.pbar = tqdm(
            total=init_combinations, desc="Feature selection process", unit="step"
        )

    def _truncate_string(self, label: str, string: str) -> str:
        string = label + string
        max_content_length = self.max_line_length - len(label)
        if len(string) > max_content_length:
            return string[: max_content_length - 3] + "..."
        return string

    def _send_result(
        self, result: Result, cur_methods: list[str], end: bool = False
    ) -> None:
        self.total_combinations += 1

        if result:
            self.last_performance = result.performance

        if not self.best_performance or (
            self.last_performance
            and is_maximizing_metric(self.metric)
            == (self.last_performance[self.metric] > self.best_performance[self.metric])
        ):
            self.best_performance = self.last_performance

        elapsed_time = datetime.now() - self.start_time
        if self.pbar is None:
            raise ValueError("Progress bar not initialized. Call `_init_pbar` first.")

        if not end:
            self.pbar.update(1)

        trunked_methods_list = self._truncate_string(
            "Current running methods: ", str(cur_methods)
        )

        def split_string(string: str) -> tuple[str, str]:
            parts = string.split("|")
            mid_index = len(parts) // 2

            part_1 = " | ".join(parts[:mid_index]).strip()
            part_2 = " | ".join(parts[mid_index:]).strip()
            return part_1, part_2

        recent_result_1, recent_result_2 = split_string(str(self.last_performance))
        best_result_1, best_result_2 = split_string(str(self.best_performance))

        status = (
            "\n"
            f"{self._truncate_string('Recent performance: ', recent_result_1)}\n"
            f"{self._truncate_string('                  | ', recent_result_2)}\n"
            f"{self._truncate_string('Best performance: ', best_result_1)}\n"
            f"{self._truncate_string('                 | ', best_result_2)}\n"
            f"Total combinations: {self.total_combinations}\n"
            f"Running time: {str(elapsed_time).split('.')[0]}\n"
            f"{trunked_methods_list}\n"
            "\n"
        )
        if self.total_combinations != 1:
            self._clear_last_n_lines(10)
        tqdm.write(status)

        if end:
            self.pbar.close()

    def _clear_last_n_lines(self, n_lines: int) -> None:
        for _ in range(n_lines):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
        sys.stdout.flush()

    def _increment_total_combinations(self, to_add: int) -> None:
        if self.pbar is None:
            raise ValueError("Progress bar not initialized. Call `_init_pbar` first.")

        self.pbar.total += to_add
        self.pbar.refresh()

    def reset_ui(self) -> None:
        self.start_time: datetime = None
        self.pbar: tqdm = None
        self.initialized: bool = True
        self.total_combinations: int = 0

        self.metric: Metric = None
        self.last_performance: Result = None
        self.best_performance: Result = None


console_ui = ConsoleUI()
