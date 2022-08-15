import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# logging.getLogger('matplotlib.font_manager').disabled = True
from baselines_lab3.utils import util
from baselines_lab3.utils.tensorboard.log_reader import (
    interpolate,
    LogReader,
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


class Plotter:
    """
    Class for automated plot creation from tensorboard log files.

    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    """

    def __init__(self, output_path: str, file_format: str = "pdf") -> None:
        self.path = util.create_log_directory(output_path)
        self.file_format = file_format
        self.cmap = plt.get_cmap("tab10")
        self.has_data = False

    def from_reader(
        self,
        reader: LogReader,
        tags: List[str],
        names: List[str],
        y_labels: Optional[List[str]] = None,
        x_label: str = "Steps",
        alias: Optional[Dict[str, str]] = None,
        plot_avg_only: bool = False,
        smoothing: float = 0.6,
        max_step: Optional[int] = None,
        trial: Optional[Union[int, List[int]]] = None,
        step_type: str = "step",
    ):
        """
        Creates and saves the plots defined by the given tags.

        :param y_labels: (Optional[List[str]]) Labels for the y axis.
        :param smoothing: (float) Factor for the exponential weighted average smoothing.
        :param plot_avg_only: (bool) Weather or not to only plot the average for runs with multiple trials, or additional std around.
        :param tags: (List[str]) Tags which correspond to summary tag names in the tensorboard logs
        :param names: (List[str]) Names for the tags. Will be used as ylabel in the plot and as file name.
        :param step_type: (str) Type of the x-axis ("step" for steps and "time" for minutes)
        """
        y_labels = y_labels if y_labels is not None else names
        assert (
            len(tags) == len(names) == len(y_labels)
        ), "There must be a name for each tag and vise versa!"
        if alias:
            assert len(alias) == len(
                reader.logs
            ), "There must be an alias for every log directory!"

        if len(reader.logs) > 10:
            self.cmap = plt.cm.get_cmap("hsv", len(reader.logs) + 5)

        logging.info("Creating plots.")
        values = reader.read_data(tags, max_step=max_step, step_type=step_type)

        logging.info("Saving plots to {}.".format(self.path))

        alias_map = self._map_alias(alias, reader)
        groups = self.get_ordered_groups(alias, reader)

        for tag, name, label in zip(tags, names, y_labels):
            self.prepare_plot(x_label, label, name)

            for i, group in enumerate(groups):
                step_data, value_data = self.prepare_data(group, tag, values)
                legend_label = alias_map[group]
                if len(step_data[0]) == 0:
                    continue
                if trial is not None:
                    if isinstance(trial, list):
                        step_data, value_data = (
                            np.asarray(step_data[trial[i]]),
                            np.asarray(value_data[trial[i]]),
                        )
                    else:
                        step_data, value_data = (
                            np.asarray(step_data[trial]),
                            np.asarray(value_data[trial]),
                        )

                self.add_plot(
                    step_data,
                    value_data,
                    self.cmap(i),
                    legend_label,
                    plot_avg_only,
                    smoothing,
                )

            if self.has_data:
                plt.ylim(bottom=0)
                if alias:
                    plt.legend()
                self.save_fig(
                    str(
                        self.path.joinpath(
                            "{}.{}".format(name.replace(" ", "_"), self.file_format)
                        )
                    )
                )

    def get_ordered_groups(self, alias: Dict[str, str], reader: LogReader):
        groups = list(alias.keys())
        for group in reader.logs.keys():
            if group not in groups:
                groups.append(group)
        return groups

    def prepare_data(
        self,
        group: str,
        tag: str,
        values: Dict[str, Dict[str, Tuple[List[int], List[int]]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        step_data, value_data = values[group][tag]
        min_length = min(step_data, key=lambda x: len(x))
        step_data = [l[: len(min_length)] for l in step_data]
        value_data = [l[: len(min_length)] for l in value_data]
        step_data, value_data = np.asarray(step_data), np.asarray(value_data)
        return step_data, value_data

    def _map_alias(self, alias: Dict[str, str], reader: LogReader):
        alias_map = {
            group: dirs[0].parent.parent.parent.name
            for group, dirs in reader.logs.items()
        }
        if alias is not None:
            for group, dirs in reader.logs.items():
                if group in alias:
                    alias_map[group] = alias[group]

        return alias_map

    def add_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        color,
        label=None,
        plot_avg_only: bool = False,
        smoothing: float = 0.6,
    ):
        self.has_data = True
        if len(x.shape) > 1 or x.dtype == np.object:
            # Check if all rows in step data are equal. If not interpolate.
            if x.dtype == np.object or not (x == x[0]).all():
                x, y = interpolate(x, y)

            arr = np.array(y)
            if smoothing > 0:
                # arr = self._moving_average(arr, smoothing)
                for idx, run in enumerate(arr):
                    arr[idx] = self._smooth(arr[idx], weight=smoothing)
            mu = np.mean(arr, axis=0)
            min = np.min(arr, axis=0)
            max = np.max(arr, axis=0)
            if smoothing > 0:
                min = self._smooth(min, weight=0.2)
                max = self._smooth(max, weight=0.2)

            self._make_multi_plot(
                x[0], min, max, mu, color, avg_only=plot_avg_only, label=label
            )
        else:
            self._make_plot(x, y, color, smoothing=smoothing, label=label)

    def _make_multi_plot(self, x, min, max, mu, color, label=None, avg_only=False):
        if not avg_only:
            plt.fill_between(x, max, min, facecolor=color, alpha=0.2)
        plt.plot(x, mu, color=color, linewidth=1.0, label=label)

    def _make_plot(self, x, y, color, label=None, smoothing=0.0):
        if smoothing > 0.0:
            y = self._smooth(y, weight=smoothing)
        plt.plot(x, y, color=color, linewidth=1.0, label=label)

    def _moving_average(self, arr, n=3):
        if len(arr.shape) > 1:
            return np.apply_along_axis(
                lambda m: np.convolve(m, np.ones(n), "valid") / n, axis=1, arr=arr
            )
        else:
            return np.convolve(arr, np.ones(n), "valid") / n

    def _smooth(self, scalars, weight=0.6):
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = (
                last * weight + (1 - weight) * point
            )  # exponential moving average
            smoothed.append(smoothed_val)
            last = smoothed_val

        return smoothed

    def prepare_plot(self, xlabel=None, ylabel=None, name=None):
        self.close_plot()
        self.has_data = False
        plt.figure(figsize=(6.4, 3.2))

        if name:
            plt.title(name)
        plt.grid()

        if xlabel:
            plt.xlabel(xlabel)

        if ylabel:
            plt.ylabel(ylabel)

    def close_plot(self):
        plt.close()

    def save_fig(
        self, path=None, tight_layout=True, fig_extension=None, resolution=300
    ):
        if path is None:
            path = self.path
        if fig_extension is None:
            fig_extension = self.file_format

        if tight_layout:
            plt.tight_layout()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ticker.EngFormatter(sep="\N{THIN SPACE}"))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.savefig(path, format=fig_extension, dpi=resolution)
