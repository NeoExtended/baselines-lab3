import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import yaml
import numpy as np
import tensorflow as tf

# logging.getLogger('matplotlib.font_manager').disabled = True

IMAGES_PATH = ".."


def read_summary_values(file, tags, max_step=None, step_type="step"):
    steps = [list() for tag in tags]
    values = [list() for tag in tags]
    begin = None
    end = 0
    for summary in tf.train.summary_iterator(file):
        if not begin:
            begin = summary.wall_time
        if max_step is not None and summary.step > max_step:
            continue
        if summary.wall_time > end:
            end = summary.wall_time
        for value in summary.summary.value:
            for i, tag in enumerate(tags):
                if tag in value.tag:
                    if step_type == "step":
                        steps[i].append(summary.step)
                    elif step_type == "time":
                        minutes = (summary.wall_time - begin) / 60
                        steps[i].append(minutes)
                    values[i].append(value.simple_value)
    delta = end - begin
    return {tag: (step, val) for tag, step, val in zip(tags, steps, values)}, delta


def interpolate(step_data: np.ndarray, value_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mins = [np.min(steps) for steps in step_data]
    maxs = [np.max(steps) for steps in step_data]
    min_step = np.min(mins)
    max_step = np.max(maxs)
    steps = len(step_data[0])
    space = np.linspace(min_step, max_step, steps)
    interpolated = list()
    for steps, values in zip(step_data, value_data):
        interpolated.append(np.interp(space, steps, values))

    return np.array([space] * len(step_data)), np.array(interpolated)


class LogReader(ABC):
    def __init__(self, log_dirs: List[str], log_name):
        self.logs = {}
        for file in log_dirs:
            self.logs[file] = list(Path(file).glob(log_name))

        self.values = None

    @abstractmethod
    def read_data(self, tags: List[str], max_step: Optional[int] = None, step_type: str = "step") -> Dict[str, Dict[str, Tuple[List[int], List[Any]]]]:
        """
        Reads data from the log.

        :param tags: Tags to load.
        :param max_step: Maximum number of steps to load.
        :return: Returns a dictionary with an entry for each directory. Each entry contains a dicionary with a key for each tag.
                Each tag then lists the entries for all time - value pairs.
        """
        pass


class TensorboardLogReader(LogReader):
    """
    Class for automated plot creation from tensorboard log files.
    :param file_format: (str) File format for the created plots.
    :param log_dir: (str) Root directory for the tensorboard logs.
    :param max_step: (int) Last step that is read from the log.
    """
    def __init__(self, log_dirs: List[str]) -> None:
        super(TensorboardLogReader, self).__init__(log_dirs, "**/events.out.tfevents.*")

        self.deltas = None

    def read_data(self, tags: List[str], max_step: Optional[int] = None, step_type: str = "step") -> Dict[str, Dict[str, Tuple[List[int], List[Any]]]]:
        self.values = {}
        self.deltas = {}
        for dir in self.logs:
            tag_values = {}
            deltas = []
            for log_file in self.logs[dir]:
                logging.info("Reading tensorboard logs from {}. This may take a while...".format(log_file))
                data, delta = read_summary_values(str(log_file), tags, max_step, step_type)
                deltas.append(delta)
                for tag in data:
                    if tag not in tag_values:
                        tag_values[tag] = (list(), list())
                    tag_values[tag][0].append(data[tag][0])
                    tag_values[tag][1].append(data[tag][1])

            self.values[dir] = tag_values
            self.deltas[dir] = deltas
        return self.values


class EvaluationLogReader(LogReader):
    """
    Class for evaluation log reading
    """

    def __init__(self, log_dirs: List[str]) -> None:
        super(EvaluationLogReader, self).__init__(log_dirs, "**/*.episode_information.yml")

    def read_data(self, tags: List[str], max_step: Optional[int] = None, step_type: str = "step") -> Dict[str, Dict[str, Tuple[List[int], List[Any]]]]:
        self.values = {}

        for dir in self.logs:
            tag_values = {}
            for log_file in self.logs[dir]:
                file = log_file.open("r")
                values = yaml.load(file)
                file.close()

                for tag in tags:
                    steps = []
                    data = []
                    for episode in values:
                        if max_step == None:
                            steps.append(episode["x"])
                            data.append(episode[tag])
                        else:
                            step = []
                            value = []
                            for s, v in zip(episode["x"], episode[tag]):
                                if s > max_step:
                                    break
                                step.append(s)
                                value.append(v)
                            steps.append(step)
                            data.append(value)
                    if tag not in tag_values:
                        tag_values[tag] = (list(), list())
                    tag_values[tag][0].extend(steps)
                    tag_values[tag][1].extend(data)
            self.values[dir] = tag_values
        return self.values
