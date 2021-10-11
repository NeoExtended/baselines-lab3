import json
from operator import attrgetter
from typing import Dict, Any, Optional, Union
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean


class TensorboardLogger(BaseCallback):
    """
    Logs additional values into the tensorboard log. Can be used as a callback for all learning algorithms.

    :param config: (dict) Lab config. Will be logged into tensorboard log at step 0 if set.
    """

    def __init__(
        self, verbose: int = 0, config: Optional[Dict[str, Any]] = None,
    ):
        super(TensorboardLogger, self).__init__(verbose)
        self.config = config
        self.tb_formatter = None

        attributes = []
        tb_log = config["algorithm"].get("tensorboard_log", False)
        if tb_log:
            attributes = tb_log.get("attributes", [])

        self.done_attributes = {
            a: deque(maxlen=100) for a, t in attributes if t == "done"
        }
        self.step_attributes = {
            a: deque(maxlen=100) for a, t in attributes if t == "step"
        }
        self.info_attributes = {
            a: deque(maxlen=100) for a, t in attributes if t == "info"
        }

        self.single_env = config["env"].get("n_envs", 1)
        self.done = "dones" if config["algorithm"]["name"] in ["ppo", "a2c"] else "done"
        self.info = (
            "infos"  # if config["algorithm"]["name"] in ["ppo", "a2c"] else "info"
        )

    def _on_step(self) -> bool:
        if (
            len(self.step_attributes) > 0
            or len(self.done_attributes) > 0
            or len(self.info_attributes) > 0
        ):
            self._collect_data()

        return True

    def _on_rollout_end(self) -> None:
        if (
            len(self.step_attributes) > 0
            or len(self.done_attributes) > 0
            or len(self.info_attributes) > 0
        ):
            self._write_values()

    def _collect_data(self):
        dones = self.locals[self.done]
        for a in self.step_attributes:
            if "." in a:
                member, attributes = a.split(".", 2)
                objects = self.training_env.get_attr(member)
                self.step_attributes[a].extend(
                    [attrgetter(attributes)(o) for o in objects]
                )
            else:
                self.step_attributes[a].extend(self.training_env.get_attr(a))
        if dones.any():
            for a in self.done_attributes:
                if "." in a:
                    member, attributes = a.split(".", 2)
                    objects = self.training_env.get_attr(member)
                    self.done_attributes[a].extend(
                        [attrgetter(attributes)(o) for o, d in zip(objects, dones) if d]
                    )
                else:
                    self.done_attributes[a].extend(
                        [at for at, d in zip(self.training_env.get_attr(a), dones) if d]
                    )

            for a in self.info_attributes:
                self.info_attributes[a].extend(
                    [
                        i[a]
                        for i, d in zip(self.locals[self.info], dones)
                        if d and not i.get("TimeLimit.truncated", False)
                    ]
                )

    def _on_training_start(self) -> None:
        self._initialize()
        self._write_config()

    def _initialize(self):
        """
        Initializes the logger in the first step by retrieving the number of used environments.
        """
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _write_values(self):
        for key, value in self.step_attributes.items():
            self.logger.record(f"env/step/{key}", safe_mean(value))

        for key, value in self.done_attributes.items():
            self.logger.record(f"env/done/{key}", safe_mean(value))

        for key, value in self.info_attributes.items():
            self.logger.record(f"env/info/{key}", safe_mean(value))

    def _write_config(self):
        # hparams = deepcopy(self.config)
        # for key in hparams:
        #     if isinstance(hparams[key], dict):
        #         hparams[key] = json.dumps(hparams[key])

        # self.tb_formatter.writer.add_hparams(hparams, {"test": 0.0})
        self.tb_formatter.writer.add_text("config", json.dumps(self.config))
        # self.tb_formatter.writer.flush()

    def write_hparams(
        self,
        hparams: Dict[str, Union[int, str, float, bool]],
        metric_dict: Dict[str, Union[int, float]],
    ):
        self.tb_formatter.writer.add_hparams(hparams, metric_dict)
