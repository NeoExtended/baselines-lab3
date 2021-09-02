import json
from typing import Dict, Any, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


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

    def _on_step(self) -> bool:
        pass

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
