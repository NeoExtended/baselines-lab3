import copy
import glob
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

import gym
from gym.utils.seeding import create_seed
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EventCallback,
    CallbackList,
    EvalCallback,
    CheckpointCallback,
    EveryNTimesteps,
)
from stable_baselines3.common.vec_env import VecNormalize, VecEnv

from baselines_lab3.env import create_environment
from baselines_lab3.env.evaluation import Evaluator
from baselines_lab3.utils import util


class NormalizationCheckpointCallback(BaseCallback):
    def __init__(
        self,
        model_path: str,
        wrapper: VecNormalize,
        name: str = "normalization",
        checkpoints: bool = False,
        verbose: int = 0,
    ):
        super(NormalizationCheckpointCallback, self).__init__(verbose)
        self.model_path = model_path
        self.wrapper = wrapper
        self.name = name
        self.checkpoints = checkpoints

    def _on_step(self) -> bool:
        name = f"{self.name}.pkl"
        if self.checkpoints:
            name = f"{self.name}_{self.num_timesteps}.pkl"
        self.wrapper.save(os.path.join(self.model_path, name))
        return True


class RemoveOldCheckpoints(BaseCallback):
    def __init__(self, path: str, name_prefix: str, n_keep: int = 5, verbose: int = 0):
        super(RemoveOldCheckpoints, self).__init__(verbose)
        self.path = path
        self.prefix = name_prefix
        self.n_keep = n_keep

    def _on_step(self) -> bool:
        files = glob.glob(os.path.join(self.path, self.prefix) + "*")
        files.sort()
        if len(files) > self.n_keep:
            for f in files[: len(files) - self.n_keep]:
                os.remove(f)
        return True


class CheckpointManager(CallbackList):
    """
    Class to manage model checkpoints.
    :param model_dir: (str) Target directory for all the checkpoints.
    :param save_interval: (int) Interval at which models will be saved. Note that the real interval may depend on the
        frequency of calls to the step() function.
    :param n_keep: Number of models to keep. when saving the newest model n+1 the oldest will be deleted automatically.
    :param n_eval_episodes: Number of episodes used for model evaluation.
    :param keep_best: Whether or not to also save the best model. The best model is determined by running a test each
        time a new model is saved. This may take some time.
    :param config: Current lab configuration. Needed to create an evaluation environment if keep_best=True
    :param env: If the lab environment uses a running average normalization like VecNormalize, the running averages of
        the given env will be saved along with the model.
    :param tb_log: Set to true if the evaluation results should be logged. (Only works with keep_best=True)
    """

    def __init__(
        self,
        model_dir: str,
        save_interval: int = 250000,
        n_keep: int = 5,
        keep_best: bool = True,
        n_eval_episodes: int = 32,
        eval_method: str = "normal",
        config: Optional[Dict[str, Any]] = None,
        env: Optional[Union[gym.Env, VecEnv]] = None,
        tb_log: bool = False,
        verbose: int = 0,
    ):

        callbacks = []

        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep

        if keep_best or env:
            assert (
                config
            ), "You must provide an environment configuration to evaluate the model!"

        self.last_models = []
        self.wrappers = []
        self.tb_log = tb_log

        norm_wrapper = None
        if config:
            env_desc = copy.deepcopy(config["env"])
            normalization = env_desc.get("normalize", False)

            if normalization:
                norm_wrapper = util.unwrap_vec_env(env, VecNormalize)

            # The eval callback will automatically synchronize VecNormalize wrappers if they exist
            if keep_best:
                eval_cb = EvalCallback(
                    eval_env=self._create_eval_env(config),
                    n_eval_episodes=n_eval_episodes,
                    eval_freq=max(save_interval // env.num_envs, 1),
                    log_path=self.model_dir,
                    best_model_save_path=self.model_dir,
                    verbose=verbose,
                    callback_on_new_best=NormalizationCheckpointCallback(
                        model_path=model_dir,
                        wrapper=norm_wrapper,
                        name="best_model_norm",
                    )
                    if norm_wrapper
                    else None,
                )
                callbacks.append(eval_cb)

        save_cb = EveryNTimesteps(
            n_steps=save_interval,
            callback=CallbackList(
                [
                    CheckpointCallback(
                        save_freq=1, save_path=self.model_dir, verbose=verbose,
                    ),
                    NormalizationCheckpointCallback(
                        model_path=model_dir,
                        wrapper=norm_wrapper,
                        name="norm",
                        checkpoints=True,
                    )
                    if norm_wrapper
                    else None,
                    RemoveOldCheckpoints(
                        path=model_dir, name_prefix="rl_model", n_keep=self.n_keep
                    ),
                    RemoveOldCheckpoints(
                        path=model_dir, name_prefix="norm", n_keep=self.n_keep
                    ),
                ]
            ),
        )
        callbacks.append(save_cb)
        super(CheckpointManager, self).__init__(callbacks)

    def _create_eval_env(self, config):
        test_config = copy.deepcopy(config)
        test_env_config = test_config["env"]

        test_env_config["n_envs"] = min(test_env_config.get("n_envs", 8), 32)

        evaluation_specific = test_env_config.get("evaluation", {})
        test_env_config.update(evaluation_specific)

        return create_environment(test_config, create_seed())

    @classmethod
    def get_checkpoint(cls, path: str, type: str = "best", trial: int = -1) -> dict:
        """
        Returns a dictionary defining a model checkpoint. The checkpoint may contain more than one archive for different objects.
        :param path: (str) Path to a log directory (should contain subdirectories for each run).
        :param type: (str) Type of the checkpoint ("last" or "best").
        :param trial: (int) Trial to get the checkpoint from. Defaults to last trial.
        :return: (dict) Dictionary containing information about the checkpoint.
        """
        trials = sorted(
            [
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name)) and "trial" in name
            ]
        )
        if len(trials) > 0:
            cp_path = os.path.join(path, trials[trial], "checkpoints")
        else:
            cp_path = os.path.join(path, "checkpoints")
        assert os.path.exists(cp_path), "No checkpoints directory found in {}".format(
            path
        )

        if type == "best":
            model_suffix = "best"
        else:
            model_suffix = ""

        return cls._get_latest_checkpoint(cp_path, prefix="model", suffix=model_suffix)

    @classmethod
    def get_file_path(
        cls, checkpoint: dict, archive_name: str, extension: str = "zip"
    ) -> str:
        """
        Returns a file according to the given checkpoint and the requested archive of that checkpoint.
        :param checkpoint: (dict) The checkpoint.
        :param archive_name: (str) The name of the archive from the given checkpoint.
        :param extension: (str) The extension of the archive.
        :return: (str) Path to the requested archive of the given checkpoint.
        """
        file_name = cls._build_filename(checkpoint, archive_name, extension=extension)
        file_path = os.path.join(checkpoint["path"], file_name)
        assert os.path.exists(
            file_path
        ), "Could not find archive {} in the given checkpoint".format(file_name)
        return file_path

    @staticmethod
    def get_latest_run(path):
        runs = os.listdir(path)
        runs.sort()
        return os.path.join(path, runs[-1])  # Return latest run

    @staticmethod
    def _get_latest_checkpoint(path, prefix="", suffix=""):
        files = os.listdir(path)

        latest = datetime.fromisoformat("1970-01-01")
        counter = None
        for savepoint in files:
            datestring = os.path.splitext(savepoint)[0]
            if not (datestring.startswith(prefix) and datestring.endswith(suffix)):
                continue

            if len(prefix) > 0:
                datestring = datestring[len(prefix) + 1 :]
            if len(suffix) > 0:
                datestring = datestring[: -(len(suffix) + 1)]

            step, datestring = datestring.split("_", maxsplit=1)
            # If no suffix is given the datestring may contain invalid data.
            if len(datestring) > 17:
                continue

            date = datetime.strptime(datestring, util.TIMESTAMP_FORMAT)
            if date > latest:
                latest = date
                counter = step

        return CheckpointManager._checkpoint(
            path, suffix, counter, latest.strftime(util.TIMESTAMP_FORMAT)
        )

    @staticmethod
    def _checkpoint(path: str, suffix: str, counter: int, time: str) -> dict:
        """
        Creates a dictionary object describing a checkpoint.
        :param path: (str) Directory containing the checkpoint files.
        :param suffix: (str) Suffix for all the files of the checkpoint.
        :param counter: (int) Number of steps
        :param time: (str) Timestamp
        :return: (dict) Dictionary containing the checkpoint information
        """
        return {"path": path, "counter": counter, "time": time, "type": suffix}
