import copy
import glob
import os
import re
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Optional, Union

import gym
from gym.utils.seeding import create_seed
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    CheckpointCallback,
    EveryNTimesteps,
)
from stable_baselines3.common.vec_env import VecNormalize, VecEnv

from baselines_lab3 import utils
from baselines_lab3.env import create_environment
from baselines_lab3.utils import util

BEST_MODEL_NAME = "best_model"
BEST_NORMALIZATION_NAME = "best_model_norm"
MODEL_NAME = "rl_model"
NORMALIZATION_NAME = "norm"


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
    def __init__(
        self,
        path: Union[str, Path],
        name_prefix: str,
        n_keep: int = 5,
        verbose: int = 0,
    ):
        super(RemoveOldCheckpoints, self).__init__(verbose)
        self.path = Path(path)
        self.prefix = name_prefix
        self.n_keep = n_keep
        self.regex = re.compile(f"{name_prefix}_(\\d*)")

    def _on_step(self) -> bool:
        files = list(self.path.glob(self.prefix + "*"))
        counts = [int(self.regex.search(str(file)).group(1)) for file in files]

        # Sort files by number of steps
        files = [f for _, f in sorted(zip(counts, files), key=itemgetter(0))]
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
    :param training_env: The training environment. Required if a VecNormalization wrapper is used.
    :param tb_log: Set to true if the evaluation results should be logged. (Only works with keep_best=True)
    """

    def __init__(
        self,
        model_dir: str,
        save_interval: int = 250000,
        n_keep: int = 3,
        keep_best: bool = True,
        n_eval_episodes: int = 32,
        config: Optional[Dict[str, Any]] = None,
        training_env: Optional[Union[gym.Env, VecEnv]] = None,
        tb_log: bool = False,
        verbose: int = 0,
    ):

        callbacks = []

        self.model_dir = model_dir
        self.save_interval = save_interval
        self.n_keep = n_keep

        if keep_best or training_env:
            assert (
                config
            ), "You must provide an environment configuration to evaluate the model!"

        self.last_models = []
        self.wrappers = []
        self.tb_log = tb_log

        norm_wrapper = None
        if config:
            normalization = config["env"].get("normalize", False)

            if normalization:
                norm_wrapper = util.unwrap_vec_env(training_env, VecNormalize)

            # The eval callback will automatically synchronize VecNormalize wrappers if they exist
            if keep_best:
                eval_cb = EvalCallback(
                    eval_env=self._create_eval_env(config),
                    n_eval_episodes=n_eval_episodes,
                    eval_freq=max(save_interval // training_env.num_envs, 1),
                    log_path=self.model_dir,
                    best_model_save_path=self.model_dir,
                    verbose=verbose,
                    callback_on_new_best=NormalizationCheckpointCallback(
                        model_path=model_dir,
                        wrapper=norm_wrapper,
                        name=BEST_NORMALIZATION_NAME,
                    )
                    if norm_wrapper
                    else None,
                )
                callbacks.append(eval_cb)

        save_cb = self._create_checkpoint_callback(
            model_dir, norm_wrapper, save_interval, verbose
        )
        callbacks.append(save_cb)
        super(CheckpointManager, self).__init__(callbacks)

    def _create_checkpoint_callback(
        self,
        model_dir: str,
        norm_wrapper: Optional[VecNormalize],
        save_interval: int,
        verbose: int,
    ):
        checkpoints = [
            CheckpointCallback(save_freq=1, save_path=self.model_dir, verbose=verbose,),
            NormalizationCheckpointCallback(
                model_path=model_dir,
                wrapper=norm_wrapper,
                name=NORMALIZATION_NAME,
                checkpoints=True,
            )
            if norm_wrapper
            else None,
            RemoveOldCheckpoints(
                path=model_dir, name_prefix=MODEL_NAME, n_keep=self.n_keep
            ),
            RemoveOldCheckpoints(
                path=model_dir, name_prefix=NORMALIZATION_NAME, n_keep=self.n_keep
            ),
        ]
        save_cb = EveryNTimesteps(
            n_steps=save_interval,
            callback=CallbackList([cb for cb in checkpoints if cb is not None]),
        )
        return save_cb

    def _create_eval_env(self, config):
        test_config = copy.deepcopy(config)
        test_env_config = test_config["env"]

        test_env_config["n_envs"] = min(test_env_config.get("n_envs", 8), 32)

        evaluation_specific = test_env_config.get("evaluation", {})
        test_env_config.update(evaluation_specific)

        # if test_env_config.get("log_attributes", None):
        #     del test_env_config["log_attributes"]

        return utils.wrap_env(
            create_environment(test_config, create_seed()), monitor_wrapper=True
        )

    @classmethod
    def get_checkpoint(
        cls, path: str, type: str = "best", trial: int = -1
    ) -> Dict[str, Union[str, int]]:
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
            checkpoint = os.path.join(cp_path, f"{BEST_MODEL_NAME}.zip")
            normalization = os.path.join(cp_path, f"{BEST_NORMALIZATION_NAME}.pkl")
            return cls._create_checkpoint(checkpoint, normalization)
        elif type == "last":
            models = glob.glob(os.path.join(cp_path, f"{MODEL_NAME}*"))
            models.sort()
            norms = glob.glob(os.path.join(cp_path, f"{NORMALIZATION_NAME}*"))
            norms.sort()
            return cls._create_checkpoint(models[-1], norms[-1])
        else:
            raise ValueError(f"Unkown checkpoint type {type}.")

    @classmethod
    def _create_checkpoint(
        cls, checkpoint: str, normalization: Optional[str] = None,
    ) -> Dict[str, Union[str, int]]:
        if normalization:
            normalization = normalization if os.path.exists(normalization) else None
        return {"model": checkpoint, "normalization": normalization}

    @staticmethod
    def get_latest_run(path):
        runs = os.listdir(path)
        runs.sort()
        return os.path.join(path, runs[-1])  # Return latest run
