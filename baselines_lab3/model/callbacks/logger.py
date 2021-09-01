import json
from collections import deque
from copy import deepcopy
from typing import Dict, Any, Optional, Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from baselines_lab3.utils import safe_mean, unwrap_vec_env

import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


class TensorboardLogger(BaseCallback):
    """
    Logs additional values into the tensorboard log. Can be used as a callback for all learning algorithms.

    :param smoothing: (int) Number of episodes over which the running average for episode length and return
        will be calculated.
    :param min_log_delay: (int) Minimum number of timesteps between log entries.
    :param config: (dict) Lab config. Will be logged into tensorboard log at step 0 if set.
    """

    def __init__(
        self,
        smoothing: int = 100,
        min_log_delay: int = 500,
        verbose: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super(TensorboardLogger, self).__init__(verbose)
        self.ep_len_buffer = deque(maxlen=smoothing)
        self.reward_buffer = deque(maxlen=smoothing)
        self.n_episodes = None
        self.last_timesteps = 0
        self.min_log_delay = min_log_delay
        self.config = config
        self.tb_formatter = None

    def _on_training_start(self) -> None:
        self._initialize()
        self._write_config()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_timesteps > self.min_log_delay:
            self.last_timesteps = self.num_timesteps
            self._retrieve_values()
            self._write_summary()
        return True

    def reset(self):
        """
        Resets the logger.
        """
        self.ep_len_buffer.clear()
        self.reward_buffer.clear()
        self.n_episodes = None

    def _initialize(self):
        """
        Initializes the logger in the first step by retrieving the number of used environments.
        """
        if isinstance(self.training_env, VecEnv):
            episode_rewards = self.training_env.env_method("get_episode_rewards")
            self.n_episodes = [0] * len(episode_rewards)
        else:
            self.n_episodes = [0]

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _retrieve_values(self):
        """
        This method makes use of methods from the Monitor environment wrapper to retrieve episode information
        independent of the used algorithm.
        """
        if isinstance(self.training_env, VecEnv):
            self._retrieve_from_vec_env(self.training_env)
        else:
            self._retrieve_from_env(self.training_env)

    def _retrieve_from_env(self, env):
        episode_rewards = env.get_episode_rewards()
        episode_lengths = env.get_episode_lengths()

        known = self.n_episodes[0]
        self.ep_len_buffer.extend(episode_lengths[known:])
        self.reward_buffer.extend(episode_rewards[known:])
        self.n_episodes[0] = len(episode_rewards)

    def _retrieve_from_vec_env(self, env):
        # Use methods indirectly if we are dealing with a vecotorized environment
        episode_rewards = env.env_method("get_episode_rewards")
        episode_lengths = env.env_method("get_episode_lengths")

        for i, (ep_reward, ep_length) in enumerate(
            zip(episode_rewards, episode_lengths)
        ):
            known = self.n_episodes[i]
            self.ep_len_buffer.extend(ep_length[known:])
            self.reward_buffer.extend(ep_reward[known:])
            self.n_episodes[i] = len(ep_reward)

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

    def _write_summary(self):
        if len(self.ep_len_buffer) > 0:
            self.logger.record(
                "episode_length/ep_length_mean", safe_mean(self.ep_len_buffer)
            )
            self.logger.record("reward/ep_reward_mean", safe_mean(self.reward_buffer))
