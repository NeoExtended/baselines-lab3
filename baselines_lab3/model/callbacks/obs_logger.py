from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional

from baselines_lab3.utils.util import unwrap_vec_env
from baselines_lab3.env.wrappers import VecStepSave

import numpy as np


class ObservationLogger(BaseCallback):
    def __init__(
        self,
        render_all: bool = False,
        random_render: bool = True,
        random_render_interval: int = 25000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.step_save = None  # type: Optional[VecStepSave]
        self.writer = None
        self.random_render_interval = random_render_interval
        self.random_render = random_render
        self.render_all = render_all
        self.last_interval = 0
        self.next_render_interval = np.random.randint(1, random_render_interval)
        self.next_render = 0

    def _on_training_start(self) -> None:
        self.step_save = unwrap_vec_env(self.training_env, VecStepSave)
        if not isinstance(self.step_save, VecStepSave):
            raise ValueError(
                "The observation logger requires the env to be wrapped with a step save wrapper!"
            )

    def _on_step(self) -> bool:
        if self.render_all:
            for i, done in enumerate(self.step_save.last_dones):
                if done:
                    self._render_obs(i)
        else:
            if self.step_save.last_dones[0]:
                self._render_obs(0)

        if self.random_render:
            self._random_render()

        return True

    def _render_obs(self, env_id):
        self.logger.add_image(
            "obs",
            self.step_save.last_infos[env_id]["terminal_observation"],
            self.num_timesteps,
        )

    def _random_render(self):
        if self.num_timesteps >= self.next_render:
            img = self.training_env.render("rgb_array")
            self.logger.add_image("render_result", img, self.num_timesteps)

            next_interval = np.random.randint(1, self.random_render_interval)
            self.next_render = (
                self.num_timesteps
                + (self.random_render_interval - self.last_interval)
                + next_interval
            )
            self.last_interval = next_interval
