import logging
from typing import Union, Optional

import gym
import numpy as np
import torch
import cv2
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.base_vec_env import tile_images


class RenderLogger(BaseCallback):
    def __init__(
        self,
        render_env: Union[gym.Env, VecEnv],
        include_obs: bool = True,
        n_episodes: Optional[int] = None,
        render_mode: str = "rgb_array",
        deterministic: bool = True,
        fps: int = 25,
        verbose: int = 0,
    ):
        super(RenderLogger, self).__init__(verbose=verbose)

        # Convert to VecEnv for consistency
        if not isinstance(render_env, VecEnv):
            render_env = DummyVecEnv([lambda: render_env])

        self.render_env = render_env
        self.n_episodes = (
            n_episodes if n_episodes is not None else self.render_env.num_envs
        )
        self.deterministic = deterministic
        self.fps = fps

        # Check available render modes
        self.render_mode = render_mode
        render_modes_avail = render_env.metadata.get("render.modes")
        if render_modes_avail is not None:
            assert (
                render_mode in render_modes_avail
            ), f"The choosen render mode {render_mode} is not available for the given environment!"
        else:
            logging.warning(
                "The environment does not provide any render modes. Rendering might fail."
            )
        self.tb_formatter = None
        self.include_obs = include_obs

    def _on_training_start(self) -> None:
        self._initialize()

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

    def _on_step(self) -> bool:
        images = []
        observations = []

        def callback(locals_, globals_):
            images.append(locals_["env"].render(mode=self.render_mode))
            observations.append(locals_["observations"])

        if self.verbose > 0:
            logging.info("Logging evaluation video.")

        evaluate_policy(
            self.model,
            self.render_env,
            n_eval_episodes=self.n_episodes,
            deterministic=self.deterministic,
            callback=callback,
        )

        if self.include_obs:
            images = [
                self.hconcat_resize_max(
                    [
                        img,
                        self.expand_channels(
                            tile_images(np.transpose(obs, (0, 2, 3, 1)))
                        ).astype(np.uint8),
                    ]
                )
                for img, obs in zip(images, observations)
            ]

        images = np.transpose(np.array(images), (0, 3, 1, 2))[np.newaxis, :, :, :, :]
        self.tb_formatter.writer.add_video(
            "env/render",
            torch.from_numpy(images),
            global_step=self.num_timesteps,
            fps=self.fps,
        )

        # image = self.training_env.render(mode="rgb_array")
        return True

    def expand_channels(self, image):
        n_channels = image.shape[2]

        if n_channels == 1:
            image = np.concatenate([image] * 3, axis=2)
        elif n_channels == 2:
            image = np.concatenate(
                [image, np.zeros((image.shape[0], image.shape[1], 1))], axis=2
            )
        elif n_channels == 3:
            return image
        elif n_channels > 3:
            image = image[:, :, :3]
        return image

    def hconcat_resize_max(self, im_list, interpolation=cv2.INTER_CUBIC):
        h_min = max(im.shape[0] for im in im_list)  # Assumes HxWxC
        im_list_resize = [
            cv2.resize(
                im,
                (int(im.shape[1] * h_min / im.shape[0]), h_min),
                interpolation=interpolation,
            )
            for im in im_list
        ]
        return cv2.hconcat(im_list_resize)
