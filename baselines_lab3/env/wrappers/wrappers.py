from typing import Union, Dict

import cv2
import gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize


class WarpGrayscaleFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        """
        Warp grayscale frames to 84x84 as done in the Nature paper and later work.

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.n_channels = env.observation_space.shape[2]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, self.n_channels),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        # cv2 removes the single channel axis.
        if self.n_channels == 1:
            return frame[:, :, np.newaxis]
        return frame


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds gaussian noise to the observation
    """

    def __init__(self, env, scale=0.01):
        super(ObservationNoiseWrapper, self).__init__(env)
        self.var = scale
        self.sigma = scale ** 0.5
        self.mean = 0

    def observation(self, observation):
        gauss = np.random.normal(self.mean, self.sigma, observation.shape)
        result = np.clip(observation + gauss * 255, 0, 255).astype(
            self.observation_space.dtype
        )
        return result


class RepeatedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, probability=0.1):
        super(RepeatedActionWrapper, self).__init__(env)
        self.probability = probability
        self.last_action = 0
        self.true_action = 0

    def action(self, action):
        self.true_action = action
        if np.random.rand() < self.probability:
            return self.last_action
        else:
            self.last_action = action
            return action

    def reverse_action(self, action):
        return action


class VecScaledFloatFrame(VecEnvWrapper):
    """
    Scales image observations to [0.0, 1.0]. May be less memory efficient due to float conversion.
    """

    def __init__(self, env, dtype=np.float16):
        self.dtype = dtype
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=env.observation_space.shape, dtype=dtype
        )
        VecEnvWrapper.__init__(self, env, observation_space=self.observation_space)

    def reset(self):
        obs = self.venv.reset()
        return self._scale_obs(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self._scale_obs(obs), rews, dones, infos

    def _scale_obs(self, obs):
        return np.array(obs).astype(self.dtype) / 255.0


class VecStepSave(VecEnvWrapper):
    def __init__(self, env):
        super(VecStepSave, self).__init__(env)
        self.last_obs = None
        self.last_rews = None
        self.last_infos = None
        self.last_dones = None

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.last_obs = obs
        self.last_rews = rews
        self.last_dones = dones
        self.last_infos = infos
        return obs, rews, dones, infos


class VecSynchronizedNormalize(VecNormalize):
    def __init__(
        self,
        venv: VecEnv,
        source: VecNormalize,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):

        super(VecSynchronizedNormalize, self).__init__(
            venv, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon
        )
        self.source = source
        self.training = False

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        self.obs_rms = self.source.obs_rms
        self.ret_rms = self.source.ret_rms
        obs = self.venv.reset()
        return obs
