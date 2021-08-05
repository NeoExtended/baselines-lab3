import logging
from collections import deque
from typing import List

import cv2
import gym
import numpy as np

from algorithms.gym_maze_wrapper import GymMazeWrapper
from algorithms.particle_moving_algorithm import ParticleMovingAlgorithm


class ReplayWrapper(gym.Wrapper):
    """
    This wrapper replays the moves made by a particle moving algorithm step by step to another env.
    This simulates the behaviour of a RL model and can be used to generate an expert dataset for pretraining.
    """
    def __init__(self, replay_env: gym.Env, calculation_env: GymMazeWrapper, prep_algs: List[ParticleMovingAlgorithm], alg: ParticleMovingAlgorithm,
                 target_alg: ParticleMovingAlgorithm, frame_stack: bool = False, downscale: bool = False):
        super().__init__(replay_env)
        self.prep_algs = prep_algs
        self.alg = alg
        self.target_alg = target_alg
        self.env_copy = calculation_env

        self.action_counter = 0
        self.movement_buffer = []
        self.frame_stack = frame_stack
        self.downscale = downscale

        if downscale:
            # TODO: Make configurable
            self.width = 84
            self.height = 84
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                                    dtype=self.observation_space.dtype)
        if frame_stack:
            # TODO: Make configurable
            n_frames = 4
            self.n_frames = n_frames
            self.frames = deque([], maxlen=n_frames)
            shp = self.observation_space.shape
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                                dtype=self.observation_space.dtype)

    def seed(self, seed=None):
        super(ReplayWrapper, self).seed(seed)
        self.env_copy.seed(seed)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        self.movement_buffer = []
        self.action_counter = 0
        self.env_copy.reset()
        self._run_algorithms()

        if self.downscale:
            obs = self._downscale_frame(obs)

        if self.frame_stack:
            for _ in range(self.n_frames):
                self.frames.append(obs)
            return self._get_stacked_obs()
        return obs

    def _get_stacked_obs(self):
        assert len(self.frames) == self.n_frames
        return np.concatenate(list(self.frames), axis=2).astype(self.observation_space.dtype)

    def _run_algorithms(self):
        logging.info("Precalculating moves...")
        for alg in self.prep_algs:
            logging.info("Running preprocessing algorithm {}".format(alg))
            alg.reset_and_run()
            self.movement_buffer.extend(alg.get_movements())

        logging.info("Running algorithm {}".format(self.alg))
        self.alg.reset_and_run()
        self.movement_buffer.extend(self.alg.get_movements())
        logging.info("Running target algorithm.")
        self.target_alg.reset_and_run()
        self.movement_buffer.extend(self.target_alg.get_movements())

    def step(self, action):
        obs, reward, done, info = super(ReplayWrapper, self).step(action)
        if self.downscale:
            obs = self._downscale_frame(obs)

        if self.frame_stack:
            self.frames.append(obs)
            return self._get_stacked_obs(), reward, done, info
        else:
            return obs, reward, done, info


    def next(self, obs):
        #self.env.render(mode="human")
        self.action_counter += 1
        return self.env_copy.action_to_number(self.movement_buffer[self.action_counter-1])

    def _downscale_frame(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]