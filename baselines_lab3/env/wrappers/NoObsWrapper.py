import gym
import numpy as np


class NoObsWrapper(gym.Wrapper):
    def __init__(self, env, rew_is_obs=False):
        """
        Deletes the observation from the env and replaces it with a counter which increases with each step.
        """
        gym.Wrapper.__init__(self, env)
        self.counter = np.array([0])
        self.rew_is_obs = rew_is_obs
        if rew_is_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.int64
            )

    def step(self, action):
        obs, rew, done, info = super(NoObsWrapper, self).step(action)
        self.counter[0] += 1
        if self.rew_is_obs:
            return np.array(self.counter[0], rew), rew, done, info
        else:
            return self.counter, rew, done, info

    def reset(self, **kwargs):
        super(NoObsWrapper, self).reset(**kwargs)
        self.counter[0] = 0

        if self.rew_is_obs:
            return np.array([self.counter[0], 0])
        else:
            return self.counter
