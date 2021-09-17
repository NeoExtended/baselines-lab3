import gym
import numpy as np


class RecurrentActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(RecurrentActionWrapper, self).__init__(env)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action_obs = gym.spaces.Box(
                low=0, high=1, shape=(env.action_space.n,), dtype=int
            )
            self.shape = (env.action_space.n,)
        #    self.convert_box_actions = True
        # elif isinstance(env.action_space, gym.spaces.Box):
        #     action_obs = env.action_space
        #     self.convert_box_actions = False
        else:
            raise ValueError(f"Unsupported action space {type(env.action_space)}!")

        self.observation_space = gym.spaces.Dict(
            {"env_obs": env.observation_space, "last_action": action_obs}
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self.make_obs(obs, 0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.make_obs(observation, action), reward, done, info

    def make_obs(self, obs, action):
        one_hot_action = np.zeros(self.shape, dtype=int)
        one_hot_action[action] = 1
        return {"env_obs": obs, "last_action": one_hot_action}
