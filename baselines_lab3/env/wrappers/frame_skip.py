from typing import Union

import gym


class FrameSkip(gym.Wrapper):
    def __init__(self, env: Union[gym.Env, gym.Wrapper], skip: int = 4):
        assert skip > 0
        self.skip = skip
        super(FrameSkip, self).__init__(env)

    def step(self, action):
        r = 0.0

        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            r += reward

            if done:
                break
        return obs, r, done, info
