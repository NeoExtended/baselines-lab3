import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from baselines_lab3.env.environment import make_env
from baselines_lab3.env.wrappers import WarpGrayscaleFrame

if __name__ == "__main__":
    env_creation_fns = [
        make_env(
            "gym_maze:Maze0318Continuous-v0",
            {},
            i,
            42,
            "./logs/Labtest",
            [(WarpGrayscaleFrame, {}), (MaxAndSkipEnv, {"skip": 4})],
            evaluation=False,
        )
        for i in range(16)
    ]
    env = SubprocVecEnv(env_creation_fns)

    # env = make_vec_env("gym_maze:Maze0318Continuous-v0", n_envs=8)
    # env2 = make_vec_env("CartPole-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env)
    # env = VecScaledFloatFrame(env)
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()

    # model = PPO2(MlpPolicy, env, verbose=1)
    model = PPO("CnnPolicy", env, verbose=1, n_steps=2048, batch_size=1024)
    model.learn(total_timesteps=100000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
