from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

if __name__ == '__main__':
    env = make_vec_env("CartPole-v0", n_envs=1)
    env2 = make_vec_env("CartPole-v0", n_envs=8, vec_env_cls=SubprocVecEnv)
    # env = VecNormalize(env)
    # env = VecScaledFloatFrame(env)
    #env = VecFrameStack(env, n_stack=4)
    obs = env.reset()


    #model = PPO2(MlpPolicy, env, verbose=1)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env2.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env2.step(action)
        env2.render()