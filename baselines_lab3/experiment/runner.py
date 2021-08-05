import numpy as np
from stable_baselines.common.vec_env import VecEnv

class Runner:
    """
    Executes the basic agent - environment interaction loop.
    :param env: (gym.Env or VecEnv) The environment for the agent
    :param agent: The agent to make predictions from observations. If None, random actions will be generated.
    :param render: (bool) Weather or not to render the environment during execution.
    :param deterministic: (bool) Weather or not the agent should make deterministic predictions.
    :param close_env: (bool) Weather or not the environment should be closed after a call to run().
    """
    def __init__(self, env, agent, render=True, deterministic=True, close_env=True, other_agents=None, agreement_type="action", random=False):
        self.env = env
        self.agent = agent
        self.deterministic = deterministic
        self.render = render
        self.close_env = close_env
        self.other_agents = other_agents if other_agents is not None else []
        self.agreement_type = agreement_type
        self.random = random

        self.episode_counter = 0
        self.step_counter = 0
        self.total_reward = 0
        self.kl_div = np.zeros((len(self.other_agents)+1, len(self.other_agents)+1))
        self.agreement = np.zeros((len(self.other_agents)+1, len(self.other_agents)+1))

        if isinstance(env, VecEnv):
            self.vec_env = True
        else:
            self.vec_env = False

    def run(self, n_episodes):
        """
        Simulates at least n_episodes of environment interactions.
        """
        obs = self.env.reset()

        while self.episode_counter < n_episodes:
            if not self.random:
                action, _states = self.agent.predict(obs, deterministic=self.deterministic)
            else:
                action = [self.env.action_space.sample() for i in range(self.env.unwrapped.num_envs)]
            self.update_agreement(obs, action)
            obs, rewards, dones, info = self.env.step(action)
            if self.render:
                self.env.render(mode='human')
            self.update_values(obs, rewards, dones, info, action)

            # Reset env after done. DummyVecEnv and SubprocVecEnvs have an integrated reset mechanism.
            if not self.vec_env:
                if dones:
                    self.env.reset()

        # logging.info("Performed {} episodes with an avg length of {} and {} avg reward".format(episode_counter, step_counter / episode_counter, total_reward / episode_counter))
        if self.close_env:
            self.env.close()

    def update_values(self, obs, reward, done, info, action):
        if self.vec_env:
            self.episode_counter += np.sum(done)
            self.step_counter += len(obs)
            self.total_reward += sum(reward)
        else:
            self.episode_counter += done
            self.step_counter += 1
            self.total_reward += reward


    def _update_agreement_kl(self, obs, action):
        actions = [self.agent.action_probability(obs)]
        for agent in self.other_agents:
            other_action = agent.action_probability(obs)
            actions.append(other_action)
        actions = np.asarray(actions)

        if self.vec_env:
            self._calculate_agreement_kl_vec(actions)
        else:
            self._calculate_agreement_kl(actions)

    def _update_agreement_count(self, obs, action):
        actions = [self.agent.predict(obs, deterministic=self.deterministic)[0]]
        for agent in self.other_agents:
            other_action, other_states = agent.predict(obs, deterministic=self.deterministic)
            actions.append(other_action)
        actions = np.asarray(actions)

        if self.vec_env:
            self._calculate_agreement_count_vec(actions)
        else:
            self._calculate_agreement_count(actions)

    def update_agreement(self, obs, action):
        if len(self.other_agents) == 0:
            return
        if self.agreement_type == "action":
            self._update_agreement_count(obs, action)
            self._update_agreement_kl(obs, action)
        else:
            self._update_agreement_kl(obs, action)

    def kl_divergence(self, p, q):
        p =  1.0*p / np.sum(p, keepdims=True)
        q = 1.0*q / np.sum(q, keepdims=True)
        p += np.finfo(float).eps
        q += np.finfo(float).eps
        return np.sum(p * np.log(p / q))

    def kl_divergence_1(self, p, q):
        return 0.5 * np.mean(np.square(np.log(p) - np.log(q)))

    def _calculate_agreement_count_vec(self, actions):
        for i, action in enumerate(actions):
            for j, a in enumerate(action):
                self.agreement[i] += (a == actions[:, j])

    def _calculate_agreement_count(self, actions):
        for i, action in enumerate(actions):
            self.agreement[i] += (actions == action)

    def _calculate_agreement_kl_vec(self, actions):
        for i, actions_alg_1 in enumerate(actions):
            for j, actions_alg_2 in enumerate(actions):
                for env_action_alg_1, env_action_alg_2 in zip(actions_alg_1, actions_alg_2):
                    self.kl_div[i][j] += self.kl_divergence(env_action_alg_1, env_action_alg_2)

    def _calculate_agreement_kl(self, actions):
        for i, actions_alg_1 in enumerate(actions):
            for j, actions_alg_2 in enumerate(actions):
                self.kl_div[i][j] += self.kl_divergence(actions_alg_1, actions_alg_2)

