from copy import deepcopy

from gym.utils.seeding import create_seed
from stable_baselines.common.vec_env import VecNormalize

from baselines_lab3.env import create_environment
from baselines_lab3.env.wrappers import EvaluationWrapper, VecEvaluationWrapper
from baselines_lab3.utils import unwrap_env, unwrap_vec_env


class Evaluator:
    """
    Class for easy model evaluation. Supports multiple evaluation methods for speed/accuracy tradeoff.
    Evaluates average reward and number of steps.

    :param config: (dict) Lab config used to create the evaluation environment for normal and slow evaluation mode.
    :param n_eval_episodes: (int) Number of episodes for evaluation.
    :param deterministic: (bool) Weather model actions should be deterministic or stochastic.
    :param render: (bool) Weather or not to render the environment during evaluation.
    :param eval_method: (str) One of the available evaluation types ("fast", "normal", "slow").
        Slow will only use a single env and will be the most accurate.
        Normal uses VecEnvs and fast requires env to be set and wrapped in a Evaluation Wrapper.
    :param env: (gym.Env or VecEnv) Environment used in case of eval_mode=="fast". Must be wrapped in an evaluation wrapper.
    :param seed: (int) Seed for the evaluation environment. If None a random seed will be generated.
    """
    def __init__(self, config=None, n_eval_episodes=32, deterministic=True, render=False, eval_method="normal",
                 env=None, seed=None):
        self.eval_method = eval_method
        self.config = config
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render

        if eval_method in ["normal", "slow"]:
            assert config, "You must provide an environment configuration, if the eval_method is not fast!"
            test_config = deepcopy(config)
            test_env_config = test_config['env']
            if eval_method == "slow":
                test_env_config['n_envs'] = 1

            if not test_env_config.get('n_envs', None) and not eval_method == "slow":
                test_env_config['n_envs'] = 8

            if not seed:
                seed = create_seed()
            if test_env_config['n_envs'] > 32:
                test_env_config['n_envs'] = 32
            test_env_config['curiosity'] = False # TODO: Sync train and test curiosity wrappers and reenable

            # Disable dynamic episode length on evaluation to get comparable test results independent of rewards.
            if test_env_config.get("reward_kwargs", None):
                if test_env_config["reward_kwargs"].get("dynamic_episode_length", False):
                    test_env_config["reward_kwargs"]["dynamic_episode_length"] = False

            self.test_env = create_environment(test_config,
                                               seed,
                                               evaluation=True)
            self.eval_wrapper = unwrap_env(self.test_env, VecEvaluationWrapper, EvaluationWrapper)
        elif eval_method == "fast":
            assert env, "You must provide an environment with an EvaluationWrapper if the eval_method is fast!"
            self.test_env = None
            self.eval_wrapper = unwrap_env(env, VecEvaluationWrapper, EvaluationWrapper)
        else:
            raise AttributeError("Unknown eval method '{}'".format(eval_method))

    def evaluate(self, model):
        """
        Evaluates the given model on the evaluation environment.
        """
        if self.eval_method == "fast":
            return self._evaluate_fast()
        else:
            return self._evaluate_normal(model)

    def close(self):
        if self.test_env is not None:
            self.test_env.close()

    def _evaluate_fast(self):
        return self.eval_wrapper.aggregator.reward_rms, self.eval_wrapper.aggregator.step_rms

    def _evaluate_normal(self, model):
        self.eval_wrapper.reset_statistics()

        if self.config.get('normalize', None): # Update normalization running means if necessary
            norm = unwrap_vec_env(self.test_env, VecNormalize)
            model_norm = unwrap_vec_env(model.env, VecNormalize)
            norm.obs_rms = model_norm.obs_rms
            norm.ret_rms = model_norm.ret_rms
            norm.training = False

        from baselines_lab3.experiment import Runner
        runner = Runner(self.test_env, model, render=self.render, deterministic=self.deterministic, close_env=False)
        runner.run(self.n_eval_episodes)
        return self.eval_wrapper.aggregator.mean_reward, self.eval_wrapper.aggregator.mean_steps
