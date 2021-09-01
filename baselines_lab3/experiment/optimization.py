import logging
import os
from collections import MutableMapping
from copy import deepcopy
from typing import Optional

import numpy as np
import optuna
from gym.utils.seeding import create_seed
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from baselines_lab3.env import create_environment
from baselines_lab3.experiment.samplers import Sampler
from baselines_lab3.model import create_model
from baselines_lab3.model.callbacks import TensorboardLogger
from baselines_lab3.utils import send_email


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        best_model_save_path: Optional[str] = None,
        log_path: Optional[str] = None,
    ):

        super(TrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            logging.info(f"Evaluated model with mean reward of {self.last_mean_reward}")
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class HyperparameterOptimizer:
    """
    Class for automated hyperparameter optimization with optuna.
    :param config: (dict) Lab config.
    :param log_dir: (str) Global log directory.
    :param mail: (str) Weather or not to send mail information about training progress.
    """

    def __init__(self, config, log_dir, mail=None):
        search_config = config["search"]
        self.config = config

        # Number of test episodes per evaluation
        self.n_test_episodes = search_config.get("n_test_episodes", 10)
        # Number of evaluations per trial
        self.n_evaluations = search_config.get("n_evaluations", 15)
        # Timesteps per trial
        self.n_timesteps = search_config.get("n_timesteps", 10000)
        self.evaluation_interval = self.n_timesteps // self.n_evaluations
        self.n_trials = search_config.get("n_trials", 10)
        self.n_jobs = search_config.get("n_jobs", 1)
        self.seed = config["meta"]["seed"]
        self.sampler_method = search_config.get("sampler", "random")
        self.pruner_method = search_config.get("pruner", "median")
        self.eval_method = search_config.get("eval_method", "normal")
        self.deterministic_evaluation = search_config.get("deterministic", False)
        self.train_env = None
        self.test_env = None
        self.log_dir = log_dir
        self.logger = TensorboardLogger(config=self.config)  # TODO: Set config
        self.verbose_mail = mail
        self.current_best = -np.inf

    def optimize(self):
        """
        Starts the optimization process. This function will return even if the program receives a keyboard interrupt.
        :return (optuna.study.Study) An optuna study object containing all information about each trial that was run.
        """
        sampler = self._make_sampler()
        pruner = self._make_pruner()
        logging.info("Starting optimization process.")
        logging.info(f"Sampler: {self.sampler_method} - Pruner: {self.pruner_method}")

        study_name = "hypersearch"
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{os.path.join(self.log_dir, 'search.db')}",
            load_if_exists=True,
            direction="maximize",
        )
        objective = self._create_objective_function()

        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            # Still save results after keyboard interrupt!
            pass

        return study

    def _make_pruner(self):
        if isinstance(self.pruner_method, str):
            if self.pruner_method == "halving":
                pruner = SuccessiveHalvingPruner(
                    min_resource=self.n_timesteps // 6,
                    reduction_factor=4,
                    min_early_stopping_rate=0,
                )
            elif self.pruner_method == "median":
                pruner = MedianPruner(
                    n_startup_trials=5, n_warmup_steps=self.n_timesteps // 6
                )
            elif self.pruner_method == "none":
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError(f"Unknown pruner: {self.pruner_method}")
        elif isinstance(self.pruner_method, dict):
            method_copy = deepcopy(self.pruner_method)
            method = method_copy.pop("method")
            if method == "halving":
                pruner = SuccessiveHalvingPruner(**method_copy)
            elif method == "median":
                pruner = MedianPruner(**method_copy)
            elif method == "none":
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError(f"Unknown pruner: {self.pruner_method}")
        else:
            raise ValueError("Wrong type for pruner settings!")
        return pruner

    def _make_sampler(self):
        if self.sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif self.sampler_method == "tpe":
            sampler = TPESampler(n_startup_trials=5, seed=self.seed)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_method}")
        return sampler

    def _get_envs(self, config):
        if self.train_env:
            # Create new environments if normalization layer is learned.
            if config["env"].get("normalize", None):
                if not config["env"]["normalize"].get("precompute", False):
                    self._make_envs(config)
            # Create new environments if num_envs changed.
            elif isinstance(self.train_env, VecEnv):
                if self.train_env.unwrapped.num_envs != config["env"].get("n_envs", 1):
                    self._make_envs(config)
            else:
                self.train_env.reset()
        else:
            self._make_envs(config)

        return self.train_env, self.test_env

    def _make_envs(self, config):
        if self.train_env:
            self.train_env.close()
            del self.train_env

        self.train_env = create_environment(
            config, config["meta"]["seed"], log_dir=self.log_dir,
        )

        test_config = deepcopy(config)
        test_env_config = test_config["env"]
        if test_env_config["n_envs"] > 32:
            test_env_config["n_envs"] = 32
        if test_env_config.get("normalize", False):
            test_env_config["normalize"]["norm_reward"] = False
        if self.test_env:
            self.test_env.close()
            del self.test_env

        self.test_env = create_environment(test_config, create_seed())

    def _create_objective_function(self):
        sampler = Sampler.create_sampler(self.config)

        def objective(trial):
            trial_config = sampler.sample(trial)
            trial_config["algorithm"]["verbose"] = 0
            alg_sample, env_sample = sampler.last_sample
            logging.info(
                f"Sampled new configuration: algorithm: {alg_sample} env: {env_sample}"
            )

            train_env, test_env = self._get_envs(trial_config)
            model = create_model(
                trial_config["algorithm"], train_env, trial_config["meta"]["seed"]
            )
            self.logger.config = trial_config
            self.logger.reset()

            evaluation_callback = TrialEvalCallback(
                test_env,
                trial,
                n_eval_episodes=self.n_test_episodes,
                eval_freq=self.evaluation_interval,
                deterministic=self.deterministic_evaluation,
            )

            try:
                logging.debug("Training model...")
                model.learn(
                    trial_config["search"]["n_timesteps"],
                    callback=[evaluation_callback, self.logger],
                )
            except Exception as ex:
                # Random hyperparams may be invalid
                logging.warning(f"Something went wrong - stopping trial. {ex}")
                del model
                raise optuna.exceptions.TrialPruned()
            del model

            is_pruned = evaluation_callback.is_pruned
            reward = evaluation_callback.last_mean_reward
            # Log params by flattening the configuration dict and replacing lists with strings.
            params = deepcopy(trial_config)
            params = flatten_dict(params)
            for k in params:
                if type(params[k]) not in [str, int, bool, float]:
                    params[k] = str(params[k])
            self.logger.write_hparams(params, {"reward": reward})

            if reward > self.current_best:
                self.current_best = reward
                if self.verbose_mail:
                    send_email(
                        self.verbose_mail,
                        "Hyperparametersearch new best mean reward {:.4f}".format(
                            self.current_best
                        ),
                        "Found new parameters with mean of {} and parameters {} {}".format(
                            self.current_best, alg_sample, env_sample
                        ),
                    )

            if is_pruned:
                logging.info("Pruned trial.")
                raise optuna.exceptions.TrialPruned()

            return reward

        return objective
