import logging
import os
from copy import deepcopy

import numpy as np
import optuna
from optuna.pruners import SuccessiveHalvingPruner, MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv

from baselines_lab3.env import create_environment
from baselines_lab3.env.evaluation import Evaluator
from baselines_lab3.experiment.samplers import Sampler
from baselines_lab3.model import create_model
from baselines_lab3.model.callbacks import TensorboardLogger
from baselines_lab3.utils import send_email


class EvaluationCallback(BaseCallback):
    """
    Callback for model evaluation and early stopping.
    """
    def __init__(self, evaluator, evaluation_interval, trial, verbose=0):
        super(EvaluationCallback, self).__init__(verbose)
        self.evaluator = evaluator
        self.evaluation_interval = evaluation_interval
        self.trial = trial

        self.pruned = False
        self.last_mean_test_reward = -np.inf
        self.last_time_evaluated = 0
        self.eval_idx = 0
        self.best_test_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_evaluated) < self.evaluation_interval:
            return True

        self.last_time_evaluated = self.num_timesteps
        logging.debug("Evaluating model at {} timesteps".format(self.num_timesteps))

        mean_reward, mean_steps = self.evaluator.evaluate(self.model)
        logging.info("Evaluated model at {} timesteps. Reached a mean reward of {}".format(self.num_timesteps, mean_reward))
        self.last_mean_test_reward = mean_reward
        self.eval_idx += 1

        if mean_reward > self.best_test_mean_reward:
            self.best_test_mean_reward = mean_reward

        # report best or report current ?
        # report num_timesteps or elasped time ?
        self.trial.report(-1 * mean_reward, self.num_timesteps)
        # Prune trial if need
        if self.trial.should_prune(self.num_timesteps):
            logging.debug("Pruning - aborting training...")
            self.pruned = True
            return False

        return True

    def is_pruned(self) -> bool:
        return self.pruned

    def best_mean_reward(self) -> float:
        return self.best_test_mean_reward

    def cost(self) -> float:
        return -1 * self.best_test_mean_reward


class HyperparameterOptimizer:
    """
    Class for automated hyperparameter optimization with optuna.
    :param config: (dict) Lab config.
    :param log_dir: (str) Global log directory.
    :param mail: (str) Weather or not to send mail information about training progress.
    """
    def __init__(self, config, log_dir, mail=None):
        search_config = config['search']
        self.config = config

        # Number of test episodes per evaluation
        self.n_test_episodes = search_config.get('n_test_episodes', 10)
        # Number of evaluations per trial
        self.n_evaluations = search_config.get('n_evaluations', 15)
        # Timesteps per trial
        self.n_timesteps = search_config.get('n_timesteps', 10000)
        self.evaluation_interval = int(self.n_timesteps / self.n_evaluations)
        self.n_trials = search_config.get('n_trials', 10)
        self.n_jobs = search_config.get('n_jobs', 1)
        self.seed = config['meta']['seed']
        self.sampler_method = search_config.get('sampler', 'random')
        self.pruner_method = search_config.get('pruner', 'median')
        self.eval_method = search_config.get('eval_method', 'normal')
        self.deterministic_evaluation = search_config.get('deterministic', False)
        self.train_env = None
        self.evaluator = None
        self.log_dir = log_dir
        self.logger = TensorboardLogger(config=self.config)  # TODO: Set config
        self.integrated_evaluation = True if self.eval_method == "fast" else False
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
        logging.info("Sampler: {} - Pruner: {}".format(self.sampler_method, self.pruner_method))

        study_name = "hypersearch"
        study = optuna.create_study(study_name=study_name,
                                    sampler=sampler,
                                    pruner=pruner,
                                    storage='sqlite:///{}'.format(os.path.join(self.log_dir, "search.db")),
                                    load_if_exists=True)
        objective = self._create_objective_function()

        try:
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            # Still save results after keyboard interrupt!
            pass

        return study

    def _make_pruner(self):
        if isinstance(self.pruner_method, str):
            if self.pruner_method == 'halving':
                pruner = SuccessiveHalvingPruner(min_resource=self.n_timesteps // 6, reduction_factor=4,  min_early_stopping_rate=0)
            elif self.pruner_method == 'median':
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=self.n_timesteps // 6)
            elif self.pruner_method == 'none':
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError('Unknown pruner: {}'.format(self.pruner_method))
        elif isinstance(self.pruner_method, dict):
            method_copy = deepcopy(self.pruner_method)
            method = method_copy.pop('method')
            if method == 'halving':
                pruner = SuccessiveHalvingPruner(**method_copy)
            elif method == 'median':
                pruner = MedianPruner(**method_copy)
            elif method == 'none':
                # Do not prune
                pruner = NopPruner()
            else:
                raise ValueError('Unknown pruner: {}'.format(self.pruner_method))
        else:
            raise ValueError("Wrong type for pruner settings!")
        return pruner

    def _make_sampler(self):
        if self.sampler_method == 'random':
            sampler = RandomSampler(seed=self.seed)
        elif self.sampler_method == 'tpe':
            sampler = TPESampler(n_startup_trials=5, seed=self.seed)
        else:
            raise ValueError('Unknown sampler: {}'.format(self.sampler_method))
        return sampler

    def _get_train_env(self, config):
        if self.train_env:
            # Create new environments if normalization layer is learned.
            if config['env'].get('normalize', None):
                if not config['env']['normalize'].get('precompute', False):
                    self.train_env.close()
                    self._make_train_env(config)
            # Create new environments if num_envs changed.
            if isinstance(self.train_env, VecEnv):
                if self.train_env.unwrapped.num_envs != config['env'].get('n_envs', 1):
                    self.train_env.close()
                    self._make_train_env(config)
        else:
            self._make_train_env(config)

        return self.train_env

    def _make_train_env(self, config):
        self.train_env = create_environment(config,
                                            config['meta']['seed'],
                                            evaluation=self.integrated_evaluation,
                                            log_dir=self.log_dir)

        self.evaluator = Evaluator(config,
                                   n_eval_episodes=self.n_test_episodes,
                                   deterministic=self.deterministic_evaluation,
                                   eval_method=self.eval_method,
                                   env=self.train_env)

    def _create_objective_function(self):
        sampler = Sampler.create_sampler(self.config)

        def objective(trial):
            trial_config = sampler.sample(trial)
            trial_config['algorithm']['verbose'] = 0
            alg_sample, env_sample = sampler.last_sample
            logging.info("Sampled new configuration: algorithm: {} env: {}".format(alg_sample, env_sample))

            train_env = self._get_train_env(trial_config)
            model = create_model(trial_config['algorithm'], train_env, trial_config['meta']['seed'])
            self.logger.config = trial_config
            self.logger.reset()

            evaluation_callback = EvaluationCallback(self.evaluator, self.evaluation_interval, trial)
            try:
                logging.debug("Training model...")
                model.learn(trial_config['search']['n_timesteps'],
                            callback=[evaluation_callback, self.logger])
            except Exception as ex:
                # Random hyperparams may be invalid
                logging.warning("Something went wrong - stopping trial. {}".format(ex))
                del model
                raise optuna.exceptions.TrialPruned()
            del model

            if evaluation_callback.best_mean_reward() > self.current_best:
                self.current_best = evaluation_callback.best_mean_reward()
                if self.verbose_mail:
                    send_email(self.verbose_mail,
                               "Hyperparametersearch new best mean reward {:.4f}".format(self.current_best),
                               "Found new parameters with mean of {} and parameters {} {}".format(self.current_best, alg_sample, env_sample))

            if evaluation_callback.is_pruned():
                logging.info("Pruned trial.")
                raise optuna.exceptions.TrialPruned()

            return evaluation_callback.cost()
        return objective
