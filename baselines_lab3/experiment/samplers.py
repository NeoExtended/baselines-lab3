from abc import ABC, abstractmethod
from copy import deepcopy


class Sampler(ABC):
    """
    Base class for all samplers. Generates parameter sets from a specific sampler for hyperparameter optimization.
    :param config: (dict) Lab configuration.
    """
    def __init__(self, config, parameters=None):
        self.config = config
        self.alg_parameters = {}
        self.env_parameters = {}
        self.alg_sample = None
        self.env_sample = None

        if config.get('search', None):
            search_config = config['search']
            if search_config.get('algorithm', None):
                for sample_name in search_config['algorithm']:
                    if sample_name not in self.config['algorithm']:
                        self.alg_parameters[sample_name] = self._parse_config(search_config['algorithm'][sample_name])

            if search_config.get('env', None):
                for sample_name in search_config['env']:
                    if sample_name not in self.config['env']:
                        self.env_parameters[sample_name] = self._parse_config(search_config['env'][sample_name])

        for key in list(parameters.keys()):
            if key in self.config['algorithm']:
                del parameters[key]

        parameters.update(self.alg_parameters)
        self.alg_parameters.update(parameters)

    def __call__(self, *args, **kwargs):
        return self.sample(**kwargs)

    def sample(self, trial):
        """
        Samples a new parameter set and returns an updated lab config.
        :param trial: (optuna.trial.Trial) Optuna trial object containing the sampler that should be used to sample the
            parameters.
        """
        alg_sample = self.sample_parameters(trial, self.alg_parameters, 'alg_')
        env_sample = self.sample_parameters(trial, self.env_parameters, 'env_')
        return self.update_config(alg_sample, env_sample)

    def sample_parameters(self, trial, parameters, prefix=""):
        """
        Samples parameters for a trial, from a given parameter set.
        :param trial: (optuna.trial.Trial) Optuna trial to sample parameters for.
        :param parameters: (dict) The parameter set which should be used for sampling.
            Must be in form of key: (method, data)
        :param prefix: (str) A name prefix for all parameters.
        :return (dict) Returns a dict containing the sampled parameters
        """
        sample = {}
        for name, (method, data) in parameters.items():
            p_name = prefix + name
            if method == 'categorical':
                sample[name] = trial.suggest_categorical(p_name, data)
            elif method == 'loguniform':
                low, high = data
                sample[name] = trial.suggest_loguniform(p_name, low, high)
            elif method == 'int':
                low, high = data
                sample[name] = trial.suggest_int(p_name, low, high)
            elif method == 'uniform':
                low, high = data
                sample[name] = trial.suggest_uniform(p_name, low, high)
            elif method == 'discrete_uniform':
                low, high, q = data
                sample[name] = trial.suggest_discrete_uniform(p_name, low, high, q)
        return sample

    def update_config(self, alg_sample, env_sample):
        """
        Updates the config with a set of sampled parameters. Will not overwrite existing keys.
        :param alg_sample: (dict) Set of parameters to update the algorithm configuration.
        :param env_sample: (dict) Set of parameters to update the environment configuration.
        :return (dict) The updated lab config.
        """
        alg_sample, env_sample = self.transform_samples(alg_sample, env_sample)
        sampled_config = deepcopy(self.config)

        alg_sample.update(self.config['algorithm'])
        sampled_config['algorithm'].update(alg_sample)

        env_sample.update(self.config['env'])
        sampled_config['env'].update(env_sample)
        self.alg_sample, self.env_sample = alg_sample, env_sample
        return sampled_config

    @property
    def last_sample(self):
        return self.alg_sample, self.env_sample

    @abstractmethod
    def transform_samples(self, alg_sample, env_sample):
        """
        Method which is called before updating the dictionary. May be used to filter out invalid configurations.
        :param alg_sample: (dict) Set of parameters to update the algorithm configuration.
        :param env_sample: (dict) Set of parameters to update the environment configuration.
        :return (dict, dict) The updated or filtered alg_sample and env_sample
        """
        pass

    def _parse_config(self, parameter_config):
        method = parameter_config.get('method')
        if method == 'categorical':
            data = parameter_config.get('choices')
        elif method == 'loguniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')))
        elif method == 'int':
            data = (int(parameter_config.get('low')), int(parameter_config.get('high')))
        elif method == 'uniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')))
        elif method == 'discrete_uniform':
            data = (float(parameter_config.get('low')), float(parameter_config.get('high')), float(parameter_config.get('q')))
        else:
            raise NotImplementedError("Unknown sampling method {}".format(method))

        return method, data

    @staticmethod
    def create_sampler(config):
        """
        Creates a new sampler for the given lab configuration.
        :param config: (dict) Lab configuration.
        """
        alg_name = config['algorithm']['name']
        if alg_name == 'ppo2':
            return PPO2Sampler(config)
        elif alg_name == 'acktr':
            return ACKTRSampler(config)
        elif alg_name == 'acer':
            return ACERSampler(config)
        elif alg_name == 'dqn':
            return DQNSampler(config)
        else:
            raise NotImplementedError("There is currently no parameter sampler available for {}".format(alg_name))


class DQNSampler(Sampler):
    """
    Sampler for DQN Parameters
    """
    def __init__(self, config):
        parameters = {'gamma': ('categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
                      'prioritized_replay' : ('categorical', [True, False]),
                      'learning_rate': ('loguniform', (0.5e-5, 0.05)),
                      'buffer_size': ('categorical', [250000, 500000, 1000000, 1500000]),
                      'train_freq': ('categorical', [2, 4, 8, 16]),
                      'batch_size': ('categorical', [8, 16, 32, 64]),
                      'target_network_update_freq': ('categorical', [500, 1000, 2000, 4000, 8000, 16000]),
                      'exploration_fraction': ('categorical', [0.1, 0.2, 0.3, 0.4, 0.5]),
                      'exploration_final_eps': ('categorical', [0.01, 0.02, 0.04, 0.07, 0.1]),
                      #'prioritized_replay_alpha': ('uniform', (0.2, 0.8)),
                      #'prioritized_replay_beta0': ('uniform', (0.2, 0.8)),
                      'learning_starts': ('categorical', [1000, 2000, 4000, 8000, 16000])}
        super().__init__(config, parameters)

    def transform_samples(self, alg_sample, env_sample):
        return alg_sample, env_sample


class PPO2Sampler(Sampler):
    """
    Sampler for basic PPO2 parameters.
    """
    def __init__(self, config):
        parameters = {'batch_size': ('categorical', [32, 64, 128, 256]),
                      'n_steps': ('categorical', [16, 32, 64, 128, 256, 512, 1024, 2048]),
                      'gamma': ('categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
                      'learning_rate': ('loguniform', (0.5e-5, 0.2)),
                      'ent_coef': ('loguniform', (1e-8, 0.1)),
                      'cliprange': ('categorical', [0.05, 0.1, 0.2, 0.3, 0.4]),
                      'cliprange_vf': ('categorical', [-1, None]),
                      'noptepochs': ('categorical', [1, 5, 10, 20, 30, 50]),
                      'lam': ('categorical', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])}
        super().__init__(config, parameters)

    def transform_samples(self, alg_sample, env_sample):
        batch_size = alg_sample.pop('batch_size')
        if alg_sample['n_steps'] < batch_size:
            alg_sample['nminibatches'] = 1
        else:
            alg_sample['nminibatches'] = int(alg_sample['n_steps'] / batch_size)

        if alg_sample['n_steps'] < 100:
            if alg_sample['noptepochs'] > 20:
                alg_sample['noptepochs'] = 20

        return alg_sample, env_sample


class ACKTRSampler(Sampler):
    def __init__(self, config):
        parameters = {'gamma': ('categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
                      'n_steps': ('categorical', [16, 32, 64, 128]),
                      'lr_schedule': ('categorical', ['linear', 'constant', 'double_linear_con', 'middle_drop']),
                      'learning_rate': ('loguniform', (1e-5, 1.0)),
                      'ent_coef': ('loguniform', (1e-8, 0.1)),
                      'vf_coef': ('uniform', (0, 1)),
                      'gae_lambda': ('categorical', [0.8, 0.9, 0.95, 0.99, None]),
                      'async_eigen_decomp': ('categorical', [True, False])}
        super().__init__(config, parameters)

    def transform_samples(self, alg_sample, env_sample):
        return alg_sample, env_sample


class ACERSampler(Sampler):
    def __init__(self, config):
        parameters = {'gamma': ('categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
                      'n_steps': ('categorical', [16, 32, 64, 128, 256, 512, 1024, 2048]),
                      'lr_schedule': (
                      'categorical', ['linear', 'constant', 'double_linear_con', 'middle_drop', 'double_middle_drop']),
                      'learning_rate': ('loguniform', (1e-5, 0.2)),
                      'ent_coef': ('loguniform', (1e-8, 0.1)),
                      'q_coef': ('uniform', (0.01, 0.9)),
                      'buffer_size': ('categorical', [5000, 10000, 20000, 40000, 50000])}
        super().__init__(config, parameters)

    def transform_samples(self, alg_sample, env_sample):
        alg_sample['replay_start'] = int(alg_sample['buffer_size'] // 5)
        return alg_sample, env_sample
