import collections
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Tuple, Union, List, Any, Dict

from baselines_lab3.utils.util import update_dict, flatten_dict, unflatten_dict
import optuna

NAMESPACES = ["algorithm", "env"]


class Sampler(ABC):
    """
    Base class for all samplers. Generates parameter sets from a specific sampler for hyperparameter optimization.
    :param config: (dict) Lab configuration.
    """

    def __init__(self, config, parameters=None):
        self.config = config
        self.parameters = {}
        self.last_sample = None

        # Add default parameters
        for key in parameters:
            self.add_parameters(key, parameters[key])

        # Add parameters from config file
        if config.get("search", False):
            search_config = config["search"]
            for namespace in NAMESPACES:
                if search_config.get(namespace, False):
                    self._extract_search_parameters(
                        namespace, "", search_config[namespace]
                    )

            # Remove parameters which are explicitly excluded
            if search_config.get("exclude", False):
                exclude = search_config["exclude"]
                for namespace in NAMESPACES:
                    self._remove_excluded_parameters(namespace, exclude)

    def _remove_excluded_parameters(self, namespace: str, exclude: Dict):
        if exclude.get(namespace, False):
            for key in exclude[namespace]:
                self.remove_parameter(namespace, key)

    def _extract_search_parameters(
        self,
        namespace: str,
        parent: str,
        search_config: collections.abc.MutableMapping,
        separator: str = ".",
    ):
        for key, v in search_config.items():
            path = parent + separator + key if parent else key

            if isinstance(v, collections.MutableMapping):
                self._extract_search_parameters(
                    parent=path,
                    search_config=v,
                    separator=separator,
                    namespace=namespace,
                )

                if v.get("sample", False):
                    self.add_parameter(namespace, path, self._parse_config(v))

    def add_parameters(self, namespace: str, parameters: Dict):
        for k, v in parameters.items():
            self.add_parameter(namespace, k, v)

    def add_parameter(self, namespace: str, key: str, value: Any):
        self.parameters[namespace + "." + key] = value

    def get_parameter(self, namespace: str, key: str) -> Any:
        return self.parameters[namespace + "." + key]

    def remove_parameter(self, namespace: str, key: str):
        del self.parameters[namespace + "." + key]

    def __call__(self, *args, **kwargs):
        return self.sample(**kwargs)

    def sample(self, trial):
        """
        Samples a new parameter set and returns an updated lab config.
        :param trial: (optuna.trial.Trial) Optuna trial object containing the sampler that should be used to sample the
            parameters.
        """
        self.last_sample = self.sample_parameters(trial)
        sampled_config = deepcopy(self.config)
        sample = update_dict(sampled_config, unflatten_dict(self.last_sample))
        transformed = self.transform_sample(sample)

        return transformed

    def sample_parameters(self, trial: optuna.Trial):
        """
        Samples parameters for a trial, from a given parameter set.
        :param trial: (optuna.trial.Trial) Optuna trial to sample parameters for.
        :param parameters: (dict) The parameter set which should be used for sampling.
            Must be in form of key: (method, data)
        :param prefix: (str) A name prefix for all parameters.
        :return (dict) Returns a dict containing the sampled parameters
        """
        sample = {}
        for name, (method, data) in self.parameters.items():
            if method == "categorical":
                sample[name] = trial.suggest_categorical(name, data)
            elif method == "loguniform":
                low, high = data
                sample[name] = trial.suggest_loguniform(name, low, high)
            elif method == "int":
                low, high = data
                sample[name] = trial.suggest_int(name, low, high)
            elif method == "uniform":
                low, high = data
                sample[name] = trial.suggest_uniform(name, low, high)
            elif method == "discrete_uniform":
                low, high, q = data
                sample[name] = trial.suggest_discrete_uniform(name, low, high, q)
        return sample

    @abstractmethod
    def transform_sample(self, sample: Dict):
        """
        Method which is called after updating the dictionary. May be used to filter out invalid configurations.
        :return (dict) The updated or filtered alg_sample and env_sample
        """
        pass

    def _parse_config(
        self, parameter_config
    ) -> Tuple[str, Union[List[Any], Tuple[int, int], Tuple[int, int, int]]]:
        method = parameter_config.get("method")
        if method == "categorical":
            data = parameter_config.get("choices")
        elif method == "loguniform":
            data = (
                float(parameter_config.get("low")),
                float(parameter_config.get("high")),
            )
        elif method == "int":
            data = (int(parameter_config.get("low")), int(parameter_config.get("high")))
        elif method == "uniform":
            data = (
                float(parameter_config.get("low")),
                float(parameter_config.get("high")),
            )
        elif method == "discrete_uniform":
            data = (
                float(parameter_config.get("low")),
                float(parameter_config.get("high")),
                float(parameter_config.get("q")),
            )
        else:
            raise NotImplementedError("Unknown sampling method {}".format(method))

        return method, data

    @staticmethod
    def create_sampler(config):
        """
        Creates a new sampler for the given lab configuration.
        :param config: (dict) Lab configuration.
        """
        alg_name = config["algorithm"]["name"]
        if alg_name == "ppo":
            return PPOSampler(config)
        elif alg_name == "dqn":
            return DQNSampler(config)
        else:
            raise NotImplementedError(
                "There is currently no parameter sampler available for {}".format(
                    alg_name
                )
            )


class DQNSampler(Sampler):
    """
    Sampler for DQN Parameters
    """

    def __init__(self, config):
        alg_parameters = {
            "gamma": ("categorical", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
            "prioritized_replay": ("categorical", [True, False]),
            "learning_rate": ("loguniform", (0.5e-5, 0.05)),
            "buffer_size": ("categorical", [250000, 500000, 1000000, 1500000]),
            "train_freq": ("categorical", [2, 4, 8, 16]),
            "batch_size": ("categorical", [8, 16, 32, 64]),
            "target_network_update_freq": (
                "categorical",
                [500, 1000, 2000, 4000, 8000, 16000],
            ),
            "exploration_fraction": ("categorical", [0.1, 0.2, 0.3, 0.4, 0.5]),
            "exploration_final_eps": ("categorical", [0.01, 0.02, 0.04, 0.07, 0.1]),
            #'prioritized_replay_alpha': ('uniform', (0.2, 0.8)),
            #'prioritized_replay_beta0': ('uniform', (0.2, 0.8)),
            "learning_starts": ("categorical", [1000, 2000, 4000, 8000, 16000]),
        }

        parameters = {"algorithm": alg_parameters}

        super().__init__(config, parameters)

    def transform_sample(self, sample):
        return sample


class PPOSampler(Sampler):
    """
    Sampler for basic PPO parameters.
    """

    def __init__(self, config):

        self.net_arch = {
            "small": [64, 64],
            "medium": [128, 128],
            "diverging": [128, dict(pi=[128], vf=[128])],
        }

        alg_params = {
            "batch_size": ("categorical", [32, 64, 128, 256, 512, 1024, 2048]),
            "n_steps": ("categorical", [16, 32, 64, 128, 256, 512, 1024, 2048]),
            "gamma": ("categorical", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]),
            "learning_rate": ("loguniform", (0.5e-5, 0.5)),
            "ent_coef": ("loguniform", (1e-8, 0.1)),
            "clip_range": ("categorical", [0.05, 0.1, 0.2, 0.3, 0.4]),
            #            "cliprange_vf": ("categorical", [-1, None]),
            "n_epochs": ("categorical", [1, 5, 10, 20, 30]),
            "gae_lambda": ("categorical", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]),
            "max_grad_norm": ("categorical", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]),
            "policy.ortho_init": ("categorical", [False, True]),
            "vf_coef": ("uniform", (0.0, 1.0)),
            "policy.activation_fn": (
                "categorical",
                [
                    "torch.nn.Tanh",
                    "torch.nn.ReLU",
                    "torch.nn.LeakyReLU",
                    "torch.nn.SELU",
                ],
            ),
            "network": ("categorical", ["small", "medium", "diverging"]),
        }

        env_params = {"n_envs": ("categorical", [4, 8, 16, 32, 64])}

        parameters = {"algorithm": alg_params, "env": env_params}

        super().__init__(config, parameters)

    def transform_sample(self, sample):
        if (
            sample["algorithm"]["n_steps"] * sample["env"]["n_envs"]
            < sample["algorithm"]["batch_size"]
        ):
            sample["algorithm"]["batch_size"] = (
                sample["algorithm"]["n_steps"] * sample["env"]["n_envs"]
            )

        network_type = sample["algorithm"].pop("network")
        sample["algorithm"]["policy"]["net_arch"] = self.net_arch[network_type]

        return sample
