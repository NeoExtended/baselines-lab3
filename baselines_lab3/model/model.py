import copy
import logging
from typing import Any, Dict

import gym

from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from baselines_lab3.model.schedules import get_schedule
from baselines_lab3.utils import util


ALGOS = {"a2c": A2C, "dqn": DQN, "her": HER, "sac": SAC, "ppo": PPO, "td3": TD3}


def create_model(config: Dict[str, Any], env: gym.Env, seed: int) -> BaseAlgorithm:
    """
    Creates a stable-baselines model according to the given lab configuration.
    :param config: (dict) The current lab model configuration (from config['algorithm']).
    :param env: (gym.Env) The environment to learn from.
    :param seed: The current seed for the model prngs.
    :return: (BaseRLModel) A model which can be used to learn in the given environment.
    """
    config = copy.deepcopy(config)
    name = config.pop("name")
    tlog = config.pop("tensorboard_log", None)
    verbose = config.pop("verbose", 0)
    policy_config = config.pop("policy")

    tlog_location = _get_tensorflow_log_location(tlog)

    # Create lr schedules if supported
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key in config and isinstance(config[key], dict):
            config[key] = get_schedule(config[key].pop("type"), **config[key])

    if "trained_agent" in config:  # Continue training
        logging.info(
            "Loading pretrained model from {}.".format(config["trained_agent"])
        )

        model = ALGOS[name].load(
            config["trained_agent"],
            seed=seed,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            **config
        )

    else:
        logging.info("Creating new model for {}.".format(name))
        policy_name = policy_config.pop("name")

        model = ALGOS[name](
            seed=seed,
            policy=policy_name,
            policy_kwargs=policy_config,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            **config
        )

    logging.info("Network Architecture:")
    logging.info(model.policy)
    return model


def _get_tensorflow_log_location(tlog):
    """
    Returns the tensorflow log directory.
    :param tlog: The tensorboard-log parameter from config['algorithm']['tensorboard_log']
    """
    if tlog:
        if isinstance(tlog, bool):
            return util.get_log_directory()
        else:
            return tlog.get("path", util.get_log_directory())
    else:
        return None
