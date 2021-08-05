import copy
import logging

import gym
import tensorflow as tf

# from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, DDPG, TRPO, SAC, TD3
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN, HER, SAC, TD3
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common.base_class import BaseRLModel

from baselines_lab3.model.schedules import get_schedule
from baselines_lab3.utils import util

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
#    'ddpg': DDPG, mpi dependency
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
#    'trpo': TRPO, mpi dependency
    'td3': TD3
}

def create_model(config: dict, env: gym.Env, seed: int) -> BaseRLModel:
    """
    Creates a stable-baselines model according to the given lab configuration.
    :param config: (dict) The current lab model configuration (from config['algorithm']).
    :param env: (gym.Env) The environment to learn from.
    :param seed: The current seed for the model prngs.
    :return: (BaseRLModel) A model which can be used to learn in the given environment.
    """
    config = copy.deepcopy(config)
    name =  config.pop('name')
    tlog = config.pop('tensorboard_log', None)
    verbose = config.pop('verbose', 0)
    policy_config = config.pop('policy')

    tlog_location = _get_tensorflow_log_location(tlog)

    # Create lr schedules if supported
    if name in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', "cliprange", "cliprange_vf"]:
            if key in config and isinstance(config[key], dict):
                config[key] = get_schedule(config[key].pop("type"), **config[key])

    if 'trained_agent' in config: # Continue training
        logging.info("Loading pretrained model from {}.".format(config['trained_agent']))

        return ALGOS[name].load(
            config['trained_agent'],
            seed=seed,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            **config)

    else:
        logging.info("Creating new model for {}.".format(name))
        policy_name = policy_config.pop('name')

        return ALGOS[name](
            seed=seed,
            policy=policy_name,
            policy_kwargs=policy_config,
            env=env,
            tensorboard_log=tlog_location,
            verbose=verbose,
            **config)


def _get_tensorflow_log_location(tlog):
    """
    Returns the tensorflow log directory.
    :param tlog: The tensorboard-log parameter from config['algorithm']['tensorboard_log']
    """
    if tlog:
        if isinstance(tlog, bool):
            return util.get_log_directory()
        else:
            return tlog.get('path', util.get_log_directory())
    else:
        return None
