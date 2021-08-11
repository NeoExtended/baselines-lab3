"""
General helper functions.
"""
import importlib
import os
import time
from datetime import datetime
from subprocess import Popen, PIPE
import logging
from typing import Type, Any, Optional

import gym
import numpy as np
from gym.wrappers import Monitor
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.preprocessing import (
    check_for_nested_spaces,
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import (
    VecEnvWrapper,
    VecEnv,
    DummyVecEnv,
    is_vecenv_wrapped,
    VecTransposeImage,
)

log_dir = None
TIMESTAMP_FORMAT = "%Y_%m_%d_%H%M%S"


def load_class_from_module(identifier: str) -> Optional[Type[Any]]:
    """
    Dynamically loads a class from a given module.

    :param identifier: A fully qualified class name (e.g. baselines_lab3.experiment.scheduler.Scheduler)
    :return: The imported type. May be none, if the import was not successful.
    """
    if identifier:
        wrapper_module = importlib.import_module(identifier.rsplit(".", 1)[0])
        return getattr(wrapper_module, identifier.split(".")[-1])
    else:
        logging.warning(f"Could not import class {identifier}")
        return None


def get_timestamp(pattern=TIMESTAMP_FORMAT):
    """
    Generates a string timestamp to use for logging.
    :param pattern: (str) Pattern for the timestamp following the python datetime format.
    :return: (str) The current timestamp formated according to pattern
    """
    time = datetime.now()
    return time.strftime(pattern)


def create_log_directory(root):
    """
    Creates a global log directory at a given place. The directory will be named with a current timestamp.
    :param root: (str) Parent directory for the log directory. Will be created if it does not exist.
    :return: (str) Location of the created log directory.
    """
    if not root:
        return None

    timestamp = get_timestamp()
    path = os.path.join(root, timestamp)

    global log_dir
    log_dir = os.path.abspath(path)
    os.makedirs(log_dir)
    return log_dir


def set_log_directory(log_directory):
    """
    Sets the global log directory.
    """
    global log_dir
    log_dir = log_directory


def get_log_directory():
    """
    Returns the current log directory. May be None if create_log_directory() has not been called before.
    """
    global log_dir
    return log_dir


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def unwrap_env(env, target_vec_wrapper, target_wrapper):
    """
    Unwraps the given environment until the target wrapper is found.
    Returns the first wrapper if target wrapper was not found.
    :param env: (gym.Wrapper or VecEnvWrapper) The wrapper to unwrap.
    :param target_wrapper: (gym.Wrapper) Class of the target wrapper in case of type(env)==gym.Wrapper.
    :param target_vec_wrapper: (VecEnvWrapper) Class of the target wrapper in case of type(env)==VecEnvWrapper
    """
    if "Vec" in type(env).__name__:
        return unwrap_vec_env(env, target_vec_wrapper)
    else:
        return unwrap_standard_env(env, target_wrapper)


def unwrap_standard_env(env, target_wrapper):
    """
    Unwraps the given environment until the target wrapper is found.
    Returns the first wrapper if target wrapper was not found.
    :param env: (gym.Wrapper) The wrapper to unwrap.
    :param target_wrapper: (gym.Wrapper) Class of the target wrapper.
    """
    while not isinstance(env, target_wrapper) and isinstance(env.env, gym.Wrapper):
        env = env.env
    return env


def unwrap_vec_env(env, target_wrapper):
    """
    Unwraps the given environment until the target wrapper is found.
    Returns the first wrapper if target wrapper was not found.
    :param env: (VecEnvWrapper) The wrapper to unwrap.
    :param target_wrapper: (VecEnvWrapper) Class of the target wrapper.
    """
    while not isinstance(env, target_wrapper) and isinstance(env.venv, VecEnvWrapper):
        env = env.venv
    return env


def send_email(receiver, subject, content):
    run_command_line_command(
        'echo "{}" | mail -s "[baselines-lab] {}" {}'.format(content, subject, receiver)
    )


def run_command_line_command(command):
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    response = process.communicate()
    return response


def wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
    """ "
    This is a copy of stable_baselines3.common.base_class.BaseAlgorithm._wrap_env
    Wrap environment with the appropriate wrappers if needed.
    For instance, to have a vectorized environment
    or to re-order the image channels.

    :param env:
    :param verbose:
    :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
    :return: The wrapped environment.
    """
    if not isinstance(env, VecEnv):
        if not is_wrapped(env, Monitor) and monitor_wrapper:
            if verbose >= 1:
                print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])

    # Make sure that dict-spaces are not nested (not supported)
    check_for_nested_spaces(env.observation_space)

    if isinstance(env.observation_space, gym.spaces.Dict):
        for space in env.observation_space.spaces.values():
            if isinstance(space, gym.spaces.Dict):
                raise ValueError(
                    "Nested observation spaces are not supported (Dict spaces inside Dict space)."
                )

    if not is_vecenv_wrapped(env, VecTransposeImage):
        wrap_with_vectranspose = False
        if isinstance(env.observation_space, gym.spaces.Dict):
            # If even one of the keys is a image-space in need of transpose, apply transpose
            # If the image spaces are not consistent (for instance one is channel first,
            # the other channel last), VecTransposeImage will throw an error
            for space in env.observation_space.spaces.values():
                wrap_with_vectranspose = wrap_with_vectranspose or (
                    is_image_space(space) and not is_image_space_channels_first(space)
                )
        else:
            wrap_with_vectranspose = is_image_space(
                env.observation_space
            ) and not is_image_space_channels_first(env.observation_space)

        if wrap_with_vectranspose:
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

    return env
