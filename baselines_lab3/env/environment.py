import copy
import logging
import os
from typing import Any, Dict, Optional, List, Type, Tuple, Union

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    SubprocVecEnv,
    VecNormalize,
    DummyVecEnv,
    VecTransposeImage,
    VecEnvWrapper,
)

from baselines_lab3 import utils
from baselines_lab3.env.wrappers import (
    EvaluationWrapper,
    VecEvaluationWrapper,
    VecImageRecorder,
    VecScaledFloatFrame,
    VecStepSave,
    VecAttributeLogger,
)
from baselines_lab3.env.wrappers.evaluation_wrappers import ParticleInformationWrapper


def make_env(
    env_id: str,
    env_kwargs: Dict[str, Any],
    rank: int = 0,
    seed: int = 0,
    log_dir: str = None,
    wrappers: Optional[List[Tuple[Type[gym.Wrapper], Dict[str, Any]]]] = None,
    evaluation: bool = False,
    monitor: bool = False,
):
    """
    Helper function to multiprocess training and log the progress.
    :param env_kwargs: Additional arguments for the environment
    :param monitor: Weather or not to wrap the environment with a monitor wrapper.
    :param env_id: (str) Name of the environment.
    :param rank: (int) Pseudo-RNG seed shift for the environment.
    :param seed: (int) Pseudo-RNG seed for the environment.
    :param log_dir: (str) Log directory for the environment.
    :param wrappers: (list) Subclasses of gym.Wrapper, provided as a list of tuples (class, class_kwargs).
        Will be used to wrap the env with.
    :return (function) a function to create environments, e.g. for use in SubprocVecEnv or DummyVecEnv
    """

    def _init():
        # set_global_seeds(seed + rank)
        env = gym.make(env_id, **env_kwargs)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        if log_dir and evaluation:
            env = ParticleInformationWrapper(env, path=os.path.join(log_dir, str(rank)))

        if wrappers:
            for wrapper, args in wrappers:
                env = wrapper(env=env, **args)

        if monitor:
            env = Monitor(
                env, filename=None, allow_early_resets=True
            )  # filename=os.path.join(log_dir, str(rank))
        return env

    return _init


def create_environment(
    config: Dict[str, Any],
    seed: int,
    log_dir: str = None,
    video_path: str = None,
    evaluation: bool = False,
    monitor: bool = True,
):
    """
    Creates a new environment according to the parameters from the given lab config dictionary.
    :param monitor: Weather or not to wrap the environment with a Monitor wrapper.
    :param config: (dict) Lab config.
    :param seed: (int) Pseudo-RNG seed for the environment. Vectorized environments will use linear increments
        from this seed.
    :param log_dir: (str) Path to the log directory.
    :param video_path: (str) If a video path is given the environment will create a gif of the env observation space
        before the normalization layer (if present).
    :param evaluation: (bool) Weather or not to create an evaluation wrapper for the environment.
    :return: (gym.Env) New gym environment created according to the given configuration.
    """
    alg_config = copy.deepcopy(config["algorithm"])
    record_images = config["meta"].get("record_images", False)
    config = copy.deepcopy(config["env"])

    config.pop("evaluation", None)  # Always ignore evaluation specific configuration.

    env_id = config.pop("name")
    n_envs = config.pop("n_envs", 1)
    normalize = config.pop("normalize", None)
    frame_stack = config.pop("frame_stack", None)
    multiprocessing = config.pop("multiprocessing", True)
    log_env_attributes = config.pop("log_attributes", None)

    scale = config.pop("scale", None)
    logging.info(
        "Creating environment with id {} and {} instances.".format(env_id, n_envs)
    )

    # Get tuples with (wrapper_class, wrapper_kwargs)
    wrappers_config = config.pop("wrappers", [])
    wrappers = load_wrappers(wrappers_config)
    vec_env_config = config.pop("vec_env_wrappers", [])
    vec_env_wrappers = load_wrappers(vec_env_config)

    return _create_vectorized_env(
        env_id,
        config,
        n_envs,
        multiprocessing,
        seed,
        log_dir,
        wrappers,
        vec_env_wrappers,
        normalize,
        frame_stack,
        video_path,
        evaluation,
        monitor,
        scale,
        record_images,
        alg_config["name"],
        log_env_attributes,
    )


def load_wrappers(wrappers_config):
    wrappers = []
    for wrapper in wrappers_config:
        if isinstance(wrapper, dict):
            wrapper_name = list(wrapper.keys())[0]
            wrappers.append(
                (utils.load_class_from_module(wrapper_name), wrapper[wrapper_name])
            )
        elif isinstance(wrapper, str):
            wrappers.append((utils.load_class_from_module(wrapper), {}))
        else:
            raise ValueError("Got invalid wrapper with value {}".format(str(wrapper)))
    return wrappers


def _create_vectorized_env(
    env_id: str,
    env_kwargs: Dict[str, Any],
    n_envs: int,
    multiprocessing: bool,
    seed: int,
    log_dir: str,
    wrappers: List[Tuple[Type[gym.Wrapper], Dict[str, Any]]],
    vec_env_wrappers: List[Tuple[Type[VecEnvWrapper], Dict[str, Any]]],
    normalize: Union[bool, Dict[str, Any]],
    frame_stack: Union[bool, Dict[str, Any]],
    video_path: str,
    evaluation: bool,
    monitor: bool,
    scale: Union[bool, Dict[str, Any]],
    buffer_step_data: bool,
    algorithm_name: str,
    log_env_attributes: Optional[List[Tuple[str, str]]] = None,
):
    env_creation_fns = [
        make_env(
            env_id,
            env_kwargs,
            i,
            seed,
            log_dir,
            wrappers,
            evaluation=evaluation,
            monitor=monitor,
        )
        for i in range(n_envs)
    ]
    if multiprocessing:
        env = SubprocVecEnv(env_creation_fns)
    else:
        env = DummyVecEnv(env_creation_fns)

    for wrapper, args in vec_env_wrappers:
        env = wrapper(venv=env, **args)

    if log_env_attributes:
        env = VecAttributeLogger(env, log_dir=log_dir, attributes=log_env_attributes)

    if video_path:
        env = VecImageRecorder(env, video_path, record_obs=True)

    if evaluation:
        env = VecEvaluationWrapper(env)

    # Add normalization wrapper for all algorithms except dqn here to save computations before frame stack
    if normalize and "dqn" not in algorithm_name:
        env = _add_normalization_wrapper(env, n_envs, normalize)

    if scale:
        if isinstance(scale, dict):
            env = VecScaledFloatFrame(env, **scale)
        else:
            env = VecScaledFloatFrame(env)

    if frame_stack:
        env = VecFrameStack(env, **frame_stack)

    # Add normalization wrapper here to include frame stack when training with dqn.
    if normalize and "dqn" in algorithm_name:
        env = _add_normalization_wrapper(env, n_envs, normalize)

    if buffer_step_data:
        env = VecStepSave(env)

    # Wrap if needed to re-order channels
    # (switch from channel last to channel first convention)
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)

    return env


def _add_normalization_wrapper(env, n_envs, normalize):
    if isinstance(normalize, bool):
        env = VecNormalize(env)
    elif isinstance(normalize, dict):
        if "trained_agent" in normalize:
            path = normalize.pop("trained_agent")
            logging.info(f"Loading pretrained normalization parameters from {path}.")
            env = VecNormalize.load(path, env)
            env.training = normalize.pop("training", True)
        elif normalize.pop("precompute", False):
            samples = normalize.pop("samples", 10000)
            env = _precompute_normalization(env, n_envs, samples, normalize)
        else:
            env = VecNormalize(env, **normalize)
    return env


def _create_standard_env(
    env_id,
    env_kwargs,
    seed,
    log_dir,
    wrappers,
    normalize,
    frame_stack,
    evaluation,
    scale,
):
    env_maker = make_env(
        env_id, env_kwargs, 0, seed, log_dir, wrappers, evaluation=evaluation
    )
    env = env_maker()

    if evaluation:
        env = EvaluationWrapper(env)
    if normalize:
        logging.warning("Normalization is not supported for DDPG/DQN methods.")
    if scale:
        logging.warning("Scaling is not supported for DDPG/DQN methods.")
    if frame_stack:
        logging.warning("Frame-Stacking is not supported for DDPG/DQN methods.")

    return env


def _precompute_normalization(env, num_envs, samples, config):
    env = VecNormalize(env, training=True, **config)

    logging.info("Precomputing normalization. This may take a while.")
    env.reset()
    log_step = 5000 // num_envs
    for i in range(samples // num_envs):
        actions = [env.action_space.sample() for _ in range(num_envs)]
        obs, rewards, dones, info = env.step(actions)

        if i % log_step == 0:
            logging.info("Progress: {}/{}".format(i * num_envs, samples))

    logging.info("Successfully precomputed normalization parameters.")
    env.reset()
    env.training = False
    return env
