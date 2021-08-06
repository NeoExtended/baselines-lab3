import copy
import importlib
import logging
import os

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    SubprocVecEnv,
    VecNormalize,
    DummyVecEnv,
)

from baselines_lab3.env.wrappers import (
    EvaluationWrapper,
    VecEvaluationWrapper,
    VecImageRecorder,
    VecScaledFloatFrame,
    VecStepSave,
)
from baselines_lab3.env.wrappers.evaluation_wrappers import ParticleInformationWrapper


def make_env(
    env_id, env_kwargs, rank=0, seed=0, log_dir=None, wrappers=None, evaluation=False
):
    """
    Helper function to multiprocess training and log the progress.
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
            for wrapper in wrappers:
                env = wrapper[0](env=env, **wrapper[1])

        if log_dir:
            env = Monitor(
                env, filename=None, allow_early_resets=True
            )  # filename=os.path.join(log_dir, str(rank))
        return env

    return _init


def get_wrapper_class(wrapper):
    """
    Get a Gym environment wrapper class from a string describing the module and class name
    e.g. env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    :param wrapper: (str)
    :return: (gym.Wrapper) A subclass of gym.Wrapper (class object) you can use to create another Gym env
        giving an original env.
    """
    if wrapper:
        wrapper_module = importlib.import_module(wrapper.rsplit(".", 1)[0])
        return getattr(wrapper_module, wrapper.split(".")[-1])
    else:
        return None


def create_environment(config, seed, log_dir=None, video_path=None, evaluation=False):
    """
    Creates a new environment according to the parameters from the given lab config dictionary.
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

    env_id = config.pop("name")
    n_envs = config.pop("n_envs", 1)
    normalize = config.pop("normalize", None)
    frame_stack = config.pop("frame_stack", None)
    multiprocessing = config.pop("multiprocessing", True)

    scale = config.pop("scale", None)
    logging.info(
        "Creating environment with id {} and {} instances.".format(env_id, n_envs)
    )

    # Get tuples with (wrapper_class, wrapper_kwargs)
    wrappers_config = config.pop("wrappers", [])
    wrappers = []
    for wrapper in wrappers_config:
        if isinstance(wrapper, dict):
            wrapper_name = list(wrapper.keys())[0]
            wrappers.append((get_wrapper_class(wrapper_name), wrapper[wrapper_name]))
        elif isinstance(wrapper, str):
            wrappers.append((get_wrapper_class(wrapper), {}))
        else:
            raise ValueError("Got invalid wrapper with value {}".format(str(wrapper)))

    return _create_vectorized_env(
        env_id,
        config,
        n_envs,
        multiprocessing,
        seed,
        log_dir,
        wrappers,
        normalize,
        frame_stack,
        video_path,
        evaluation,
        scale,
        record_images,
        alg_config["name"],
    )


def _create_vectorized_env(
    env_id,
    env_kwargs,
    n_envs,
    multiprocessing,
    seed,
    log_dir,
    wrappers,
    normalize,
    frame_stack,
    video_path,
    evaluation,
    scale,
    buffer_step_data,
    algorithm_name,
):
    env_creation_fns = [
        make_env(env_id, env_kwargs, i, seed, log_dir, wrappers, evaluation=evaluation,)
        for i in range(n_envs)
    ]
    if multiprocessing:
        env = SubprocVecEnv(env_creation_fns)
    else:
        env = DummyVecEnv(env_creation_fns)

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

    return env


def _add_normalization_wrapper(env, n_envs, normalize):
    if isinstance(normalize, bool):
        env = VecNormalize(env)
    elif isinstance(normalize, dict):
        if "trained_agent" in normalize:
            path = normalize.pop("trained_agent")
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
