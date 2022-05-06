"""
Defines helper functions for reading and writing the lab config file
"""
import copy
import itertools
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple

import yaml
from gym.utils import seeding

from baselines_lab3.utils import util


def parse_config_args(config_args: List[str], args: Namespace):
    """
    Parses the config string the user provides to the lab and returns the according config dictionaries.
    :param config_args:
    :param args:
    :return:
    """

    configs = []
    for config in config_args:
        path = Path(config)
        assert path.exists(), "Invalid input file/directory {}".format(config)
        if path.is_dir():
            files = path.glob("**/*.yml")
            for file in files:
                configs.extend(create_tests(get_config(file, args)))
        else:
            configs.extend(create_tests(get_config(path, args)))
    return configs


def get_config(config_file: Path, args: Namespace) -> Dict:
    """
    Reads the lab config from a given file and configures it for use with to the current lab mode.
    :param config_file: (Path) Path to the config file.
    :param args: (dict) parsed args dict
    :return: (dict) The parsed config file as dictionary.
    """
    config = read_config(config_file)
    config = resolve_imports(config)
    config = extend_meta_data(config)
    config = clean_config(config, args)
    return config


def create_default_values(config: Dict) -> Dict:
    def set_default(cfg, order, value):
        return util.set_nested_value(cfg, order, value.get("default_value"))

    found_key, ret = config_dfs(config, "default_value", set_default)

    if found_key:
        return ret[-1]
    else:
        return config


def config_dfs(
    config: Dict[str, Any], callback_on: str, callback: Callable
) -> Tuple[bool, List]:
    found_key = False
    stack = list(config.items())
    order = []
    n_children = []
    ret = []
    while stack:
        key, value = stack.pop()
        order.append(key)
        if isinstance(value, dict):
            if value.get(callback_on, False):
                found_key = True
                ret.append(callback(config, order, value))
            else:
                items = list(value.items())
                n_children.append(len(items))
                stack.extend(items)
        else:
            order.pop()
            n_children[-1] = n_children[-1] - 1
            while len(n_children) > 0 and n_children[-1] == 0:
                order.pop()
                n_children.pop()
                if len(n_children) > 0:
                    n_children[-1] = n_children[-1] - 1
    return found_key, ret


def create_tests(config: Dict[str, Any]):
    default_config = util.delete_keys_from_dict(
        create_default_values(copy.deepcopy(config)), ["test_values", "default_value"]
    )

    def create_test(cfg, order, value):
        tests = []
        for v in value["test_values"]:
            cg = copy.deepcopy(default_config)
            tests.append(util.set_nested_value(cg, order, v))
        return tests

    found_test_cases, config_files = config_dfs(config, "test_values", create_test)

    if not found_test_cases:
        return [config]
    else:
        return list(itertools.chain.from_iterable(config_files))  # Unpack nested lists


def resolve_imports(config: Dict) -> Dict:
    """
    Resolves all imports, updating the values in the current config. Existing keys will not be overwritten!
    :param config: (dict) Lab config
    :return: (dict) Lab config with resolved import statements.
    """
    complete_config = {}
    for c in config.get("import", []):
        complete_config = util.update_dict(complete_config, read_config(Path(c)))

    config = util.update_dict(complete_config, config)
    return config


def clean_config(config: Dict, args: Namespace) -> Dict:
    """
    Deletes or modifies keys from the config which are not compatible with the current lab mode.
    :param config: (dict) The config dictionary
    :param args: (dict) parsed args dict
    :return: (dict) The cleaned config dictionary
    """

    if args.lab_mode == "enjoy":
        return _clean_enjoy_config(args, config)
    elif args.lab_mode == "train":
        return _clean_train_config(args, config)
    elif args.lab_mode == "search":
        return _clean_search_config(args, config)


def _clean_search_config(args, config):
    resume = config["search"].get("resume", False) or args.resume
    if resume and isinstance(resume, bool):
        from baselines_lab3.model.callbacks import CheckpointManager

        path = CheckpointManager.get_latest_run(config["meta"]["log_dir"])
        config["search"]["resume"] = path
    return config


def _clean_train_config(args, config):
    # Allow fast loading of recently trained agents via "last" and "best" checkpoints
    if args.resume:
        config["algorithm"]["trained_agent"] = "last"

    if config["algorithm"].get("trained_agent", None):
        if config["algorithm"]["trained_agent"] in ["best", "last"]:
            from baselines_lab3.model.callbacks import CheckpointManager

            checkpoint_type = config["algorithm"]["trained_agent"]

            path = CheckpointManager.get_latest_run(config["meta"]["log_dir"])
            set_checkpoints(config, path, checkpoint_type, args.trial)
        else:
            if config["env"].get("normalize", False):
                if not config["env"]["normalize"].get("trained_agent", False):
                    logging.warning(
                        "Loading from a specific save-file requires setting a trained_agent for the normalization wrapper."
                    )
    if config["algorithm"]["name"] in ["dqn", "ddpg"]:
        if config["env"].get("n_envs", 1) > 1:
            logging.warning(
                "Number of envs must be 1 for dqn and ddpg! Reducing n_envs to 1."
            )
            config["env"]["n_envs"] = 1
    return config


def _clean_enjoy_config(args, config):
    # Do not change running averages in enjoy mode
    if "normalize" in config["env"] and config["env"]["normalize"]:
        if isinstance(config["env"]["normalize"], bool):
            config["env"].pop("normalize")
            config["env"]["normalize"] = {"training": False}
        else:
            config["env"]["normalize"]["training"] = False
    # Find checkpoints
    if len(args.checkpoint_path) > 0:
        config["meta"]["session_dir"] = args.checkpoint_path
        set_checkpoints(config, args.checkpoint_path, args.type, args.trial)
    else:
        from baselines_lab3.model.callbacks import CheckpointManager

        path = CheckpointManager.get_latest_run(config["meta"]["log_dir"])
        config["meta"]["session_dir"] = path
        set_checkpoints(config, path, args.type, args.trial)
    # Reduce number of envs if there are too many
    if config["env"]["n_envs"] > 32:
        config["env"]["n_envs"] = 32
    if args.strict:
        config["env"]["n_envs"] = 1
    return config


def set_checkpoints(config, path, type, trial=-1):
    from baselines_lab3.model.callbacks import CheckpointManager

    normalization = "normalize" in config["env"] and config["env"]["normalize"]

    checkpoint = CheckpointManager.get_checkpoint(path, type=type, trial=trial)
    config["algorithm"]["trained_agent"] = checkpoint["model"]

    if normalization:
        config["env"]["normalize"]["trained_agent"] = checkpoint["normalization"]


def read_config(config_file: Path) -> Dict:
    """
    Reads a config file from disc. The file must follow JSON or YAML standard.
    :param config_file: (Path) Path to the config file.
    :return: (dict) A dict with the contents of the file.
    """
    file = config_file.open("r")
    ext = os.path.splitext(config_file)[-1]

    if ext == ".json":
        config = json.load(file)
    elif ext == ".yml":
        config = yaml.safe_load(file)
    else:
        raise NotImplementedError("File format unknown")

    file.close()
    return config


def save_config(config: Dict, path: Path):
    """
    Saves a given lab configuration to a file.
    :param config: (dict) The lab configuration.
    :param path: (Path) Desired file location.
    """
    ext = os.path.splitext(path)[-1]
    file = path.open("w")
    if ext == ".json":
        json.dump(config, file, indent=2, sort_keys=True)
    elif ext == ".yml":
        yaml.dump(config, file, indent=2)
    file.close()


def seed_from_config(config, increment=0, max_bytes=4):
    """
    Cretaes a new generated seed for the given config. Creates a random seed if config["meta"]["seed"] is None.
    :param max_bytes:
    :param config:
    :param increment: Number that is added to the seed in config.
    :return:
    """
    seed = config["meta"].get("seed", None)
    if seed is not None:
        seed = seed + increment

    return seeding.create_seed(max_bytes=max_bytes, a=seed)


def extend_meta_data(config: Dict) -> Dict:
    """
    Extends the meta-data dictionary of the file to save additional information at training time.
    :param config: (dict) The config dictionary.
    :return: (dict) The updated config dictionary.
    """
    seed = seed_from_config(config)

    extended_info = {
        "timestamp": util.get_timestamp(),
        "generated_seed": seed,
    }
    config["meta"].update(extended_info)
    return config
