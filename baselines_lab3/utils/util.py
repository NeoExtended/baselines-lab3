"""
General helper functions.
"""
import collections
import importlib
from datetime import datetime
from pathlib import Path
from subprocess import Popen, PIPE
import logging
from typing import Type, Any, Optional, Dict, List, Union
from collections import abc

import numpy as np

log_dir: Path = Path(".")
TIMESTAMP_FORMAT = "%Y_%m_%d_%H%M%S"


def set_nested_value(
    dictionary: Dict[str, Any], keys: List[str], value
) -> Dict[str, Any]:
    dct = dictionary
    for key in keys[:-1]:
        try:
            dct = dct[key]
        except KeyError:
            return dictionary
    dct[keys[-1]] = value
    return dictionary


def delete_keys_from_dict(dictionary, keys):
    keys_set = set(keys)

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, abc.MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            else:
                modified_dict[
                    key
                ] = value  # or copy.deepcopy(value) if a copy is desired for non-dicts.
    return modified_dict


def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value


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


def get_timestamp(pattern: str = TIMESTAMP_FORMAT) -> str:
    """
    Generates a string timestamp to use for logging.
    :param pattern: (str) Pattern for the timestamp following the python datetime format.
    :return: (str) The current timestamp formated according to pattern
    """
    now = datetime.now()
    return now.strftime(pattern)


def create_log_directory(root: Optional[Union[Path, str]]) -> Path:
    """
    Creates a global log directory at a given place. The directory will be named with a current timestamp.
    :param root: (Path) Parent directory for the log directory. Will be created if it does not exist.
    :return: (Path) Location of the created log directory.
    """
    if not root:
        root = Path(".")
    if isinstance(root, str):
        root = Path(root)

    path = root / get_timestamp()
    path.mkdir(parents=True)
    return set_log_directory(path)


def set_log_directory(log_directory: Path) -> Path:
    """
    Sets the global log directory.
    """
    if log_directory.exists():
        global log_dir
        log_dir = log_directory

        return log_directory
    else:
        raise FileNotFoundError(f"The log directory {log_directory} does not exist!")


def get_log_directory() -> Path:
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


def send_email(receiver, subject, content):
    run_command_line_command(
        'echo "{}" | mail -s "[baselines-lab] {}" {}'.format(content, subject, receiver)
    )


def run_command_line_command(command):
    process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    response = process.communicate()
    return response


def update_dict(d, u):
    """
    Updates dict d to match the parameters of dict u without overwriting lower level keys completely
    :param d: (dict)
    :param u: (dict)
    :return: (dict) The updated dict.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def flatten_dict(d: abc.MutableMapping, separator: str = ".", parent: str = "") -> Dict:
    items = []

    for k, v in d.items():
        new_key = parent + separator + k if parent else k

        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, separator, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d: abc.MutableMapping, separator: str = "."):
    result = {}

    for k, v in d.items():
        parts = k.split(separator)

        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = dict()
            current = current[part]
        current[parts[-1]] = v
    return result
