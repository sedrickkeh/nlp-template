"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import yaml
from easydict import EasyDict


def create_config(config_file_path, prefix=None):
    """Loads YAML config file as dictionary.
    Args:
        config_file_path (str): Path to config file.
        prefix (str): Config file prefix.
    Returns:
        dict: The config as a dictionary.
    """

    with open(config_file_path, "r") as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()
    for k, v in config.items():
        cfg[k] = v

    # For keys with path in the name, the prefix is prepended.
    # The prefix is typically set to the project root folder when this function is called.
    if prefix is not None:
        for key in cfg:
            if "path" in key:
                cfg[key] = os.path.join(prefix, cfg[key])

    return cfg