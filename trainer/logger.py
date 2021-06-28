import logging
import logging.config
from pathlib import Path

from utils import read_json


def setup_logging(
    log_dir,
    log_config="trainer/logger_config.json",
    default_level=logging.INFO,
    is_train=True,
):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                if is_train:
                    handler["filename"] = "{}/{}".format(log_dir, handler["filename"])
                else:
                    handler["filename"] = "{}/{}".format(log_dir, "testing.log")
        logging.config.dictConfig(config)

    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


def get_logger(log_dir, name, verbosity=1, is_train=True):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
        verbosity, log_levels.keys()
    )
    assert verbosity in log_levels, msg_verbosity
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])

    setup_logging(log_dir, is_train=is_train)
    return logger
