# -*- coding: utf-8 -*-

import json
import logging.config
import os

"""
CfiLogger
Utility class for logging.
msgs below warning level will be printed to stdout
msgs of warning and above will be printed to stderr

example:
    # this call will create a logger named after the module, eg "main.py"
    logger = CfiLogger.get_logger(__file__)

    # this call will create a logger named "root" - this can be used in simple scripts or the main script
    logger = CfiLogger.get_logger()

    logger.debug(...)
    logger.info(...)
    logger.error(...)


To override the logging level (for example, use DEBUG level for debugging purposes)
put logger_config.json under the CURRENT WORKING DIRECTORY before the program starts, and follow the steps below

1. to change the logging level of the default/root logger(returned by get_logger()), update "loggers"->""->"level".
2. to change the logging level of a specific logger(returned by get_logger(__file__)), add a section under "loggers"
3. to change the logging level of all loggers, add "all_loggers"->"level"
4. start the program as usual

"""


class BelowWarningFilter:
    def __init__(self, low=logging.DEBUG, high=logging.CRITICAL):
        self.__low = low
        self.__high = high

    def filter(self, record):
        return self.__low <= record.levelno <= self.__high


class CfiLogger:
    __DEFAULT_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s.%(msecs)03d %(name)s [%(process)d] [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "filters": {
            "below_warning": {
                "()": BelowWarningFilter,
                "high": logging.INFO
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "filters": ["below_warning"],
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "standard",
                "stream": "ext://sys.stderr"
            }
        },
        "loggers": {
            "": {
                "handlers": ["stdout", "stderr"],
                "level": "INFO"
            }
        }
    }

    __CONFIG_FILE = os.path.join(os.getcwd(), "logger_config.json")
    __LOADED_OVERRIDE_CONFIG = False
    __LEVEL_OVERRIDE_FOR_ALL = None

    @staticmethod
    def get_logger(name=None):
        logging.config.dictConfig(CfiLogger.__DEFAULT_CONFIG)
        loaded_override = False
        loaded_level_override_for_all = False
        if not CfiLogger.__LOADED_OVERRIDE_CONFIG and os.path.isfile(CfiLogger.__CONFIG_FILE):
            with open(CfiLogger.__CONFIG_FILE) as config_fin:
                config_dict = json.load(config_fin)
                logging.config.dictConfig(config_dict)
                loaded_override = True
                CfiLogger.__LOADED_OVERRIDE_CONFIG = True
                if "all_loggers" in config_dict and "level" in config_dict["all_loggers"]:
                    level = config_dict["all_loggers"]["level"]
                    CfiLogger.__LEVEL_OVERRIDE_FOR_ALL = getattr(logging, level.upper(), None)
                    if not CfiLogger.__LEVEL_OVERRIDE_FOR_ALL:
                        raise RuntimeError(f"Invalid level override [{level}] for all loggers")
                    loaded_level_override_for_all = True

        if name:
            module = os.path.basename(os.path.abspath(name))
            if module.endswith(".py"):
                module = module[:-3]
        else:
            module = None
        logger = logging.getLogger(module)
        if CfiLogger.__LEVEL_OVERRIDE_FOR_ALL:
            logger.setLevel(CfiLogger.__LEVEL_OVERRIDE_FOR_ALL)
        if loaded_override:
            logger.info(f"Loaded logger config override:{CfiLogger.__CONFIG_FILE}")
            if loaded_level_override_for_all and CfiLogger.__LEVEL_OVERRIDE_FOR_ALL:
                logger.info(
                    f"Overriding level for all loggers:{logging.getLevelName(CfiLogger.__LEVEL_OVERRIDE_FOR_ALL)}")
        logger.debug("Instantiated logger")
        return logger
