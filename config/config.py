# config/config.py
import json
import os
import logging

logger = logging.getLogger(__name__)


def load_config(file_name="config.json"):
    """
    Loads configuration from a JSON file

    Parameters:
    -----------
    file_name : str
        Name of the configuration file to load

    Returns:
    --------
    dict
        Configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        return {}
    except Exception as e:
        logger.exception(f"Error loading configuration: {e}")
        return {}


def load_test_symbols():
    """
    Loads test symbols configuration

    Returns:
    --------
    dict
        Test symbols dictionary
    """
    return load_config("test_symbols.json")


def get_config(section=None):
    """
    Retrieves configuration values from the config file.

    Parameters:
    -----------
    section : str, optional
        Section to retrieve. If None, returns the entire config.

    Returns:
    --------
    dict
        Configuration dictionary or section
    """
    if CONFIG is None:
        logger.warning("Configuration not loaded yet")
        return {}

    if section is None:
        return CONFIG
    elif section in CONFIG:
        return CONFIG[section]
    else:
        logger.warning(f"Section {section} not found in configuration")
        return {}


# Load configurations at module import time
CONFIG = load_config()
TEST_SYMBOLS = load_test_symbols()