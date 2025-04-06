# config/config.py
import json
import os

def load_config(file_name="config.json"):
    config_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(config_path, "r") as f:
        return json.load(f)

def load_test_symbols():
    return load_config("test_symbols.json")

CONFIG = load_config()
TEST_SYMBOLS = load_test_symbols()