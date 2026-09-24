import random
import numpy as np
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
