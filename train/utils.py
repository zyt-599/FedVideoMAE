import os
import yaml
import torch
from torch.optim import AdamW, SGD


def load_config(path: str) -> dict:
    # If it's a relative path, convert to absolute path
    if not os.path.isabs(path):
        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get project root directory (FedVideomae_DP)
        project_root = os.path.dirname(current_dir)
        # Construct absolute path
        abs_path = os.path.join(project_root, path)
    else:
        abs_path = path
    
    with open(abs_path, 'r') as f:
        return yaml.safe_load(f)


def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_adamw(params, lr: float, weight_decay: float):
    # Ensure lr and weight_decay are floats
    lr = float(lr)
    weight_decay = float(weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def make_sgd(params, lr: float, weight_decay: float, momentum: float = 0.9):
    return SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)

