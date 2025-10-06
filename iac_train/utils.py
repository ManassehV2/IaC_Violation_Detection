import os
import random
import numpy as np
import torch
from transformers import logging as hf_logging

def init_env():
    hf_logging.set_verbosity_error()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False