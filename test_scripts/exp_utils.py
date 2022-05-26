import random

import numpy as np
import torch

class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_public_attrs(obj):
    return [k for k in dir(obj) if not k.startswith('_')]

