
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import math
import os
import copy
from tqdm import tqdm

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def convert_model_precision(model,dtype=torch.float32,device='cpu'):
    """
    Convert model parameters between HalfTensor(16) and Tensor(32)
    """
    for p in model.parameters():
        if str(dtype)[-2:] == '16':
            p.data = p.data.type(torch.HalfTensor).to(device)
        if str(dtype)[-2:] == '32':
            p.data = p.data.type(torch.Tensor).to(device)

def module_index_to_name(module_index):
    return module_index.replace('[','.').replace(']','')