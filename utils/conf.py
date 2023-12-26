# Based on https://github.com/aimagelab/mammoth

import random
import torch
import numpy as np

def get_device(GPU=None) -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    gpu = "cuda:"+str(GPU)
    dev = torch.device(gpu if torch.cuda.is_available() else "cpu")
    print("\n\nTraining on GPU no: ", dev)
    print("GPU Name: ", torch.cuda.get_device_name(0))
    return dev
   # return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
