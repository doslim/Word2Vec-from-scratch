# help_function.py
# Define the functions to check the configuration

import yaml
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR

from model import CBOW_Model

def get_model_class(model_name: str):
    '''
    Get model class according to the model name.

    parameter:
    - model_name: str, expected to be "cbow"

    return:
    - CBOW_Model: torch.nn.Module
    '''

    if model_name == "cbow":
        return CBOW_Model
    else:
        raise ValueError("Choose model_name from: cbow")


def get_optimizer_class(opt_name: str):
    '''
    Get the optimizer class according to the name.

    parameter:
    - opt_name: str, expected to be "Adam"

    return:
    - optim.Adam
    '''

    if opt_name == "Adam":
        return optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")


def get_lr_scheduler(optimizer, total_epochs, verbose=True):
    '''
    Get the scheduler to tune the learning rate

    parameter:
    - optimizer: expected to be classes in torch.optim.
    - total_epochs: total number of epochs
    - verbose: whether to print the adjustment of the learning rate

    return:
    - scheduler: torch.nn.optim.lr_scheduler
    '''

    # lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    # lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5, verbose=verbose)
    
    return lr_scheduler


def save_config(config: dict, model_dir: str, embed_size):
    """Save config file to `model_dir` directory"""
    config_path = os.path.join(model_dir, "config_{}.yaml".format(embed_size))
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)


def save_vocab(vocab, model_dir: str, embed_size: int):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab_{}.pt".format(embed_size))
    torch.save(vocab, vocab_path)
