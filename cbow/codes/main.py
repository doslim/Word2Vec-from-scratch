# main.py

import argparse
import yaml
import os
import torch
import torch.nn as nn

from dataloader import get_dataloader_and_vocab
from train import Trainer
from help_function import get_model_class, get_optimizer_class, get_lr_scheduler, save_config, save_vocab


def train(config):
    '''
    Function to control the training process
    '''
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        dataset_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        dataset_name=config["dataset"],
        dataset_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
        vocab=vocab,
    )

    vocab_size = vocab.__len__()
    print("Vocabulary size: {}".format(vocab_size))

    model_class = get_model_class(config["model_name"])
    embed_size = int(config["embed_size"])
    model = model_class(vocab_size=vocab_size, embed_size=embed_size)

    criterion = nn.CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished")

    save_config(config, config['model_dir'], embed_size)
    save_vocab(vocab, config['model_dir'], embed_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train(config)