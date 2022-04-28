import argparse
import yaml
import os
import torch
import torch.nn as nn

from data_finetune import get_dataloader
from train_finetune import Trainer
from model import CBOW_Model
from help_function import get_lr_scheduler, get_optimizer_class, save_config, save_vocab
from model_finetune import Finetune_Model

def train(config):
    '''
    Function to control the training process
    '''

    cbow = torch.load(config["model_dir"])
    vocab = torch.load(config["data_dir"])
    embeddings = list(cbow.parameters())[0]
    train_loader, val_loader = get_dataloader(vocab)

    vocab_size = vocab.__len__()
    print("Vocabulary size: {}".format(vocab_size))

    model = Finetune_Model(vocab_size=vocab_size,
                       embed_size_1=config['embed_size_1'],
                       embed_size_2=config['embed_size_2'],
                       dropout=config['dropout'],
                       max_norm=0.2,
                       embedding=embeddings)

    criterion = nn.CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished")

    embed_size = config['embed_size_1'] + config['embed_size_2']
    save_config(config, config['model_dir'], embed_size)
    save_vocab(vocab, config['model_dir'], embed_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    train(config)