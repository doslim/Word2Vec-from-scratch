import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils import data


class SynonymsDataset(data.Dataset):
    '''
    Expected data shape like: (word_id_1, word_id_2, label)
    '''

    def __init__(self, words, label):
        self.words = words
        self.label = label

    def __getitem__(self, idx):
        return self.words[idx][0], self.words[idx][1], self.label[idx]

    def __len__(self):
        return len(self.words)


def get_synonyms(word):

    synonyms = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return list(set(synonyms))


def get_dataloader(vocab):
    word_synonyms = {}
    for i in tqdm(vocab.get_itos()):
        synonyms = get_synonyms(i)
        if len(synonyms) <= 1:
            continue
        synonyms = [i for i in synonyms if i in vocab]
        if len(synonyms) == 0:
            continue
        word_synonyms[i] = synonyms

    # negative sampling
    word_in_dictionary = set(word_synonyms.keys())

    np.random.seed(21)
    random.seed(21)
    negative_words = {}
    for key in tqdm(word_synonyms.keys()):
        synonyms = set(word_synonyms[key])
        candidate_word = list(word_in_dictionary - synonyms)
        negative = random.sample(candidate_word, len(synonyms))
        negative_words[key] = negative

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []

    random.seed(21)
    for key in tqdm(word_synonyms.keys()):
        synonyms = word_synonyms[key]
        negative = negative_words[key]
        key_idx = vocab([key])[0]

        if len(synonyms) > 4:
            val_word = random.choice(synonyms)
            synonyms.remove(val_word)
            val_data.append([key_idx, vocab([val_word])[0]])
            val_labels.append(1)

            val_word = random.choice(negative)
            negative.remove(val_word)
            val_data.append([key_idx, vocab([val_word])[0]])
            val_labels.append(0)

        for i in synonyms:
            train_data.append([key_idx, vocab([i])[0]])
            train_labels.append(1)

        for i in negative:
            train_data.append([key_idx, vocab([i])[0]])
            train_labels.append(0)

    train_data = torch.tensor(train_data, dtype=torch.long)
    train_labels = torch.tensor(train_labels, dtype=torch.float)

    val_data = torch.tensor(val_data, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.float)

    train_dataset = SynonymsDataset(train_data, train_labels)
    val_dataset = SynonymsDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True)

    return train_loader, val_loader










